from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import argparse
import csv
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sparse
import scipy.stats as stats


METADATA_COLUMNS = {"case_id", "project", "survival_time", "event"}
MODEL_ORDER = ("APPNP-linear", "SGC-linear", "Linear-sem-grafo", "RBF-SVM")


def parse_seeds(value: str) -> list[int]:
    """Aceita ``inicio:fim`` (inclusive) ou uma lista separada por virgulas."""
    if ":" in value:
        start, end = (int(part) for part in value.split(":", maxsplit=1))
        if end < start:
            raise ValueError("a seed final deve ser maior ou igual a inicial")
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def ensure_patient_table(
    target_file: Path,
    bio_file: Path,
    clinical_file: Path,
    project: str,
) -> None:
    """Cria o Parquet com o pipeline original somente se ele estiver ausente."""
    if target_file.exists():
        return
    if not Path("rna_files").is_dir():
        raise FileNotFoundError(
            f"{target_file} nao existe e a pasta rna_files/ nao foi encontrada. "
            "Gere patients_table pelo pipeline original ou informe seu caminho "
            "com --target-file."
        )

    from filter import create_project_map_to_parquet

    print(f"{target_file} nao encontrado; criando a partir de rna_files/...")
    bio = pl.read_csv(bio_file, separator="\t")
    clinical = pl.read_csv(clinical_file, separator="\t")
    result = create_project_map_to_parquet(
        bio, clinical, [project], str(target_file)
    )
    if result is None or not target_file.exists():
        raise RuntimeError(f"nao foi possivel criar {target_file}")


def load_five_year_endpoint(
    target_file: Path,
    project: str,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega a expressao e cria um alvo binario com status conhecido."""
    table = pl.read_parquet(target_file)
    if "project" in table.columns:
        table = table.filter(pl.col("project") == project)
    if table.is_empty():
        raise ValueError(f"nenhum paciente encontrado para {project}")
    if "case_id" in table.columns:
        # Torna as seeds independentes da ordem produzida pelo group_by do Polars.
        table = table.sort("case_id")

    genes = [column for column in table.columns if column not in METADATA_COLUMNS]
    if not genes:
        raise ValueError(
            "patients_table nao possui colunas de expressao genica")

    # CORRECAO 1: o alvo original ``event`` confunde censura precoce com
    # ausencia de obito. Aqui so mantemos pacientes cujo status aos cinco anos
    # e conhecido.
    survival = table["survival_time"].to_numpy().astype(np.float64)
    event = table["event"].to_numpy().astype(np.int8)
    died_before_horizon = (event == 1) & (survival < horizon_days)
    survived_horizon = survival >= horizon_days
    known_outcome = died_before_horizon | survived_horizon

    # No layout do GDC, gene ausente para um paciente equivale a contagem zero.
    expression = (
        table.filter(pl.Series(known_outcome))
        .select(genes)
        .fill_null(0)
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    np.nan_to_num(expression, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(expression < 0):
        raise ValueError("a matriz de expressao contem valores negativos")
    np.log1p(expression, out=expression)

    labels = died_before_horizon[known_outcome].astype(np.int8)
    return expression, labels


def select_and_scale(
    expression: np.ndarray,
    labelled: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    """Seleciona genes e ajusta o scaler usando apenas os rotulados."""
    if expression.shape[1] < n_genes:
        raise ValueError(
            f"o experimento requer {n_genes} genes, mas patients_table possui "
            f"somente {expression.shape[1]}"
        )
    # CORRECAO 2: selecao e scaler sao ajustados DEPOIS do split e somente nos
    # pacientes rotulados. No pipeline original, o grafo/scaler era preparado
    # antes dos splits usados na comparacao.
    # CORRECAO 3: a mesma lista de n_genes e usada por APPNP, SGC, linear e SVM.
    variances = np.var(expression[labelled], axis=0, dtype=np.float64)
    selected = np.argsort(-variances, kind="stable")[:n_genes]
    selected_expression = expression[:, selected]
    scaler = StandardScaler().fit(selected_expression[labelled])
    return scaler.transform(selected_expression).astype(np.float32)


def symmetric_normalization(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Adiciona self-loops e calcula D^(-1/2) A D^(-1/2)."""
    adjacency = adjacency + sparse.eye(
        adjacency.shape[0], dtype=np.float32, format="csr"
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = sparse.diags(np.power(degree, -0.5), format="csr")
    return (inverse_sqrt @ adjacency @ inverse_sqrt).tocsr()


def build_patient_graph(features: np.ndarray, k: int = 3) -> sparse.csr_matrix:
    """kNN cosseno, uniao, similaridade RBF local e normalizacao simetrica."""
    search = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", n_jobs=-1
    ).fit(features)
    distances, indices = search.kneighbors(features)
    distances, indices = distances[:, 1:], indices[:, 1:]

    rows = np.repeat(np.arange(len(features)), k)
    columns = indices.reshape(-1)
    local_scale = np.maximum(distances[:, -1], 1e-8)
    denominator = local_scale[rows] * local_scale[columns]
    # CORRECAO 4: graph.py usava a distancia como edge_weight, de modo que um
    # vizinho mais distante recebia peso maior. A RBF abaixo converte distancia
    # em similaridade: quanto mais proximo, maior o peso.
    weights = np.exp(
        -np.square(distances.reshape(-1)) / np.maximum(denominator, 1e-12)
    ).astype(np.float32)

    directed = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(len(features), len(features))
    )
    return symmetric_normalization(directed.maximum(directed.T))


def propagate_sgc(
    features: np.ndarray,
    adjacency: sparse.csr_matrix,
    hops: int = 3,
) -> np.ndarray:
    """SGC canonico: A_norm^K X, sem camada escondida e sem ReLU."""
    # CORRECAO 5: substitui SGConv -> ReLU -> Linear do graph.py. O SGC
    # canonico faz apenas a propagacao fixa e depois usa uma cabeca linear.
    propagated = features
    for _ in range(hops):
        propagated = adjacency @ propagated
    return np.asarray(propagated)


def propagate_appnp(
    features: np.ndarray,
    adjacency: sparse.csr_matrix,
    alpha: float = 0.1,
    hops: int = 10,
) -> np.ndarray:
    """APPNP linear: propagacao com reinicio seguida por cabeca linear."""
    # CORRECAO 6: adiciona reinicio alpha para reduzir oversmoothing, mantendo
    # o preditor simples e robusto no regime de apenas 65 rotulos.
    initial = features
    propagated = initial
    for _ in range(hops):
        propagated = (
            (1.0 - alpha) * (adjacency @ propagated) + alpha * initial
        )
    return np.asarray(propagated)


def linear_head(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=0.01,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )


def evaluate(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
) -> tuple[float, float]:
    prediction = model.predict(features[test])
    score = model.decision_function(features[test])
    return (
        float(balanced_accuracy_score(labels[test], prediction)),
        float(roc_auc_score(labels[test], score)),
    )


def run(args: argparse.Namespace) -> list[dict]:
    ensure_patient_table(
        args.target_file, args.bio_file, args.clinical_file, args.project
    )
    expression, labels = load_five_year_endpoint(
        args.target_file, args.project, args.horizon_days
    )
    seeds = parse_seeds(args.seeds)
    all_indices = np.arange(len(labels))
    rows: list[dict] = []

    print(
        f"Coorte: {len(labels)} pacientes; {int(labels.sum())} positivos; "
        f"{len(labels) - int(labels.sum())} negativos"
    )
    print(
        f"Protocolo: {args.train_fraction:.0%} rotulados, {args.genes} genes, "
        f"{len(seeds)} seeds; grafo transdutivo sem rotulos de teste"
    )

    for position, seed in enumerate(seeds, start=1):
        # CORRECAO 2: o split vem primeiro. Nenhum rotulo de ``test`` entra na
        # selecao de genes, no scaler ou no ajuste dos classificadores.
        labelled, test = train_test_split(
            all_indices,
            train_size=args.train_fraction,
            stratify=labels,
            random_state=seed,
        )
        features = select_and_scale(expression, labelled, args.genes)

        # O grafo e transdutivo: atributos de todos os pacientes definem a
        # topologia, mas os rotulos de teste nunca sao consultados.
        adjacency = build_patient_graph(features, args.graph_k)
        appnp_features = propagate_appnp(
            features,
            adjacency,
            alpha=args.appnp_alpha,
            hops=args.appnp_hops,
        )
        sgc_features = propagate_sgc(
            features,
            adjacency,
            hops=args.sgc_hops,
        )

        # CORRECAO 3: os quatro modelos partem exatamente do mesmo split e dos
        # mesmos 400 genes. ``Linear-sem-grafo`` e a ablacao A=I, necessaria
        # para separar o efeito do grafo do efeito da cabeca linear.
        models_and_features = {
            "APPNP-linear": (
                linear_head(seed).fit(
                    appnp_features[labelled], labels[labelled]
                ),
                appnp_features,
            ),
            "SGC-linear": (
                linear_head(seed).fit(
                    sgc_features[labelled], labels[labelled]),
                sgc_features,
            ),
            "Linear-sem-grafo": (
                linear_head(seed).fit(features[labelled], labels[labelled]),
                features,
            ),
            "RBF-SVM": (
                # Parametros escolhidos nas seeds de desenvolvimento, antes do
                # bloco de confirmacao 800--899.
                SVC(
                    C=1.0,
                    gamma=0.001,
                    kernel="rbf",
                    class_weight="balanced",
                    cache_size=2000,
                    random_state=seed,
                ).fit(features[labelled], labels[labelled]),
                features,
            ),
        }

        seed_scores = {}
        for model_name in MODEL_ORDER:
            model, model_features = models_and_features[model_name]
            balanced_accuracy, auc = evaluate(
                model, model_features, labels, test
            )
            seed_scores[model_name] = (balanced_accuracy, auc)
            rows.append(
                {
                    "seed": seed,
                    "n_labelled": len(labelled),
                    "n_test": len(test),
                    "model": model_name,
                    "balanced_accuracy": balanced_accuracy,
                    "auc": auc,
                }
            )

        print(
            f"seed {position:02d}/{len(seeds)} | "
            + " | ".join(
                f"{name}: BA={seed_scores[name][0]:.4f}, "
                f"AUC={seed_scores[name][1]:.4f}"
                for name in MODEL_ORDER
            )
        )
    return rows


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    margin = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return float(values.mean() - margin), float(values.mean() + margin)


def print_summary(rows: list[dict]) -> None:
    # CORRECAO 7: reporta distribuicao entre seeds e comparacoes pareadas, em
    # vez de concluir a partir de uma unica divisao treino/teste.
    print("\nResumo (media +/- desvio-padrao amostral)")
    by_model = {
        model: [row for row in rows if row["model"] == model]
        for model in MODEL_ORDER
    }
    for model, model_rows in by_model.items():
        balanced_accuracy = np.asarray(
            [row["balanced_accuracy"] for row in model_rows]
        )
        auc = np.asarray([row["auc"] for row in model_rows])
        print(
            f"{model:17s} | BA {balanced_accuracy.mean():.4f} +/- "
            f"{balanced_accuracy.std(ddof=1):.4f} | "
            f"AUC {auc.mean():.4f} +/- {auc.std(ddof=1):.4f}"
        )

    print("\nComparacoes pareadas (intervalos descritivos entre holdouts)")
    for candidate in ("APPNP-linear", "SGC-linear"):
        for baseline in ("RBF-SVM", "Linear-sem-grafo"):
            for metric, label in (("balanced_accuracy", "BA"), ("auc", "AUC")):
                candidate_values = np.asarray(
                    [row[metric] for row in by_model[candidate]]
                )
                baseline_values = np.asarray(
                    [row[metric] for row in by_model[baseline]]
                )
                difference = candidate_values - baseline_values
                low, high = confidence_interval(difference)
                print(
                    f"{candidate} - {baseline} em {label}: "
                    f"{difference.mean()                       :+.4f}; IC95% [{low:+.4f}, {high:+.4f}]; "
                    f"vitorias {int((difference > 0).sum())}/{len(difference)}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-file", type=Path,
                        default=Path("patients_table"))
    parser.add_argument("--bio-file", type=Path,
                        default=Path("bio_filtrado.tsv"))
    parser.add_argument(
        "--clinical-file", type=Path, default=Path("clinical_v2_fixed.tsv")
    )
    parser.add_argument("--project", default="TARGET-AML")
    parser.add_argument("--horizon-days", type=int, default=5 * 365)
    parser.add_argument("--train-fraction", type=float, default=0.05)
    parser.add_argument("--genes", type=int, default=400)
    parser.add_argument("--seeds", default="800:899")
    parser.add_argument("--output", type=Path,
                        default=Path("few_labels_gnn.csv"))

    # Configuracoes fixadas no bloco de desenvolvimento, antes das seeds acima.
    parser.add_argument("--graph-k", type=int, default=3)
    parser.add_argument("--appnp-alpha", type=float, default=0.1)
    parser.add_argument("--appnp-hops", type=int, default=10)
    parser.add_argument("--sgc-hops", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"\nResultados por seed: {args.output}")


if __name__ == "__main__":
    main()
