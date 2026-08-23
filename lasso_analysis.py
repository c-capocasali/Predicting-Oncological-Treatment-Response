import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance


def lasso(target_file):

    # Carrega arquivo parquet
    df = pl.read_parquet(target_file)

    target_cols = ['survival_time', 'event', 'case_id', 'project']

    # Seleciona apenas colunas numéricas (evita o erro do log1p com case_id/project)
    X = df.select(cs.numeric()).drop(target_cols, strict=False).to_pandas()

# Array estruturado é exigido pela função do Cox Lasso
    y = np.array(
        list(zip(df['event'].cast(pl.Boolean).to_numpy(),
                 df['survival_time'].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

# Separação em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['event'].to_numpy())

# --- Pré-processamento pesado FORA do Pipeline/CV (feito uma única vez) ---
# Log(x+1): igual ao benchmark, aplicado antes do filtro de variância para não
# distorcer a seleção a favor de genes de alta expressão absoluta (ribossomais/
# mitocondriais). Rodar aqui evita recalcular isso a cada fold/alpha do CV.
    X_train_log = np.log1p(X_train.to_numpy())
    X_test_log = np.log1p(X_test.to_numpy())

# Filtro de Variância (ajuste o threshold se quiser remover mais ou menos genes)
    var_filter = VarianceThreshold(threshold=0.5)
    X_train_var = var_filter.fit_transform(X_train_log)
    X_test_var = var_filter.transform(X_test_log)
    genes_pos_variancia = X_train.columns[var_filter.get_support()]

# Pipeline Batedor: Descobre a faixa de alphas ideal para esses dados
    pipeline_batedor = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0))
    ])
    pipeline_batedor.fit(X_train_var, y_train)
    alphas_estimados = pipeline_batedor.named_steps['cox_lasso'].alphas_

# Prepara a grade para testar os alphas um por um
    param_grid = {'cox_lasso__alphas': [[a] for a in alphas_estimados]}

# Pipeline Final (com fit_baseline_model=True)
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True))
    ])

# K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_train_var, y_train['Status']))

# GridSearch configurado com error_score=0.5
    gcv = GridSearchCV(
        pipeline_final,
        param_grid=param_grid,
        cv=cv_splits,
        n_jobs=-1,
        error_score=0.5
    )

    gcv.fit(X_train_var, y_train)

    melhor_alpha = gcv.best_params_['cox_lasso__alphas'][0]
    c_index_val = gcv.best_score_
    print(f"\n[Treino/CV] Melhor Alpha encontrado: {melhor_alpha:.6f}")
    print(f"[Treino/CV] C-Index médio na Validação Cruzada: {c_index_val:.4f}")

# O método .score() do scikit-survival calcula automaticamente o Harrell's C-index
    c_index_teste = gcv.score(X_test_var, y_test)

# Extração dos genes
    # 1. Recupera os coeficientes do modelo Lasso
    melhor_modelo_cox = gcv.best_estimator_.named_steps['cox_lasso']
    coeficientes_finais = melhor_modelo_cox.coef_[:, 0]
    # 2. Filtra apenas os que o Lasso não zerou, usando os nomes já filtrados pela variância
    genes_selecionados = genes_pos_variancia[coeficientes_finais != 0].tolist()

    print(f"\nTotal de genes sobreviventes ao Lasso: {
          len(genes_selecionados)}")

    return genes_selecionados, c_index_teste


def train_test_case_ids(target_file: str, test_size: float = 0.2, random_state: int = 42) -> tuple[list, list]:
    """
    Faz o ÚNICO split treino/teste (por case_id) que deve ser reaproveitado
    em todo o pipeline: seleção de genes (Lasso/RSF), construção do grafo e
    avaliação das GNNs/baselines. Isso evita que a seleção de genes "veja"
    (mesmo que indiretamente, via correlação gene x evento) pacientes que
    depois serão usados como conjunto de teste da GNN/baseline.
    """
    df = pl.read_parquet(target_file)
    train_ids, test_ids = train_test_split(
        df["case_id"].to_list(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["event"].to_list(),
    )
    return train_ids, test_ids


def _prepare_survival_data(target_file, case_id_filter: list | None = None):
    """
    Reproduz exatamente o pré-processamento comum usado em lasso() (colunas
    numéricas, array estruturado de sobrevivência, split treino/teste, log1p
    e filtro de variância), para ser reaproveitado pelas curvas de C-index
    do Lasso e do Random Survival Forest sem duplicar/alterar a lógica original.

    Se case_id_filter for passado (ex: os case_ids de treino do split externo
    único de train_test_case_ids), a tabela é restrita a esses pacientes ANTES
    de qualquer outra coisa. Assim, o split treino/teste interno feito aqui
    fica inteiramente contido dentro do conjunto de treino externo, e nunca
    toca nos pacientes reservados para o teste final da GNN/baseline.
    """
    df = pl.read_parquet(target_file)
    if case_id_filter is not None:
        df = df.filter(pl.col("case_id").is_in(case_id_filter))

    target_cols = ['survival_time', 'event', 'case_id', 'project']
    X = df.select(cs.numeric()).drop(target_cols, strict=False).to_pandas()

    y = np.array(
        list(zip(df['event'].cast(pl.Boolean).to_numpy(),
                 df['survival_time'].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['event'].to_numpy())

    X_train_log = np.log1p(X_train.to_numpy())
    X_test_log = np.log1p(X_test.to_numpy())

    var_filter = VarianceThreshold(threshold=0.5)
    X_train_var = var_filter.fit_transform(X_train_log)
    X_test_var = var_filter.transform(X_test_log)
    genes_pos_variancia = X_train.columns[var_filter.get_support()]

    return X_train_var, X_test_var, y_train, y_test, genes_pos_variancia


def _subsample_sorted(values: list[int], max_points: int) -> list[int]:
    """
    Reduz uma lista de valores (já ordenada, sem duplicatas) para no máximo
    max_points, escolhendo índices igualmente espaçados. Usado para limitar
    o custo de treinar um RSF por ponto do eixo X.
    """
    values = sorted(set(values))
    if len(values) <= max_points:
        return values
    idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [values[i] for i in idx]


def lasso_path_curve(target_file: str, n_points: int = 15, case_id_filter: list | None = None) -> dict:
    """
    Percorre o path de alphas do Cox Lasso (mesmo path já calculado pelo
    'pipeline_batedor' em lasso()) e, para cada alpha, mede quantos genes
    sobrevivem e qual o C-index no teste. n_points controla quantos alphas
    do path são de fato avaliados (subamostragem igualmente espaçada), para
    não refazer o fit para cada um dos ~100 alphas do path completo.

    case_id_filter: se passado, restringe a seleção de genes a esses
    pacientes (tipicamente os train_ids de train_test_case_ids), para não
    vazar informação do conjunto de teste externo para dentro da seleção.

    Retorna um dict {n_genes: {"c_index": float, "genes": list[str]}}.
    """
    X_train_var, X_test_var, y_train, y_test, genes_pos_variancia = _prepare_survival_data(
        target_file, case_id_filter=case_id_filter)

    # Mesmo "pipeline batedor" de lasso(), só para obter o path de alphas
    pipeline_batedor = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0))
    ])
    pipeline_batedor.fit(X_train_var, y_train)
    alphas_estimados = pipeline_batedor.named_steps['cox_lasso'].alphas_

    alphas_avaliados = _subsample_sorted(
        list(range(len(alphas_estimados))), n_points)

    curve = {}
    for i in alphas_avaliados:
        alpha = alphas_estimados[i]
        pipeline_final = Pipeline([
            ('scaler', StandardScaler()),
            ('cox_lasso', CoxnetSurvivalAnalysis(
                l1_ratio=1.0, alphas=[alpha], fit_baseline_model=True))
        ])

        # Mesma tolerância a erro numérico que o GridSearchCV (error_score=0.5)
        # já dava em lasso() — alphas muito baixos podem estourar numericamente,
        # especialmente com conjuntos de treino menores (ex: só o split de treino
        # externo). Em vez de propagar o erro, pula esse alpha.
        try:
            pipeline_final.fit(X_train_var, y_train)
        except ArithmeticError as e:
            print(f"[Lasso] Alpha {alpha:.6f} ignorado (erro numérico: {e})")
            continue

        coeficientes = pipeline_final.named_steps['cox_lasso'].coef_[:, 0]
        genes_selecionados = genes_pos_variancia[coeficientes != 0].tolist()
        n_genes = len(genes_selecionados)

        if n_genes == 0:
            continue

        c_index = pipeline_final.score(X_test_var, y_test)

        # Se dois alphas do path derem o mesmo n_genes, mantém o de maior C-index
        if n_genes not in curve or c_index > curve[n_genes]["c_index"]:
            curve[n_genes] = {"c_index": float(
                c_index), "genes": genes_selecionados}

    print(f"\n[Lasso] Curva C-index x n_genes calculada em {len(curve)} pontos.")
    return curve


def rsf_feature_curve(
    target_file: str,
    n_features_list: list[int],
    n_estimators: int = 200,
    n_repeats_importance: int = 5,
    case_id_filter: list | None = None,
) -> dict:
    """
    Ranqueia os genes (pós filtro de variância) por importância de permutação
    de um Random Survival Forest treinado uma única vez com todos eles, e
    depois, para cada n em n_features_list, refaz o treino do RSF apenas com
    os top-n genes e mede o C-index no teste. Os n testados costumam ser os
    mesmos que saíram de lasso_path_curve, para permitir comparação direta
    no mesmo eixo X.

    case_id_filter: mesmo papel de lasso_path_curve — restringe a seleção
    de genes a esses pacientes (tipicamente os train_ids do split externo).

    Retorna um dict {n_genes: {"c_index": float, "genes": list[str]}}.
    """
    X_train_var, X_test_var, y_train, y_test, genes_pos_variancia = _prepare_survival_data(
        target_file, case_id_filter=case_id_filter)

    # Treina uma vez com todos os genes só para calcular a importância por permutação
    # (RandomSurvivalForest não tem feature_importances_ nativo baseado em impureza)
    rsf_full = RandomSurvivalForest(
        n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rsf_full.fit(X_train_var, y_train)

    importancia = permutation_importance(
        rsf_full, X_train_var, y_train,
        n_repeats=n_repeats_importance, random_state=42, n_jobs=-1
    )
    ranking = np.argsort(importancia.importances_mean)[::-1]

    n_values = _subsample_sorted(
        [n for n in n_features_list if 0 < n <= len(genes_pos_variancia)],
        max_points=len(n_features_list)
    )

    curve = {}
    for n in n_values:
        idx = ranking[:n]
        genes_selecionados = genes_pos_variancia[idx].tolist()

        rsf_sub = RandomSurvivalForest(
            n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rsf_sub.fit(X_train_var[:, idx], y_train)
        c_index = rsf_sub.score(X_test_var[:, idx], y_test)

        curve[n] = {"c_index": float(c_index), "genes": genes_selecionados}

    print(f"\n[RSF] Curva C-index x n_genes calculada em {len(curve)} pontos.")
    return curve


def plot_c_index_curves(lasso_curve: dict, rsf_curve: dict, save_path: str | None = None):
    """
    Plota as duas curvas de C-index (teste) x número de genes/features,
    uma para o Cox Lasso e outra para o Random Survival Forest.
    """
    import matplotlib.pyplot as plt

    n_lasso = sorted(lasso_curve.keys())
    c_lasso = [lasso_curve[n]["c_index"] for n in n_lasso]

    n_rsf = sorted(rsf_curve.keys())
    c_rsf = [rsf_curve[n]["c_index"] for n in n_rsf]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_lasso, c_lasso, marker='o', label='Cox Lasso')
    ax.plot(n_rsf, c_rsf, marker='o', label='Random Survival Forest')
    ax.set_xlabel('Número de genes (features)')
    ax.set_ylabel('C-index (teste)')
    ax.set_title('C-index x número de features')
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def best_gene_set(lasso_curve: dict, rsf_curve: dict) -> tuple[str, int, float, list[str]]:
    """
    Compara os melhores pontos (maior C-index de teste) das duas curvas e
    devolve qual modelo venceu, o n_genes correspondente, o C-index e a
    lista de genes a ser usada no restante do pipeline (grafo KNN + GNNs +
    baselines clássicos).

    Retorna: (nome_modelo, n_genes, c_index, genes_selecionados)
    """
    melhor_lasso_n = max(
        lasso_curve, key=lambda n: lasso_curve[n]["c_index"]) if lasso_curve else None
    melhor_rsf_n = max(
        rsf_curve, key=lambda n: rsf_curve[n]["c_index"]) if rsf_curve else None

    candidatos = []
    if melhor_lasso_n is not None:
        candidatos.append(
            ("Cox Lasso", melhor_lasso_n, lasso_curve[melhor_lasso_n]["c_index"],
             lasso_curve[melhor_lasso_n]["genes"]))
    if melhor_rsf_n is not None:
        candidatos.append(
            ("Random Survival Forest", melhor_rsf_n, rsf_curve[melhor_rsf_n]["c_index"],
             rsf_curve[melhor_rsf_n]["genes"]))

    if not candidatos:
        raise ValueError(
            "Nenhuma das duas curvas produziu pontos válidos; verifique os dados de entrada.")

    nome, n_genes, c_index, genes = max(candidatos, key=lambda c: c[2])
    print(f"\nMelhor modelo de seleção de genes: {nome} "
          f"({n_genes} genes, C-index teste = {c_index:.4f})")

    return nome, n_genes, c_index, genes
