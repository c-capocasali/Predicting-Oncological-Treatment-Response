"""Grafo de pacientes e redes neurais de grafos (GNNs).

O módulo cobre duas etapas do pipeline:

1. **Construção do grafo** (:func:`build_patient_knn_graph`): cada paciente é
   um vértice e as arestas ligam cada paciente aos seus ``k`` vizinhos mais
   próximos no espaço de expressão gênica;
2. **Classificação do desfecho** (:func:`predict_event_gnn`): treina GCN e
   SGC sobre esse grafo para prever ``event`` (0 = sem óbito registrado,
   1 = óbito), com validação cruzada interna e repetição em várias seeds.

Pré-processamento aplicado às features antes do k-NN: imputação pela média
do gene, ``log1p`` e padronização (``StandardScaler``). O mesmo vetor de
features é reaproveitado como ``Data.x`` das GNNs.

Ressalva importante sobre os pesos: nesta versão, o atributo ``weight`` de
cada aresta é a **distância euclidiana** entre os pacientes. O ``GCNConv``
interpreta esse valor como coeficiente de agregação, de modo que vizinhos
mais distantes acabam recebendo peso maior. A versão experimental da branch
``train_test_split`` substitui a distância por uma similaridade RBF local;
veja a seção "Limitações" do README.
"""

import networkx as nx
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SGConv


def build_patient_knn_graph(
    target_file: str, selected_genes: list[str], k: int = 5
) -> nx.Graph:
    """Constrói um grafo k-NN não direcionado em que cada nó é um paciente.

    Args:
        target_file: caminho do Parquet com uma linha por paciente, contendo
            ``survival_time``, ``event``, opcionalmente ``case_id`` e as
            colunas de expressão listadas em ``selected_genes``.
        selected_genes: genes usados como features (normalmente a saída de
            :func:`lasso_analysis.lasso`).
        k: número de vizinhos de cada paciente (o próprio paciente é
            excluído da contagem).

    Returns:
        ``networkx.Graph`` com atributos de nó ``genes`` (lista de floats já
        imputados, em ``log1p`` e padronizados), ``survival_time`` e
        ``event``; atributo de aresta ``weight`` (distância euclidiana).
    """
    df = pl.read_parquet(target_file)

    # O case_id é o identificador natural do vértice; se ele não existir, usa
    # a posição da linha no Parquet.
    has_case_id = "case_id" in df.columns
    cols_to_select = (
        ["case_id", "survival_time", "event"]
        if has_case_id
        else ["survival_time", "event"]
    ) + selected_genes
    df_filtered = df.select(cols_to_select)

    n_patients = len(df_filtered)
    if n_patients <= k:
        raise ValueError(
            f"k={k} precisa ser menor que o número de pacientes ({n_patients})."
        )

    # 1. Matriz de features para o k-NN (apenas genes).
    X_genes = df_filtered.select(selected_genes).to_numpy()

    # 2. Tratamento de nulls: imputação pela média do gene (ignora NaNs).
    if np.isnan(X_genes).any():
        col_means = np.nanmean(X_genes, axis=0)
        nan_idx = np.where(np.isnan(X_genes))
        X_genes[nan_idx] = np.take(col_means, nan_idx[1])

    # 3. Mesma transformação usada na seleção por Lasso, para manter o espaço
    #    de features consistente.
    X_genes = np.log1p(X_genes)
    X_genes = StandardScaler().fit_transform(X_genes)

    # 4. k + 1 porque o vizinho mais próximo de um ponto é ele mesmo.
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X_genes)
    distances, indices = nn.kneighbors(X_genes)

    G = nx.Graph()
    node_ids = (
        df_filtered["case_id"].to_list()
        if has_case_id
        else list(range(n_patients))
    )

    # 5. Vértices: guarda as features já tratadas para a conversão ao PyG.
    for i, row in enumerate(df_filtered.iter_rows(named=True)):
        G.add_node(
            node_ids[i],
            genes=X_genes[i].tolist(),
            survival_time=row["survival_time"],
            event=row["event"],
        )

    # 6. Arestas: ignora a posição 0 (o próprio paciente).
    for i, neighbors in enumerate(indices):
        source_patient = node_ids[i]

        for j in range(1, k + 1):
            target_patient = node_ids[neighbors[j]]
            edge_weight = distances[i][j]
            # Em um nx.Graph, adicionar A→B e depois B→A não duplica a aresta.
            G.add_edge(source_patient, target_patient, weight=edge_weight)

    return G


class GCNClassifier(nn.Module):
    """GCN de duas camadas + cabeça linear para classificação binária.

    A saída é um logit por paciente; a sigmoide é aplicada apenas na loss e
    na avaliação.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.lin(x)
        # logits (sem sigmoide, aplicada depois na loss/avaliação)
        return x.squeeze(-1)


class SGCClassifier(nn.Module):
    """SGC (propagação simplificada, ``K`` saltos) + cabeça linear."""

    def __init__(self, in_channels: int, hidden_channels: int, K: int = 2):
        super().__init__()
        self.conv = SGConv(in_channels, hidden_channels, K=K)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = self.lin(x)
        return x.squeeze(-1)  # logits


def _graph_to_pyg_data(G: nx.Graph) -> tuple[Data, list[str]]:
    """Converte o grafo NetworkX em um ``Data`` do PyTorch Geometric.

    As features ``genes`` já chegam tratadas (imputação + ``log1p`` +
    padronização) de :func:`build_patient_knn_graph`. Como o grafo é não
    direcionado, cada aresta é espelhada em ``edge_index``.

    Args:
        G: grafo com atributos ``genes`` e ``event`` nos nós e ``weight`` nas
            arestas.

    Returns:
        ``(data, node_ids)``, em que ``node_ids`` preserva a ordem usada para
        construir as linhas de ``data.x``/``data.y`` e as previsões.
    """
    node_ids = list(G.nodes())
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    genes = np.array([G.nodes[n]["genes"] for n in node_ids], dtype=np.float32)
    event = np.array([G.nodes[n]["event"] for n in node_ids], dtype=np.float32)

    edges = list(G.edges(data="weight"))
    src = [node_index[u] for u, v, w in edges] + [
        node_index[v] for u, v, w in edges
    ]
    dst = [node_index[v] for u, v, w in edges] + [
        node_index[u] for u, v, w in edges
    ]
    weights = [w for u, v, w in edges] * 2

    data = Data(
        x=torch.tensor(genes, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float32),
        y=torch.tensor(event, dtype=torch.float32),
    )
    return data, node_ids


def _train_model(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    """Treina um modelo transdutivo com Adam e BCE com ``pos_weight``.

    O ``pos_weight`` é calculado apenas com os nós de treino e compensa o
    desbalanceamento entre sobreviventes e óbitos.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    y_train = data.y[train_mask]
    num_positives = y_train.sum()
    num_negatives = len(y_train) - num_positives
    pos_weight = num_negatives / num_positives.clamp(min=1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = F.binary_cross_entropy_with_logits(
            out[train_mask], data.y[train_mask], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def _evaluate(
    model: nn.Module, data: Data, mask: torch.Tensor
) -> tuple[float, float, np.ndarray]:
    """Avalia o modelo em ``mask`` e devolve (balanced accuracy, AUC, probs)."""
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    probs = torch.sigmoid(out[mask]).numpy()
    preds = (probs >= 0.5).astype(int)
    targets = data.y[mask].numpy().astype(int)

    accuracy = balanced_accuracy_score(targets, preds)
    # A AUC exige as duas classes na máscara; em folds pequenos isso pode falhar.
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")

    return accuracy, auc, probs


def _run_single_split(
    data: Data,
    node_ids: list,
    architectures: dict,
    test_size: float,
    n_splits: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """Executa um split + CV interna + treino final para uma única seed.

    O split externo (treino/teste) e a validação cruzada interna são
    estratificados pela classe ``event``. A CV serve para estimar o
    desempenho dentro do treino; o modelo final é re-treinado em todo o
    conjunto de treino e avaliado no conjunto de teste.
    """
    num_nodes = data.num_nodes

    # Fixa a inicialização dos pesos das GNNs para esta seed. Sem isso, duas
    # execuções com o mesmo split produzem resultados diferentes.
    torch.manual_seed(seed)

    all_idx = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=seed, stratify=data.y.numpy()
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    seed_results = {}
    for name, build_model in architectures.items():
        fold_balanced_accuracies = []
        fold_aucs = []

        # Validação cruzada estratificada dentro do conjunto de treino.
        for fold_train_idx, fold_val_idx in skf.split(
            train_idx, data.y.numpy()[train_idx]
        ):
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx[fold_train_idx]] = True
            val_mask[train_idx[fold_val_idx]] = True

            model = build_model()
            model = _train_model(model, data, train_mask, epochs, lr)
            fold_acc, fold_auc, _ = _evaluate(model, data, val_mask)
            fold_balanced_accuracies.append(fold_acc)
            fold_aucs.append(fold_auc)

        # Modelo final: treinado com todo o conjunto de treino, avaliado no teste.
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        final_model = build_model()
        final_model = _train_model(final_model, data, train_mask, epochs, lr)
        test_balanced_accuracy, test_auc, test_probs = _evaluate(
            final_model, data, test_mask
        )

        seed_results[name] = {
            "cv_balanced_accuracy": float(np.mean(fold_balanced_accuracies)),
            "cv_auc": float(np.nanmean(fold_aucs)),
            "test_balanced_accuracy": float(test_balanced_accuracy),
            "test_auc": float(test_auc),
            "predictions": {
                node_ids[i]: {
                    "probability_event": float(prob),
                    "predicted_event": int(prob >= 0.5),
                }
                for i, prob in zip(test_idx, test_probs)
            },
        }

    return seed_results


def predict_event_gnn(
    G: nx.Graph,
    hidden_dim: int = 64,
    epochs: int = 200,
    lr: float = 0.01,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> dict:
    """Treina GCN e SGC no grafo e avalia a predição do evento.

    Para cada seed (``base_seed`` até ``base_seed + n_repeats - 1``), executa
    um split estratificado treino/teste, uma validação cruzada interna de
    ``n_splits`` folds e o treino final do modelo no conjunto de treino. O
    resultado agregado reporta média e desvio-padrão entre as seeds, além das
    previsões individuais.

    Args:
        G: grafo de pacientes produzido por
            :func:`build_patient_knn_graph`.
        hidden_dim: dimensão das camadas ocultas de GCN e SGC.
        epochs: número de épocas de treino por modelo.
        lr: taxa de aprendizado do Adam.
        test_size: fração reservada para o teste externo.
        n_splits: número de folds da CV interna.
        n_repeats: número de seeds independentes.
        base_seed: primeira seed; as demais são incrementais.

    Returns:
        Dicionário ``{nome_da_arquitetura: métricas}``. Cada entrada contém
        médias/desvios de balanced accuracy e ROC-AUC na CV e no teste, além
        de ``predictions_per_seed`` com a probabilidade e a classe prevista
        por paciente de teste em cada seed.
    """
    data, node_ids = _graph_to_pyg_data(G)

    architectures = {
        "GCN": lambda: GCNClassifier(data.num_node_features, hidden_dim),
        "SGC": lambda: SGCClassifier(data.num_node_features, hidden_dim),
    }

    seeds = [base_seed + i for i in range(n_repeats)]
    runs_per_architecture = {name: [] for name in architectures}

    for seed in seeds:
        seed_results = _run_single_split(
            data, node_ids, architectures, test_size, n_splits, epochs, lr, seed
        )
        for name in architectures:
            runs_per_architecture[name].append(seed_results[name])

    summary = {}
    for name, runs in runs_per_architecture.items():
        cv_balanced_accuracies = [r["cv_balanced_accuracy"] for r in runs]
        cv_aucs = [r["cv_auc"] for r in runs]
        test_balanced_accuracies = [r["test_balanced_accuracy"] for r in runs]
        test_aucs = [r["test_auc"] for r in runs]

        summary[name] = {
            "cv_balanced_accuracy_mean": float(np.mean(cv_balanced_accuracies)),
            "cv_balanced_accuracy_std": float(np.std(cv_balanced_accuracies)),
            "cv_auc_mean": float(np.nanmean(cv_aucs)),
            "cv_auc_std": float(np.nanstd(cv_aucs)),
            "test_balanced_accuracy_mean": float(np.mean(test_balanced_accuracies)),
            "test_balanced_accuracy_std": float(np.std(test_balanced_accuracies)),
            "test_auc_mean": float(np.nanmean(test_aucs)),
            "test_auc_std": float(np.nanstd(test_aucs)),
            "predictions_per_seed": {
                seed: r["predictions"] for seed, r in zip(seeds, runs)
            },
        }

        print(f"\n[{name}] CV Balanced Accuracy: {summary[name]['cv_balanced_accuracy_mean']:.4f} "
              f"(+/- {summary[name]['cv_balanced_accuracy_std']:.4f})")
        print(f"[{name}] CV AUC: {summary[name]['cv_auc_mean']:.4f} "
              f"(+/- {summary[name]['cv_auc_std']:.4f})")
        print(f"[{name}] Teste ({n_repeats} seeds) - Balanced Accuracy: "
              f"{summary[name]['test_balanced_accuracy_mean']:.4f} "
              f"(+/- {summary[name]['test_balanced_accuracy_std']:.4f}) | "
              f"AUC: {summary[name]['test_auc_mean']:.4f} "
              f"(+/- {summary[name]['test_auc_std']:.4f})")

    return summary
