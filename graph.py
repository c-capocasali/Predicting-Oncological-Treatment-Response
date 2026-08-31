import polars as pl
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SGConv


def build_patient_knn_graph(target_file: str, selected_genes: list[str], k: int = 5) -> nx.Graph:
    """
    Constrói um grafo KNN não-direcionado onde cada vértice é um paciente.
    """
    # 1. Carregar os dados
    df = pl.read_parquet(target_file)

    # Valida se a coluna 'case_id' existe (usada no seu filter.py),
    # senão usa o índice da linha como identificador do vértice
    has_case_id = "case_id" in df.columns
    cols_to_select = (["case_id", "survival_time", "event"] if has_case_id else [
                      "survival_time", "event"]) + selected_genes
    df_filtered = df.select(cols_to_select)

    # 2. Matriz de features para o KNN (Apenas Genes)
    X_genes = df_filtered.select(selected_genes).to_numpy()

    # 2.1 Tratamento de nulls (paciente sem valor reportado para algum gene)
    # Imputação pela média do gene, calculada ignorando os NaNs existentes
    if np.isnan(X_genes).any():
        col_means = np.nanmean(X_genes, axis=0)
        nan_idx = np.where(np.isnan(X_genes))
        X_genes[nan_idx] = np.take(col_means, nan_idx[1])

    # 2.2 Normalização: mesma transformação usada na seleção via Lasso (lasso_analysis.py),
    # para manter o espaço de features consistente entre seleção e distância do grafo
    X_genes = np.log1p(X_genes)
    # X_genes = StandardScaler().fit_transform(X_genes)

    # 3. Calcular os K Vizinhos Mais Próximos
    # Usamos k + 1 porque o algoritmo considera o próprio ponto como o vizinho mais próximo (distância 0)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn.fit(X_genes)
    distances, indices = nn.kneighbors(X_genes)

    # 4. Construir o Grafo
    G = nx.Graph()
    node_ids = df_filtered["case_id"].to_list(
    ) if has_case_id else list(range(len(df_filtered)))

    # 4.1 Adicionar os Vértices (Nós) com seus Embeddings
    for i, row in enumerate(df_filtered.iter_rows(named=True)):
        node_id = node_ids[i]

        # Salvamos as features já tratadas (imputadas e normalizadas) para facilitar a
        # conversão futura para PyTorch Geometric (Data.x e Data.y), evitando reprocessar
        # nulls/normalização em outro lugar do pipeline
        G.add_node(
            node_id,
            genes=X_genes[i].tolist(),
            survival_time=row["survival_time"],
            event=row["event"]
        )

# 4.2 Adicionar as Arestas
    for i, neighbors in enumerate(indices):
        source_patient = node_ids[i]

        # Começamos do range(1, ...) para ignorar o vizinho 0 (que é o próprio paciente)
        for j in range(1, k + 1):
            target_patient = node_ids[neighbors[j]]

            # Modificação: Transformando a distância Euclidiana em Similaridade
            raw_distance = distances[i][j]
            edge_weight = 1.0 / (1.0 + raw_distance)

            # No networkx, se A liga em B, e B liga em A, a aresta não é duplicada
            G.add_edge(source_patient, target_patient, weight=edge_weight)

    return G


class GCNClassifier(nn.Module):
    """GCN de 2 camadas + cabeça linear para classificação binária do evento."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.lin(x)
        # logits (sem sigmoid, aplicado depois na loss/avaliacao)
        return x.squeeze(-1)


class SGCClassifier(nn.Module):
    """SGC (propagacao simplificada, K saltos) + cabeca linear para classificacao binaria."""

    def __init__(self, in_channels: int, hidden_channels: int, K: int = 2):
        super().__init__()
        self.conv = SGConv(in_channels, hidden_channels, K=K)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = self.lin(x)
        return x.squeeze(-1)  # logits


def _graph_to_pyg_data(G: nx.Graph) -> tuple[Data, list]:
    """
    Converte o grafo networkx (nos com atributos 'genes' e 'event') em um
    objeto Data do PyTorch Geometric. 'genes' ja chega tratado (nulls
    imputados, log1p + StandardScaler) desde build_patient_knn_graph.
    """
    node_ids = list(G.nodes())
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    genes = np.array([G.nodes[n]["genes"] for n in node_ids], dtype=np.float32)
    event = np.array([G.nodes[n]["event"] for n in node_ids], dtype=np.float32)

    # Arestas nos dois sentidos, ja que o grafo e nao-direcionado
    edges = list(G.edges(data="weight"))
    src = [node_index[u] for u, v, w in edges] + [node_index[v]
                                                  for u, v, w in edges]
    dst = [node_index[v] for u, v, w in edges] + [node_index[u]
                                                  for u, v, w in edges]
    weights = [w for u, v, w in edges] * 2

    data = Data(
        x=torch.tensor(genes, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float32),
        y=torch.tensor(event, dtype=torch.float32),
    )
    return data, node_ids


def _train_model(model: nn.Module, data: Data, train_mask: torch.Tensor, epochs: int, lr: float) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # pos_weight compensa o desbalanceamento das classes na loss: dá mais peso à classe
    # minoritária (geralmente o evento=1), calculado a partir do próprio conjunto de treino
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
def _evaluate(model: nn.Module, data: Data, mask: torch.Tensor) -> tuple[float, float, np.ndarray]:
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    probs = torch.sigmoid(out[mask]).numpy()
    preds = (probs >= 0.5).astype(int)
    targets = data.y[mask].numpy().astype(int)

    accuracy = balanced_accuracy_score(targets, preds)
    # AUC exige as duas classes presentes na mascara; em folds pequenos isso pode falhar
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

    num_nodes = data.num_nodes

    all_idx = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=seed, stratify=data.y.numpy()
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    seed_results = {}
    for name, build_model in architectures.items():
        fold_balanced_accuracies = []
        fold_aucs = []

        # Validacao cruzada estratificada dentro do conjunto de treino
        for fold_train_idx, fold_val_idx in skf.split(train_idx, data.y.numpy()[train_idx]):
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx[fold_train_idx]] = True
            val_mask[train_idx[fold_val_idx]] = True

            # Normalização isolada no fold
            fold_data = data.clone()
            scaler = StandardScaler()
            train_nodes = train_idx[fold_train_idx]
            scaler.fit(fold_data.x[train_nodes].numpy())
            fold_data.x = torch.tensor(scaler.transform(
                fold_data.x.numpy()), dtype=torch.float32)

            model = build_model()
            # Substitua 'data' por 'fold_data'
            model = _train_model(model, fold_data, train_mask, epochs, lr)
            fold_acc, fold_auc, _ = _evaluate(model, fold_data, val_mask)
            fold_balanced_accuracies.append(fold_acc)
            fold_aucs.append(fold_auc)

        # Modelo final: treinado com todo o conjunto de treino, avaliado no teste
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        # Normalização isolada pro modelo final
        final_data = data.clone()
        scaler = StandardScaler()
        scaler.fit(final_data.x[train_idx].numpy())
        final_data.x = torch.tensor(scaler.transform(
            final_data.x.numpy()), dtype=torch.float32)

        final_model = build_model()
        # Substitua 'data' por 'final_data'
        final_model = _train_model(
            final_model, final_data, train_mask, epochs, lr)
        test_balanced_accuracy, test_auc, test_probs = _evaluate(
            final_model, final_data, test_mask)

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
    # Modificado para receber a lista de splits
    test_sizes: list[float] = [0.8, 0.6, 0.4, 0.2],
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> dict:
    """
    Treina GCN e SGC sobre o grafo de pacientes em múltiplos splits de treino/teste.
    """
    data, node_ids = _graph_to_pyg_data(G)

    architectures = {
        "GCN": lambda: GCNClassifier(data.num_node_features, hidden_dim),
        "SGC": lambda: SGCClassifier(data.num_node_features, hidden_dim),
    }

    seeds = [base_seed + i for i in range(n_repeats)]

    all_splits_summary = {}  # Dicionário para armazenar os resultados de todos os splits

    for test_size in test_sizes:
        train_pct = int(round((1 - test_size) * 100))
        test_pct = int(round(test_size * 100))

        print(f"\n{'='*50}")
        print(f"AVALIANDO SPLIT: Treino {train_pct}% / Teste {test_pct}%")
        print(f"{'='*50}")

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
            test_balanced_accuracies = [
                r["test_balanced_accuracy"] for r in runs]
            test_aucs = [r["test_auc"] for r in runs]

            summary[name] = {
                "cv_balanced_accuracy_mean": float(np.mean(cv_balanced_accuracies)),
                "cv_balanced_accuracy_std": float(np.std(cv_balanced_accuracies)),
                "cv_auc_mean": float(np.nanmean(cv_aucs)),
                "cv_auc_std": float(np.nanstd(cv_aucs)),
                "test_balanced_accuracy_mean": float(np.mean(test_balanced_accuracies)),
                "test_balanced_accuracy_std": float(np.std(test_balanced_accuracies)),
                "test_auc_mean": float(np.mean(test_aucs)),
                "test_auc_std": float(np.std(test_aucs)),
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

        all_splits_summary[f"train_{train_pct}"] = summary

    return all_splits_summary
