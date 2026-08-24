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
    X_genes = StandardScaler().fit_transform(X_genes)

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
            edge_weight = distances[i][j]

            # No networkx, se A liga em B, e B liga em A, a aresta não é duplicada (grafo não-direcionado)
            G.add_edge(source_patient, target_patient, weight=edge_weight)

    return G


class GCNClassifier(nn.Module):
    """
    GCN de 2 camadas, seguindo EXATAMENTE a estrutura da classe `Net` em
    gcn(2).py: conv1 -> relu -> dropout -> conv2 -> log_softmax. A segunda
    GCNConv já produz `num_classes` canais (não há cabeça Linear extra), e o
    treino usa nll_loss sobre log_softmax (classificação, não logit binário).
    Assim como no arquivo de referência, edge_weight não é usado — o grafo é
    tratado como não-ponderado dentro da GNN.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGCClassifier(nn.Module):
    """
    SGC seguindo EXATAMENTE a estrutura da classe `SGC` em gcn(2).py: uma
    única SGConv (propagação simplificada, K saltos) produzindo diretamente
    `num_classes` canais, sem cabeça Linear e sem não-linearidade extra entre
    a convolução e a saída — log_softmax aplicado direto sobre a SGConv,
    preservando a hipótese central da SGC (linearidade entre as camadas de
    propagação). `cached=True` como no arquivo de referência.
    """

    def __init__(self, in_channels: int, num_classes: int = 2, K: int = 2):
        super().__init__()
        self.conv1 = SGConv(in_channels, num_classes, K=K, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


def _graph_to_pyg_data(G: nx.Graph) -> tuple[Data, list]:
    """
    Converte o grafo networkx (nos com atributos 'genes' e 'event') em um
    objeto Data do PyTorch Geometric. 'genes' ja chega tratado (nulls
    imputados, log1p + StandardScaler) desde build_patient_knn_graph.
    """
    node_ids = list(G.nodes())
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    genes = np.array([G.nodes[n]["genes"] for n in node_ids], dtype=np.float32)
    # long (nao float): classes discretas para F.nll_loss + log_softmax,
    # seguindo o paradigma de classificacao usado em gcn(2).py
    event = np.array([G.nodes[n]["event"] for n in node_ids], dtype=np.int64)

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
        y=torch.tensor(event, dtype=torch.long),
    )
    return data, node_ids


def _compute_class_weight(y: torch.Tensor, train_mask: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Pesos por classe para F.nll_loss (equivalente, em espírito, ao pos_weight
    usado antes com BCE): compensa o desbalanceamento das classes calculado a
    partir do conjunto de treino. O arquivo de referência (gcn(2).py) não usa
    nenhum peso de classe — mantemos essa compensação aqui por ser a mesma
    escolha de justiça já documentada no restante do pipeline (baselines.py
    usa class_weight='balanced'), só adaptada de BCE para nll_loss.
    """
    y_train = y[train_mask]
    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    weight = class_counts.sum() / (num_classes * class_counts.clamp(min=1))
    return weight


def _train_model(model: nn.Module, data: Data, train_mask: torch.Tensor, epochs: int, lr: float) -> nn.Module:
    # weight_decay=5e-4 e optimizer identicos ao GCNClassifier.predict em gcn(2).py
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    class_weight = _compute_class_weight(data.y, train_mask)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # nll_loss sobre log_softmax, como em gcn(2).py (F.nll_loss(out[train_mask], y[train_mask]))
        loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weight)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def _evaluate(model: nn.Module, data: Data, mask: torch.Tensor) -> tuple[float, float, np.ndarray]:
    model.eval()
    out = model(data.x, data.edge_index)
    # out ja e log_softmax; exp() devolve as probabilidades por classe
    probs_all = torch.exp(out)
    probs = probs_all[mask][:, 1].numpy()  # probabilidade da classe evento=1
    preds = probs_all[mask].argmax(dim=1).numpy()
    targets = data.y[mask].numpy()

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
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_splits: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """
    Executa a CV interna + treino final para uma única seed, sobre um split
    treino/teste FIXO (train_idx/test_idx), o mesmo em todas as repetições.

    Por que o split é fixo entre repetições: a seleção de genes via Lasso
    (lasso_analysis.py) é feita uma única vez, sobre um único split
    (random_state=42). Se cada repetição gerasse um split treino/teste novo,
    pacientes que fizeram parte do treino do Lasso em outras repetições
    acabariam no conjunto de teste — vazamento de seleção de features. Fixar
    o split (mesma seed do Lasso) elimina esse vazamento. As seeds diferentes
    entre repetições continuam variando a CV interna (StratifiedKFold) e a
    inicialização dos pesos do modelo, preservando uma medida de variância.
    """
    num_nodes = data.num_nodes
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

            torch.manual_seed(seed)
            model = build_model()
            model = _train_model(model, data, train_mask, epochs, lr)
            fold_acc, fold_auc, _ = _evaluate(model, data, val_mask)
            fold_balanced_accuracies.append(fold_acc)
            fold_aucs.append(fold_auc)

        # Modelo final: treinado com todo o conjunto de treino, avaliado no teste
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        torch.manual_seed(seed)
        final_model = build_model()
        final_model = _train_model(final_model, data, train_mask, epochs, lr)
        test_balanced_accuracy, test_auc, test_probs = _evaluate(
            final_model, data, test_mask)

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
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 0.0001,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> dict:
    """
    Treina GCN e SGC sobre o grafo de pacientes para prever o evento
    (0 = vivo/censurado, 1 = obito), evitando o vies de censura que
    afeta a previsao direta do survival_time.

    GCN e SGC seguem EXATAMENTE a estrutura de Net e SGC em gcn(2).py
    (mesmas camadas/ordem, log_softmax + nll_loss, sem edge_weight).
    hidden_dim, epochs e lr default tambem replicam pNNeurons=32,
    pNEpochs=200 e pLR=0.0001 do arquivo de referencia.

    Split treino/teste UNICO e FIXO (seed=base_seed), o mesmo usado pelo
    Lasso em lasso_analysis.py — evita vazamento da selecao de genes para
    o teste (ver docstring de _run_single_split). As n_repeats seeds
    (base_seed, base_seed+1, ...) variam apenas a CV interna (5-fold) e a
    inicializacao dos pesos do modelo, medindo variabilidade sem gerar
    novos conjuntos de teste.

    Retorna, para cada arquitetura, a media e o desvio padrao (entre as n_repeats
    execucoes) de balanced accuracy e AUC, tanto da CV interna quanto do teste final,
    alem das previsoes de cada execucao.
    """
    data, node_ids = _graph_to_pyg_data(G)

    architectures = {
        "GCN": lambda: GCNClassifier(data.num_node_features, hidden_dim),
        "SGC": lambda: SGCClassifier(data.num_node_features),
    }

    # Split externo fixo (Opcao B) - calculado uma unica vez, igual ao do Lasso
    all_idx = np.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=base_seed, stratify=data.y.numpy()
    )

    seeds = [base_seed + i for i in range(n_repeats)]
    runs_per_architecture = {name: [] for name in architectures}

    for seed in seeds:
        seed_results = _run_single_split(
            data, node_ids, architectures, train_idx, test_idx, n_splits, epochs, lr, seed
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

    return summary
