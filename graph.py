import polars as pl
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_patient_knn_graph(target_file: str, selected_genes: list[str], k: int = 5) -> nx.Graph:
    """
    Constrói um grafo KNN não-direcionado onde cada vértice é um paciente.
    """
    # 1. Carregar os dados
    df = pl.read_parquet(target_file)

    # Valida se a coluna 'case_id' existe (usada no seu filter.py),
    # senão usa o índice da linha como identificador do vértice
    has_case_id = "case_id" in df.columns
    cols_to_select = (["case_id", "survival_time"] if has_case_id else [
                      "survival_time"]) + selected_genes
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

        # Salvamos as features separadas para facilitar a conversão futura para PyTorch Geometric (Data.x e Data.y)
        G.add_node(
            node_id,
            genes=[row[gene] for gene in selected_genes],
            survival_time=row["survival_time"]
        )

    # 4.2 Adicionar as Arestas
    for i, neighbors in enumerate(indices):
        source_patient = node_ids[i]

        for j in range(1, k + 1):
            target_patient = node_ids[neighbors[j]]
            edge_weight = distances[i][j]

            # No networkx, se A liga em B, e B liga em A, a aresta não é duplicada (grafo não-direcionado)
            G.add_edge(source_patient, target_patient, weight=edge_weight)

    return G
