"""Notebook marimo do pipeline principal.

Execute com::

    marimo edit main.py    # edição interativa
    marimo run main.py     # modo somente leitura

O notebook pressupõe que o arquivo ``patients_table`` (Parquet) já exista.
Para gerá-lo a partir de ``bio_filtrado.tsv``, ``clinical_v2_fixed.tsv`` e
da pasta ``rna_files/``, descomente a chamada a
``f.create_project_map_to_parquet`` na célula "Cria o arquivo". Essa etapa
lê milhares de arquivos de expressão e deve ser executada uma única vez.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo  # noqa: F401  (disponível para células de markdown)
    import polars as pl

    import filter as f
    from baselines import predict_event_classical
    from graph import build_patient_knn_graph, predict_event_gnn
    from lasso_analysis import lasso

    return (
        build_patient_knn_graph,
        f,
        lasso,
        pl,
        predict_event_classical,
        predict_event_gnn,
    )


@app.cell
def _(pl):
    # Arquivos de entrada versionados no repositório.
    df_bio = pl.read_csv("bio_filtrado.tsv", separator="\t")
    df_clinical = pl.read_csv("clinical_v2_fixed.tsv", separator="\t")

    # Ajuste para os projetos que deseja processar (ex.: ["TCGA-BRCA"]).
    project_names = ["TARGET-AML"]
    return df_bio, df_clinical, project_names


@app.cell
def _(f, df_bio, df_clinical, project_names):
    target_file = "patients_table"

    # Geração da tabela paciente x gene (cara; rode uma vez e reutilize).
    # f.create_project_map_to_parquet(
    #     df_bio, df_clinical, project_names, target_file
    # )
    return (target_file,)


@app.cell
def _(lasso, target_file):
    # Seleção de genes por Cox LASSO + C-index no conjunto de teste.
    selected_genes, c_index = lasso(target_file)
    print(selected_genes)
    print("*" * 50)
    print(f"C-index (teste): {c_index:.4f}")
    return (selected_genes,)


@app.cell
def _(build_patient_knn_graph, predict_event_gnn, selected_genes, target_file):
    # Grafo k-NN de pacientes + treinamento de GCN e SGC.
    G = build_patient_knn_graph(target_file, selected_genes, k=5)
    gnn_summary = predict_event_gnn(G)
    return G, gnn_summary


@app.cell
def _(G, predict_event_classical):
    # Comparação com modelos clássicos (KNN, SVM, MLP e Random Forest).
    baselines_summary = predict_event_classical(G)
    return (baselines_summary,)


if __name__ == "__main__":
    app.run()
