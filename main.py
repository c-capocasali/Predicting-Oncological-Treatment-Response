import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import filter as f
    import polars as pl
    from lasso_analysis import lasso
    from graph import build_patient_knn_graph, predict_event_gnn
    from baselines import predict_event_classical

    return (
        build_patient_knn_graph,
        lasso,
        pl,
        predict_event_classical,
        predict_event_gnn,
    )


@app.cell
def _(pl):
    #Arquivos principais 
    df_bio = pl.read_csv("bio_filtrado.tsv", separator='\t')
    df_clinical = pl.read_csv("clinical_v2_fixed.tsv", separator='\t')
    project_names = ["TARGET-AML"] #Ajustar para outros projetos
    return


@app.cell
def _():
    #Cria o arquivo 
    #f.create_project_map_to_parquet(df_bio, df_clinical, project_names, "patients_table")
    target_file = "patients_table"
    return (target_file,)


@app.cell
def _(lasso, target_file):
    #Calculando os genes com o laço 
    selected_genes, c_index = lasso(target_file)
    print(selected_genes) 
    print("*"*50)
    print(c_index)
    return (selected_genes,)


@app.cell
def _(build_patient_knn_graph, predict_event_gnn, selected_genes, target_file):
    #Constroi o grafo com os genes selecionados e aplica GNNs
    G = build_patient_knn_graph(target_file, selected_genes, k=5) 
    gnn_summary = predict_event_gnn(G)  
    return (G,)


@app.cell
def _(G, predict_event_classical):
    #Comparação com modelos clássicos (KNN, MLP, SVM e RF)
    baselines_summary = predict_event_classical(G)
    return


if __name__ == "__main__":
    app.run()
