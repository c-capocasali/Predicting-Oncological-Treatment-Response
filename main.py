import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import filter as f
    import polars as pl
    from lasso_analysis import (
        train_test_case_ids,
        lasso_path_curve,
        rsf_feature_curve,
        plot_c_index_curves,
        best_gene_set,
    )
    from graph import build_patient_knn_graph, predict_event_gnn, predict_event_classical

    return (
        best_gene_set,
        build_patient_knn_graph,
        lasso_path_curve,
        plot_c_index_curves,
        pl,
        predict_event_classical,
        predict_event_gnn,
        rsf_feature_curve,
        train_test_case_ids,
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
def _(target_file, train_test_case_ids):
    #Split treino/teste ÚNICO, por case_id, feito uma única vez e reaproveitado
    #em TODA a pipeline (seleção de genes -> grafo -> GNN/baselines). Isso
    #evita que a seleção de genes "veja" pacientes que depois viram teste da
    #GNN/baseline (vazamento de informação da label para dentro das features).
    train_ids, test_ids = train_test_case_ids(
        target_file, test_size=0.2, random_state=42)
    print(f"Split externo único: {len(train_ids)} treino / {len(test_ids)} teste")
    return test_ids, train_ids


@app.cell
def _(best_gene_set, lasso_path_curve, plot_c_index_curves, rsf_feature_curve, target_file, train_ids):
    #Curva C-index x n_genes para o Cox Lasso (path de alphas), calculada
    #SÓ com os pacientes de treino do split externo (case_id_filter=train_ids)
    lasso_curve = lasso_path_curve(
        target_file, n_points=15, case_id_filter=train_ids)

    #Curva C-index x n_genes para o Random Survival Forest, usando os mesmos
    #n_genes que saíram do path do Lasso (comparação no mesmo eixo X), também
    #restrita aos pacientes de treino
    rsf_curve = rsf_feature_curve(
        target_file, list(lasso_curve.keys()), case_id_filter=train_ids)

    plot_c_index_curves(lasso_curve, rsf_curve)

    #Escolhe o melhor conjunto de genes entre as duas curvas
    modelo_vencedor, n_genes, c_index, selected_genes = best_gene_set(
        lasso_curve, rsf_curve)

    print(selected_genes)
    print("*"*50)
    print(c_index)
    return (selected_genes,)


@app.cell
def _(build_patient_knn_graph, predict_event_gnn, selected_genes, target_file, test_ids):
    #Constroi o grafo com os genes selecionados (grafo usa TODOS os pacientes,
    #treino+teste, pois a GNN e transdutiva) e aplica GNNs usando o MESMO
    #conjunto de teste (test_ids) reservado desde o início, nunca visto pela
    #seleção de genes
    G = build_patient_knn_graph(target_file, selected_genes, k=5)
    gnn_summary = predict_event_gnn(G, test_case_ids=test_ids)
    return (G,)


@app.cell
def _(G, predict_event_classical, test_ids):
    #Comparação com modelos clássicos (KNN, MLP, SVM e RF), usando o mesmo
    #test_ids da GNN e da seleção de genes, para comparação justa
    baselines_summary = predict_event_classical(G, test_case_ids=test_ids)
    return


if __name__ == "__main__":
    app.run()
