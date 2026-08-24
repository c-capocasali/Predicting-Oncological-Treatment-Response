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
def _(
    build_patient_knn_graph,
    predict_event_classical,
    predict_event_gnn,
    selected_genes,
    target_file,
):
    k_list = [3, 5, 10, 20, 40, 80]
    n_genes_list = [10, 20, 40, 80, 158] 

    resultados_finais = {}

    for n in n_genes_list:
        print(f"\n{'='*60}")
        print(f"TESTANDO PARA N = {n} GENES")
        print(f"{'='*60}")
    
        # Seleciona apenas os top 'n' genes
        genes_subset = selected_genes[:n]
    
        resultados_finais[n] = {'gnn': {}, 'classical': None}

        for idx, k_element in enumerate(k_list):
            print(f"\n--- K (Vizinhos no Grafo) = {k_element} ---")
        
            # Constrói o grafo para a combinação atual de N e K
            G = build_patient_knn_graph(target_file, genes_subset, k=k_element, top_n_variance_genes = 520) 
        
            # Treina e avalia as GNNs
            gnn_summary = predict_event_gnn(G)  
            resultados_finais[n]['gnn'][k_element] = gnn_summary
        
            # Roda os baselines clássicos APENAS 1 vez por quantidade de genes
            if idx == 0:
                print("\n[!] Executando Baselines Clássicos...")
                baselines_summary = predict_event_classical(G, k_values=k_list)
                resultados_finais[n]['classical'] = baselines_summary
    return (G,)


@app.cell
def _():
    #Constroi o grafo com os genes selecionados e aplica GNNs
    # G = build_patient_knn_graph(target_file, selected_genes, k=5) 
    # gnn_summary = predict_event_gnn(G)  
    return


@app.cell
def _(G, predict_event_classical):
    #Comparação com modelos clássicos (KNN, MLP, SVM e RF)
    baselines_summary = predict_event_classical(G)
    return


if __name__ == "__main__":
    app.run()
