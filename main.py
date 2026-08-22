import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import filter as f
    import polars as pl

    return (pl,)


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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
