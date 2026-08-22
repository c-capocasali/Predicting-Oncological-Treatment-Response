import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import data_filter as df

    return (df,)


@app.cell
def _(pl):
    #Arquivos principais 
    df_bio = pl.read_csv("bio_filtrado.tsv", separator='\t')
    df_clinical = pl.read_csv("clinical_v2_fixed.tsv", separator='\t')
    project_names = ["TARGET-AML"] #Ajustar para outros projetos
    return df_bio, df_clinical, project_names


@app.cell
def _(df, df_bio, df_clinical, project_names):
    #Cria o arquivo
    df.create_project_map_to_parquet(df_bio, df_clinical, project_names, "patients_table")
    return


if __name__ == "__main__":
    app.run()
