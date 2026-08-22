# Bibliotecas importadas
import polars as pl
from pathlib import Path


def compute_survival_columns(df_clinical: pl.DataFrame) -> pl.DataFrame:
    """
    Recebe o DataFrame clínico completo e devolve um DataFrame com:
        case_id | survival_time (float) | event (int 0/1)

    Regras:
      - Se o paciente morreu (vital_status == "Dead") e days_to_death é válido
        → survival_time = days_to_death, event = 1
      - Caso contrário (Alive ou Not Reported) e days_to_last_follow_up é válido
        → survival_time = days_to_last_follow_up, event = 0
      - Se nenhum dos dois está disponível → linha descartada (null → filtrado)
    """
    return (
        df_clinical
        .select([
            "case_id",
            pl.col("demographic.vital_status").alias("vital_status"),
            pl.col("demographic.days_to_death").cast(
                pl.Float64, strict=False).alias("days_to_death"),
            pl.col("diagnoses.0.days_to_last_follow_up").cast(
                pl.Float64, strict=False).alias("days_to_last_follow_up")
        ])
        .with_columns([
            # event = 1 apenas quando morreu E temos o tempo da morte
            pl.when(
                (pl.col("vital_status") == "Dead") & pl.col(
                    "days_to_death").is_not_null()
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("event"),

            # survival_time: prioriza days_to_death para quem morreu
            pl.when(
                (pl.col("vital_status") == "Dead") & pl.col(
                    "days_to_death").is_not_null()
            )
            .then(pl.col("days_to_death"))
            .when(pl.col("days_to_last_follow_up").is_not_null())
            .then(pl.col("days_to_last_follow_up"))
            .otherwise(None)
            .alias("survival_time"),
        ])
        .filter(
            pl.col("survival_time").is_not_null()
        )
        .select(["case_id", "survival_time", "event"])
    )


def find_paths(selected_files_names: list[str]) -> list[str]:
    """
    Recebe uma lista com os nomes dos arquivos procurados e devolve uma lista 
    com os caminhos completos (paths em string) de onde eles estão no sistema.

    Regras:
      - Define o diretório alvo como a pasta "rna_files" no diretório atual.
      - Se a pasta "rna_files" não for encontrada, avisa no console e retorna lista vazia.
      - Converte os nomes procurados para um 'set' (conjunto) para deixar a checagem 
        muito mais rápida (busca O(1)).
      - Itera recursivamente (rglob) por todos os arquivos de extensão ".tsv" dentro da pasta.
      - Quando encontra um arquivo desejado: guarda o caminho final, remove ele do 
        conjunto de busca e continua.
      - Otimização (Early Exit): Se o conjunto de busca ficar vazio (ou seja, já encontrou 
        todos os arquivos do paciente), interrompe o laço imediatamente para poupar processamento.
    """
    current_dir = Path('.')
    path = current_dir / "rna_files"

    if not path.is_dir():
        print("Diretório rna_files não encontrado")
        return []

    files_to_find = set(selected_files_names)
    found_paths = []

    for file in path.rglob("*.tsv"):
        if file.name in files_to_find:
            found_paths.append(str(file))
            files_to_find.remove(file.name)
            if not files_to_find:
                break

    return found_paths


def get_rna_exp(file_paths: list[str]) -> dict:
    """
    Recebe uma lista de caminhos de arquivos de RNA e devolve um dicionário com:
        gene_name (chave) | stranded_second média (valor float)

    Regras:
      - Se a lista de arquivos estiver vazia, retorna um dicionário vazio.
      - Remove estatísticas de alinhamento que não são genes reais 
        (ex: "N_unmapped", "N_multimapping", etc.).
      - Para cada arquivo na lista (iteração):
          * Lê o arquivo de forma preguiçosa (lazy) ignorando o cabeçalho (skip_rows=1).
          * Filtra descartando as linhas indesejadas listadas em 'remove'.
          * Agrupa pelo nome do gene e soma os valores da coluna 'stranded_second' naquele arquivo.
      - Junta (concatena) as leituras de todos os arquivos do paciente.
      - Como o paciente pode ter múltiplos arquivos com o mesmo gene, faz um agrupamento final 
        pelo nome do gene, calculando a MÉDIA da expressão ('stranded_second') entre os arquivos.
      - Converte o resultado final em um dicionário (chave = gene_name, valor = stranded_second).
    """

    if not file_paths:
        return {}

    # Remove colunas indesejáveis
    remove = ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]

    lazy_reads = []
    for current_path in file_paths:
        lf = (
            pl.scan_csv(current_path, separator="\t", skip_rows=1)
            .filter(~pl.col("gene_id").is_in(remove))
            .group_by("gene_name")
            .agg(pl.col("stranded_second").sum())
        )
        lazy_reads.append(lf)

    # Processa todos os arquivos juntos e tira a média geral de cada gene
    final_result = (
        pl.concat(lazy_reads)
        .group_by("gene_name")
        .agg(pl.col("stranded_second").mean())
        .collect()
    )

    return dict(zip(final_result["gene_name"], final_result["stranded_second"]))


def create_project_map_to_parquet(df_bio: pl.DataFrame, df_clinical: pl.DataFrame, project_names: list[str], output_file: str) -> pl.DataFrame:
    """
    Recebe dados biológicos e clínicos, processa as informações de sobrevivência e gera um arquivo Parquet.

    Regras:
      - Gera as colunas de sobrevivência usando a função compute_survival_columns.
      - Faz o join (cruzamento) de df_bio com df_clinical usando o case_id.
      - Para cada projeto em project_names:
          * Filtra os pacientes que pertencem ao projeto e cujo days_to_diagnosis é 0.
          * Agrupa por paciente (case_id) para remover duplicatas, extraindo o primeiro survival_time e event, 
            e agrupando todos os file_names do paciente em uma lista.
          * Itera sobre cada paciente (linha agrupada) para buscar seus arquivos de RNA (get_rna_exp).
          * Junta as informações clínicas com os dados de RNA num dicionário.
      - Concatena os resultados de todos os projetos.
      - Salva o DataFrame final em formato Parquet com compressão snappy e o retorna.
    """
    all_data_frames = []

    # Filtra colunas de "survival" e faz join com bio
    df_survival = compute_survival_columns(df_clinical)

    lf_bio = df_bio.lazy()
    lf_survival = df_survival.lazy()

    df_base = lf_bio.join(
        lf_survival,
        left_on="cases.0.case_id",
        right_on="case_id"
    )

    for project in project_names:
        project_results = (
            df_base
            .filter(
                (pl.col("cases.0.project.project_id") == project) &
                (pl.col("cases.0.diagnoses.0.days_to_diagnosis") == 0)
            )
            .group_by("cases.0.case_id")
            .agg([
                pl.col("survival_time").first().alias("survival_time"),
                pl.col("event").first().alias("event"),
                # Cria uma lista de arquivos por paciente
                pl.col("file_name").alias("file_names")
            ])
            .collect()
        )

        rows = []
        for row in project_results.iter_rows(named=True):
            # Coleta informação do RNA de cada paciente
            file_paths = find_paths(row["file_names"])
            rna_data = get_rna_exp(file_paths)

            if rna_data:
                patient_row = {
                    "case_id": row["cases.0.case_id"],
                    "project": project,
                    "survival_time": row["survival_time"],
                    "event": row["event"],
                }
                patient_row.update(rna_data)
                rows.append(patient_row)

        if rows:
            all_data_frames.append(pl.from_dicts(rows))

    # Prevenção caso nenhum dado atenda aos filtros
    if not all_data_frames:
        print("Patient File Creation Error")
        return None

    final_df = pl.concat(all_data_frames, how="diagonal")
    final_df.write_parquet(output_file, compression="snappy")

    return final_df
