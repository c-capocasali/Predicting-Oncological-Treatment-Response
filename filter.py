"""Construção da tabela de pacientes a partir dos arquivos do GDC.

Este módulo faz a ponte entre três fontes de dados:

1. ``clinical_v2_fixed.tsv`` — metadados clínicos (um registro por caso);
2. ``bio_filtrado.tsv`` — relação caso → amostra → arquivo de expressão;
3. ``rna_files/*.tsv`` — matrizes de contagem por gene (STAR-Count do GDC).

O produto final é um arquivo Parquet (por padrão ``patients_table``) em que
cada linha é um paciente e cada coluna é um gene, acrescido das colunas de
metadados ``case_id``, ``project``, ``survival_time`` e ``event``.
"""

import polars as pl
from pathlib import Path


# Diretório em que os arquivos de expressão baixados do GDC são procurados.
# O caminho é relativo ao diretório de onde o pipeline é executado.
RNA_DIR = Path("rna_files")

# Linhas de controle do STAR-Count que não correspondem a genes.
ALIGNMENT_STAT_ROWS = [
    "N_unmapped",
    "N_multimapping",
    "N_noFeature",
    "N_ambiguous",
]


def compute_survival_columns(df_clinical: pl.DataFrame) -> pl.DataFrame:
    """Deriva tempo de acompanhamento e indicador de óbito do arquivo clínico.

    Regras de construção:

    - ``vital_status == "Dead"`` e ``days_to_death`` disponível:
      ``survival_time = days_to_death`` e ``event = 1``;
    - caso contrário (vivo, "Not Reported" ou óbito sem data), se
      ``days_to_last_follow_up`` estiver disponível:
      ``survival_time = days_to_last_follow_up`` e ``event = 0``;
    - sem nenhuma das duas medidas de tempo: a linha é descartada.

    Observação metodológica: ``event = 0`` significa "sem óbito registrado
    até o último acompanhamento", e não necessariamente "sobreviveu a um
    horizonte fixo". O código desta branch não aplica o horizonte de cinco
    anos descrito no relatório; para isso, veja a branch ``train_test_split``.

    Args:
        df_clinical: DataFrame clínico completo, com as colunas
            ``demographic.vital_status``, ``demographic.days_to_death`` e
            ``diagnoses.0.days_to_last_follow_up``.

    Returns:
        DataFrame com as colunas ``case_id``, ``survival_time`` e ``event``
        (0/1), sem linhas com tempo de sobrevivência nulo.
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
    """Localiza, em ``rna_files/``, os arquivos de expressão de um paciente.

    A busca é recursiva e casa pelo **nome exato** do arquivo (``file_name``
    vindo do biospecimen), não pelo caminho completo. Isso permite organizar
    os downloads do GDC em subpastas por projeto/amostra.

    Args:
        selected_files_names: nomes de arquivo (por exemplo,
            ``"abc-123.htseq.counts.gz"`` ou ``"abc-123.rna_seq.augmented_star_gene_counts.tsv"``)
            associados ao paciente.

    Returns:
        Lista com os caminhos encontrados. Pode ser menor que a lista de
        entrada caso algum arquivo não esteja presente; retorna ``[]`` se o
        diretório ``rna_files/`` não existir.
    """
    path = RNA_DIR

    if not path.is_dir():
        print(f"Diretório {path} não encontrado")
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


def get_rna_exp(file_paths: list[str]) -> dict[str, float]:
    """Agrega a expressão gênica de um paciente a partir de seus arquivos.

    Para cada arquivo STAR-Count do GDC, soma as contagens de
    ``stranded_second`` por ``gene_name`` (a coluna é a recomendada para
    análise de expressão gênica do GDC). Se o paciente tiver mais de uma
    amostra/arquivo, calcula a média entre os arquivos para cada gene.

    O leitor assume o layout dos arquivos STAR-Count do GDC: a primeira
    linha é um metadado comentado (por exemplo, ``# gene-model: GENCODE
    v36``), a segunda contém os nomes das colunas e, por isso,
    ``skip_rows=1``. As colunas ``gene_id``, ``gene_name`` e
    ``stranded_second`` precisam existir; ajuste o parâmetro se o arquivo
    tiver outro layout.

    Args:
        file_paths: caminhos dos arquivos de expressão do paciente.

    Returns:
        Dicionário ``{gene_name: expressão_média}``. Retorna ``{}`` se a
        lista estiver vazia.
    """

    if not file_paths:
        return {}

    lazy_reads = []
    for current_path in file_paths:
        lf = (
            pl.scan_csv(current_path, separator="\t", skip_rows=1)
            .filter(~pl.col("gene_id").is_in(ALIGNMENT_STAT_ROWS))
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


def create_project_map_to_parquet(
    df_bio: pl.DataFrame,
    df_clinical: pl.DataFrame,
    project_names: list[str],
    output_file: str,
) -> pl.DataFrame | None:
    """Monta a matriz paciente × gene e a persiste em Parquet.

    Etapas:

    1. deriva ``survival_time`` e ``event`` com
       :func:`compute_survival_columns`;
    2. faz o join de ``df_bio`` com os dados de sobrevivência pelo
       ``case_id``;
    3. para cada projeto, mantém apenas amostras coletadas no diagnóstico
       (``days_to_diagnosis == 0``) e agrupa os arquivos por paciente;
    4. lê os arquivos de RNA, agrega a expressão por gene e acrescenta as
       colunas clínicas;
    5. concatena os projetos e grava ``output_file`` em Parquet (Snappy).

    O processo é **caro**: no conjunto completo do relatório, são cerca de
    3 mil arquivos de expressão e dezenas de GB de leitura. Rode-o uma única
    vez e reutilize o Parquet gerado.

    Args:
        df_bio: conteúdo de ``bio_filtrado.tsv``.
        df_clinical: conteúdo de ``clinical_v2_fixed.tsv``.
        project_names: identificadores dos projetos do GDC a incluir, por
            exemplo ``["TCGA-BRCA", "TARGET-AML"]``.
        output_file: caminho do Parquet de saída (por convenção,
            ``"patients_table"``).

    Returns:
        O DataFrame consolidado, ou ``None`` se nenhum paciente tiver dados
        de expressão utilizáveis.
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
