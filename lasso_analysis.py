"""Seleção supervisionada de genes com Cox LASSO.

O objetivo desta etapa é reduzir a matriz de expressão (dezenas de milhares
de genes) a um subconjunto enxuto e informativo para a construção do grafo e
para as GNNs. O pipeline é:

1. separação estratificada treino/teste (80/20);
2. ``log1p`` nas contagens, para estabilizar a variância;
3. filtro de variância (limiar 0.5) ajustado **somente** no treino;
4. padronização e Cox LASSO (``CoxnetSurvivalAnalysis``, ``l1_ratio=1``);
5. busca do alpha por validação cruzada estratificada (5 folds), usando o
   C-index de Harrell como métrica;
6. retorno dos genes com coeficiente não nulo no melhor modelo.

Atenção: o ``GridSearchCV`` usa ``error_score=0.5``. Quando um fold falha
(por exemplo, por instabilidade numérica do Cox), ele recebe C-index 0.5 em
vez de interromper a busca. Isso torna a execução robusta, mas pode
favorecer alphas cujos folds falharam; os resultados devem ser interpretados
com essa ressalva.

O código também não faz uma seleção "aninhada": o conjunto de teste é usado
uma única vez, ao final, para reportar o C-index. A seleção de genes em si
depende apenas do conjunto de treino, mas o split 80/20 é fixo
(``random_state=42``) e não é repetido nas seeds externas do experimento.
"""

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis


# Colunas de metadados que não entram como features.
TARGET_COLUMNS = ["survival_time", "event", "case_id", "project"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
VARIANCE_THRESHOLD = 0.5
N_SPLITS = 5
L1_RATIO = 1.0


def lasso(target_file: str) -> tuple[list[str], float]:
    """Seleciona genes com Cox LASSO e devolve o C-index no teste.

    Args:
        target_file: caminho do Parquet gerado por
            :func:`filter.create_project_map_to_parquet`. Espera-se que as
            colunas numéricas (exceto ``survival_time`` e ``event``) sejam
            genes e que existam as colunas ``case_id``, ``project``,
            ``survival_time`` e ``event``.

    Returns:
        Uma tupla ``(genes_selecionados, c_index_teste)`` em que
        ``genes_selecionados`` contém os nomes dos genes com coeficiente
        diferente de zero e ``c_index_teste`` é o C-index de Harrell obtido
        pelo melhor modelo no conjunto de teste.
    """
    df = pl.read_parquet(target_file)

    # Apenas colunas numéricas viram features; metadados são removidos.
    X = df.select(cs.numeric()).drop(TARGET_COLUMNS, strict=False).to_pandas()

    # O scikit-survival exige um array estruturado (status booleano + tempo).
    y = np.array(
        list(
            zip(
                df["event"].cast(pl.Boolean).to_numpy(),
                df["survival_time"].to_numpy(),
            )
        ),
        dtype=[("Status", "?"), ("Survival_in_days", "<f8")],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["event"].to_numpy(),
    )

    # log1p antes do filtro de variância: evita que genes de expressão absoluta
    # muito alta (ribossomais/mitocondriais) dominem a seleção.
    X_train_log = np.log1p(X_train.to_numpy())
    X_test_log = np.log1p(X_test.to_numpy())

    # O filtro é ajustado no treino e apenas aplicado ao teste.
    var_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_train_var = var_filter.fit_transform(X_train_log)
    X_test_var = var_filter.transform(X_test_log)
    genes_pos_variancia = X_train.columns[var_filter.get_support()]

    # "Pipeline batedor": descobre a faixa de alphas sugerida pelos dados.
    pipeline_batedor = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("cox_lasso", CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO)),
        ]
    )
    pipeline_batedor.fit(X_train_var, y_train)
    alphas_estimados = pipeline_batedor.named_steps["cox_lasso"].alphas_

    # Um alpha por vez, para permitir a seleção por C-index na CV.
    param_grid = {"cox_lasso__alphas": [[a] for a in alphas_estimados]}

    # fit_baseline_model=True permite usar o modelo para predição/score.
    pipeline_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "cox_lasso",
                CoxnetSurvivalAnalysis(
                    l1_ratio=L1_RATIO, fit_baseline_model=True
                ),
            ),
        ]
    )

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    cv_splits = list(skf.split(X_train_var, y_train["Status"]))

    gcv = GridSearchCV(
        pipeline_final,
        param_grid=param_grid,
        cv=cv_splits,
        n_jobs=-1,
        error_score=0.5,
    )
    gcv.fit(X_train_var, y_train)

    melhor_alpha = gcv.best_params_["cox_lasso__alphas"][0]
    c_index_val = gcv.best_score_
    print(f"\n[Treino/CV] Melhor Alpha encontrado: {melhor_alpha:.6f}")
    print(f"[Treino/CV] C-Index médio na Validação Cruzada: {c_index_val:.4f}")

    # .score() do scikit-survival calcula o C-index de Harrell.
    c_index_teste = gcv.score(X_test_var, y_test)

    melhor_modelo_cox = gcv.best_estimator_.named_steps["cox_lasso"]
    coeficientes_finais = melhor_modelo_cox.coef_[:, 0]
    genes_selecionados = genes_pos_variancia[coeficientes_finais != 0].tolist()

    print(f"\nTotal de genes sobreviventes ao Lasso: {len(genes_selecionados)}")

    return genes_selecionados, c_index_teste
