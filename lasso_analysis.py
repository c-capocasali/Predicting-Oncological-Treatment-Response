import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold


def lasso(target_file):

    # Carrega arquivo parquet
    df = pl.read_parquet(target_file)

    target_cols = ['survival_time', 'event']
    gene_cols = [col for col in df.columns if col not in target_cols]

    X = df.select(gene_cols).to_pandas()

# Array estruturado é exigido pela função do Cox Lasso
    y = np.array(
        list(zip(df['event'].cast(pl.Boolean).to_numpy(),
                 df['survival_time'].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

# Separação em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['event'].to_numpy())

# Transformador Log(x+1)
    log_transformer = FunctionTransformer(np.log1p, validate=False)

# Pipeline Batedor: Descobre a faixa de alphas ideal para esses dados
    pipeline_batedor = Pipeline([
        ('log', log_transformer),
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0))
    ])
    pipeline_batedor.fit(X_train, y_train)
    alphas_estimados = pipeline_batedor.named_steps['cox_lasso'].alphas_

# Prepara a grade para testar os alphas um por um
    param_grid = {'cox_lasso__alphas': [[a] for a in alphas_estimados]}

# Pipeline Final (com fit_baseline_model=True)
    pipeline_final = Pipeline([
        ('log', log_transformer),
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True))
    ])

# K-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch configurado com error_score=0.5
    gcv = GridSearchCV(
        pipeline_final,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        error_score=0.5
    )

    gcv.fit(X_train, y_train)

    melhor_alpha = gcv.best_params_['cox_lasso__alphas'][0]
    c_index_val = gcv.best_score_
    print(f"\n[Treino/CV] Melhor Alpha encontrado: {melhor_alpha:.6f}")
    print(f"[Treino/CV] C-Index médio na Validação Cruzada: {c_index_val:.4f}")

# O método .score() do scikit-survival calcula automaticamente o Harrell's C-index
    c_index_teste = gcv.score(X_test, y_test)

    melhor_modelo_cox = gcv.best_estimator_.named_steps['cox_lasso']
    coeficientes_finais = melhor_modelo_cox.coef_[:, 0]

    genes_selecionados = X_train.columns[coeficientes_finais != 0].tolist()

    print(f"\nTotal de genes sobreviventes ao Lasso: {
          len(genes_selecionados)}")

    return genes_selecionados, c_index_teste
