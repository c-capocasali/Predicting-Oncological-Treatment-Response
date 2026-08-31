import polars as pl
import polars.selectors as cs
import numpy as np
import gc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored


def lasso(target_file):
    # Carrega arquivo parquet
    df = pl.read_parquet(target_file)

    target_cols = ['survival_time', 'event', 'case_id', 'project']

    # 1. Extrai colunas e converte direto para numpy float32
    feature_cols = df.select(cs.numeric()).drop(
        target_cols, strict=False).columns
    X = df.select(feature_cols).to_numpy().astype(np.float32)

    # Array estruturado exigido pela função do Cox Lasso
    y = np.array(
        list(zip(df['event'].cast(pl.Boolean).to_numpy(),
                 df['survival_time'].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

    # 2. Limpeza de memória
    del df
    gc.collect()

    # Separação em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['Status'])

    del X
    gc.collect()

    # Log(x+1)
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)

    del X_train, X_test
    gc.collect()

    # Filtro de Variância
    var_filter = VarianceThreshold(threshold=0.5)
    X_train_var = var_filter.fit_transform(X_train_log)
    X_test_var = var_filter.transform(X_test_log)
    genes_pos_variancia = np.array(feature_cols)[var_filter.get_support()]

    # 3. Filtro rigoroso para o top 520 genes com maior variância
    if X_train_var.shape[1] > 520:
        variances = X_train_var.var(axis=0)
        top_idx = np.argsort(variances)[::-1][:520]

        X_train_var = X_train_var[:, top_idx]
        X_test_var = X_test_var[:, top_idx]
        genes_pos_variancia = genes_pos_variancia[top_idx]

    # Pipeline Batedor
    pipeline_batedor = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0))
    ])
    pipeline_batedor.fit(X_train_var, y_train)
    alphas_estimados = pipeline_batedor.named_steps['cox_lasso'].alphas_

    # Prepara a grade
    param_grid = {'cox_lasso__alphas': [[a] for a in alphas_estimados]}

    # Pipeline Final
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True))
    ])

    # K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_train_var, y_train['Status']))

    # 4. n_jobs=2 para evitar OOM
    gcv = GridSearchCV(
        pipeline_final,
        param_grid=param_grid,
        cv=cv_splits,
        n_jobs=2,
        error_score=0.5
    )

    gcv.fit(X_train_var, y_train)

    melhor_alpha = gcv.best_params_['cox_lasso__alphas'][0]
    c_index_val = gcv.best_score_
    print(f"\n[Treino/CV] Melhor Alpha encontrado: {melhor_alpha:.6f}")
    print(f"[Treino/CV] C-Index médio na Validação Cruzada: {c_index_val:.4f}")

    # C-index do teste
    c_index_teste = gcv.score(X_test_var, y_test)

    # Extração dos genes
    melhor_modelo_cox = gcv.best_estimator_.named_steps['cox_lasso']
    coeficientes_finais = melhor_modelo_cox.coef_[:, 0]
    genes_selecionados = genes_pos_variancia[coeficientes_finais != 0].tolist()

    print(f"\nTotal de genes sobreviventes ao Lasso: {
          len(genes_selecionados)}")

    # ========== INÍCIO DO BLOCO PARA PLOTAGEM ==========
    cox_batedor = pipeline_batedor.named_steps['cox_lasso']
    alphas = cox_batedor.alphas_
    coefs = cox_batedor.coef_

    scaler = pipeline_batedor.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test_var)

    n_genes_list = []
    c_index_list = []

    for i in range(len(alphas)):
        coef = coefs[i]
        n_genes = np.sum(coef != 0)
        risk = X_test_scaled.dot(coef)

        # Correção: Pegando apenas o primeiro elemento da tupla
        c_idx = concordance_index_censored(
            y_test['Status'], y_test['Survival_in_days'], risk)[0]

        n_genes_list.append(n_genes)
        c_index_list.append(c_idx)

    best_idx = np.argmin(np.abs(alphas - melhor_alpha))

    plt.figure(figsize=(10, 6))
    plt.plot(n_genes_list, c_index_list, marker='o',
             linestyle='-', color='blue', label='Todos os alphas')
    plt.plot(n_genes_list[best_idx], c_index_list[best_idx],
             'r*', markersize=12, label='Melhor alpha (CV)')
    plt.xlabel('Número de genes selecionados')
    plt.ylabel('C-index (teste)')
    plt.title('Desempenho do Lasso-Cox em função do número de genes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # ========== FIM DO BLOCO PARA PLOTAGEM ==========

    return genes_selecionados, c_index_teste
