import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold


def lasso(target_file, top_n_variance_genes=520):

    # Carrega arquivo parquet
    df = pl.read_parquet(target_file)

    target_cols = ['survival_time', 'event', 'case_id', 'project']

    # Seleciona apenas colunas numéricas (evita o erro do log1p com case_id/project)
    X = df.select(cs.numeric()).drop(target_cols, strict=False).to_pandas()

# Array estruturado é exigido pela função do Cox Lasso
    y = np.array(
        list(zip(df['event'].cast(pl.Boolean).to_numpy(),
                 df['survival_time'].to_numpy())),
        dtype=[('Status', '?'), ('Survival_in_days', '<f8')]
    )

# Separação em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['event'].to_numpy())

# --- Pré-processamento pesado FORA do Pipeline/CV (feito uma única vez) ---
# Log(x+1): igual ao benchmark, aplicado antes do filtro de variância para não
# distorcer a seleção a favor de genes de alta expressão absoluta (ribossomais/
# mitocondriais). Rodar aqui evita recalcular isso a cada fold/alpha do CV.
    X_train_log = np.log1p(X_train.to_numpy())
    X_test_log = np.log1p(X_test.to_numpy())

# Filtro de Variância (ajuste o threshold se quiser remover mais ou menos genes)
    var_filter = VarianceThreshold(threshold=0.5)
    X_train_var = var_filter.fit_transform(X_train_log)
    X_test_var = var_filter.transform(X_test_log)
    genes_pos_variancia = X_train.columns[var_filter.get_support()]

# Pré-filtro adicional pelo top-N de maior variância (igual ao benchmark).
# Sem isso, o número de genes que sobra do VarianceThreshold pode ficar na
# casa dos milhares — e como o GridSearchCV abaixo usa n_jobs=-1, CADA
# worker recebe sua própria cópia de X_train_var, multiplicando o uso de
# memória pelo número de núcleos. Reduzir para um teto fixo de genes antes
# de entrar no Coxnet resolve o estouro de memória e acelera muito o fit.
    if top_n_variance_genes is not None and top_n_variance_genes < X_train_var.shape[1]:
        variances = X_train_var.var(axis=0)
        top_idx = np.argsort(variances)[::-1][:top_n_variance_genes]
        X_train_var = X_train_var[:, top_idx]
        X_test_var = X_test_var[:, top_idx]
        genes_pos_variancia = genes_pos_variancia[top_idx]
        print(f"[Lasso] Pré-filtro por variância: mantendo top {top_n_variance_genes} genes "
              f"(de {var_filter.get_support().sum()} pós VarianceThreshold)")

# Pipeline Batedor: Descobre a faixa de alphas ideal para esses dados
# alpha_min_ratio evita que o path de alphas desça até valores próximos de
# zero (quase sem regularização), que deixam o Coxnet mal-condicionado e
# geram o erro "weights are too large" (e, mesmo quando não falha, cada fit
# nesse regime é bem mais caro em tempo/memória).
    pipeline_batedor = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01))
    ])
    pipeline_batedor.fit(X_train_var, y_train)
    alphas_estimados = pipeline_batedor.named_steps['cox_lasso'].alphas_

# Prepara a grade para testar os alphas um por um
    param_grid = {'cox_lasso__alphas': [[a] for a in alphas_estimados]}

# Pipeline Final (com fit_baseline_model=True)
# max_iter/tol mais soltos que o padrão: como já restringimos o path de
# alphas (alpha_min_ratio), cada fit converge bem mais rápido; tol=1e-2
# evita gastar iterações refinando uma precisão que não muda o C-index.
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(
            l1_ratio=1.0, fit_baseline_model=True, max_iter=5_000_000, tol=1e-2))
    ])

# K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_train_var, y_train['Status']))

# GridSearch configurado com error_score=0.5
    gcv = GridSearchCV(
        pipeline_final,
        param_grid=param_grid,
        cv=cv_splits,
        n_jobs=-1,
        error_score=0.5
    )

    gcv.fit(X_train_var, y_train)

    melhor_alpha = gcv.best_params_['cox_lasso__alphas'][0]
    c_index_val = gcv.best_score_
    print(f"\n[Treino/CV] Melhor Alpha encontrado: {melhor_alpha:.6f}")
    print(f"[Treino/CV] C-Index médio na Validação Cruzada: {c_index_val:.4f}")

# O método .score() do scikit-survival calcula automaticamente o Harrell's C-index
    c_index_teste = gcv.score(X_test_var, y_test)

# Extração dos genes
    # 1. Recupera os coeficientes do modelo Lasso
    melhor_modelo_cox = gcv.best_estimator_.named_steps['cox_lasso']
    coeficientes_finais = melhor_modelo_cox.coef_[:, 0]
    # 2. Filtra apenas os que o Lasso não zerou, usando os nomes já filtrados pela variância
    genes_selecionados = genes_pos_variancia[coeficientes_finais != 0].tolist()

    print(f"\nTotal de genes sobreviventes ao Lasso: {
          len(genes_selecionados)}")

    return genes_selecionados, c_index_teste
