import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt                      # <-- NOVO
from sksurv.metrics import concordance_index_censored  # <-- NOVO


def lasso(target_file):

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

# Pipeline Batedor: Descobre a faixa de alphas ideal para esses dados
    pipeline_batedor = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0))
    ])
    pipeline_batedor.fit(X_train_var, y_train)
    alphas_estimados = pipeline_batedor.named_steps['cox_lasso'].alphas_

# Prepara a grade para testar os alphas um por um
    param_grid = {'cox_lasso__alphas': [[a] for a in alphas_estimados]}

# Pipeline Final (com fit_baseline_model=True)
    pipeline_final = Pipeline([
        ('scaler', StandardScaler()),
        ('cox_lasso', CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True))
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

    # ========== INÍCIO DO BLOCO PARA PLOTAGEM ==========
    # Reutiliza o pipeline_batedor já ajustado para obter o caminho completo
    cox_batedor = pipeline_batedor.named_steps['cox_lasso']
    alphas = cox_batedor.alphas_
    coefs = cox_batedor.coef_          # shape (n_alphas, n_features)

    scaler = pipeline_batedor.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test_var)

    n_genes_list = []
    c_index_list = []

    for i in range(len(alphas)):
        coef = coefs[i]
        n_genes = np.sum(coef != 0)
        risk = X_test_scaled.dot(coef)   # escore de risco (quanto maior, pior)
        c_idx = concordance_index_censored(
            y_test['Status'], y_test['Survival_in_days'], risk)
        n_genes_list.append(n_genes)
        c_index_list.append(c_idx)

    # Marca a posição do melhor alpha (escolhido pelo GridSearchCV)
    best_idx = np.argmin(np.abs(alphas - melhor_alpha))  # índice mais próximo

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
