import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def _graph_to_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extrai features ('genes') e alvo ('event') do grafo, na MESMA ordem de nós
    usada por _graph_to_pyg_data (gnn_survival.py). Isso garante que, para a
    mesma seed, train_test_split/StratifiedKFold gerem exatamente os mesmos
    índices de treino/teste usados pelas GNNs, permitindo comparação direta.
    """
    node_ids = list(G.nodes())
    X = np.array([G.nodes[n]["genes"] for n in node_ids], dtype=np.float64)
    y = np.array([G.nodes[n]["event"] for n in node_ids], dtype=np.int64)
    return X, y, node_ids


def _build_classical_models(k_values: list[int]) -> dict:
    """
    Monta os modelos clássicos a avaliar. Cada K do KNN é tratado como um
    modelo separado (sem busca de hiperparâmetro, conforme pedido).

    Observação de justiça na comparação: usamos class_weight='balanced' nos
    modelos que suportam esse parâmetro (SVM, Random Forest), para compensar
    o desbalanceamento das classes de forma equivalente ao pos_weight usado
    no treino das GNNs. KNN e MLP não têm suporte nativo a isso no sklearn.
    """
    models = {
        f"KNN_k{k}": (lambda k=k: KNeighborsClassifier(n_neighbors=k))
        for k in k_values
    }
    models["SVM"] = lambda: SVC(
        probability=True, class_weight="balanced", random_state=42)
    models["MLP"] = lambda: MLPClassifier(max_iter=1000, random_state=42)
    models["RandomForest"] = lambda: RandomForestClassifier(
        class_weight="balanced", random_state=42)
    return models


def _evaluate_classical(model, X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[float, float, np.ndarray]:
    preds = model.predict(X[idx])
    balanced_acc = balanced_accuracy_score(y[idx], preds)

    try:
        probs = model.predict_proba(X[idx])[:, 1]
        auc = roc_auc_score(y[idx], probs)
    except (ValueError, AttributeError):
        probs = preds.astype(float)
        auc = float("nan")

    return balanced_acc, auc, probs


def _run_single_split_classical(
    X: np.ndarray,
    y: np.ndarray,
    node_ids: list,
    models: dict,
    test_size: float,
    n_splits: int,
    seed: int,
    fixed_test_idx: np.ndarray = None,
) -> dict:
    """
    Mesma lógica de split (treino/teste + CV interna) usada em
    _run_single_split (graph.py), aplicada aos modelos clássicos.

    Se fixed_test_idx for passado, o teste é sempre esse (mesmo split externo
    único usado na seleção de genes e na GNN); só a CV interna varia por seed.
    """
    all_idx = np.arange(len(y))

    if fixed_test_idx is not None:
        test_idx = fixed_test_idx
        train_idx = np.setdiff1d(all_idx, test_idx)
    else:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=seed, stratify=y
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    seed_results = {}
    for name, build_model in models.items():
        fold_balanced_accuracies = []
        fold_aucs = []

        for fold_train_idx, fold_val_idx in skf.split(train_idx, y[train_idx]):
            train_sub_idx = train_idx[fold_train_idx]
            val_sub_idx = train_idx[fold_val_idx]

            # Protecao para K do KNN maior que o numero de amostras de treino do fold
            if name.startswith("KNN") and int(name.split("_k")[1]) > len(train_sub_idx):
                fold_balanced_accuracies.append(float("nan"))
                fold_aucs.append(float("nan"))
                continue

            model = build_model()
            model.fit(X[train_sub_idx], y[train_sub_idx])
            fold_acc, fold_auc, _ = _evaluate_classical(
                model, X, y, val_sub_idx)
            fold_balanced_accuracies.append(fold_acc)
            fold_aucs.append(fold_auc)

        # Modelo final: treinado com todo o conjunto de treino, avaliado no teste
        if name.startswith("KNN") and int(name.split("_k")[1]) > len(train_idx):
            print(f"[{name}] K maior que o numero de pacientes de treino ({len(train_idx)}); "
                  f"seed {seed} ignorada para este modelo.")
            seed_results[name] = None
            continue

        final_model = build_model()
        final_model.fit(X[train_idx], y[train_idx])
        test_balanced_accuracy, test_auc, test_probs = _evaluate_classical(
            final_model, X, y, test_idx)

        seed_results[name] = {
            "cv_balanced_accuracy": float(np.nanmean(fold_balanced_accuracies)),
            "cv_auc": float(np.nanmean(fold_aucs)),
            "test_balanced_accuracy": float(test_balanced_accuracy),
            "test_auc": float(test_auc),
            "predictions": {
                node_ids[i]: {
                    "probability_event": float(prob),
                    "predicted_event": int(prob >= 0.5),
                }
                for i, prob in zip(test_idx, test_probs)
            },
        }

    return seed_results


def predict_event_classical(
    G: nx.Graph,
    k_values: list[int] = [3, 5, 10, 20, 40, 80, 160],
    test_size: float = 0.2,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
    test_case_ids: list = None,
) -> dict:
    """
    Avalia modelos clássicos (KNN para vários K, SVM, MLP, Random Forest) na
    predição do evento (0 = vivo/censurado, 1 = óbito), usando exatamente os
    mesmos splits treino/teste e a mesma CV interna de predict_event_gnn
    (mesmas seeds, mesmo test_size, mesma estratificação), além das mesmas
    features já tratadas (genes com nulls imputados, log1p + StandardScaler).

    A comparação direta com predict_event_gnn serve para avaliar se a
    estrutura do grafo KNN + propagação da GNN agrega sinal em relação aos
    mesmos dados tratados de forma puramente tabular (sem grafo).

    test_case_ids: mesmo papel de predict_event_gnn — fixa o conjunto de
    teste (tipicamente os test_ids de lasso_analysis.train_test_case_ids)
    para ser o mesmo em todas as seeds e o mesmo já reservado na seleção de
    genes, em vez de resortear o split a cada seed.
    """
    X, y, node_ids = _graph_to_arrays(G)
    models = _build_classical_models(k_values)

    fixed_test_idx = None
    if test_case_ids is not None:
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        fixed_test_idx = np.array(
            [node_index[cid] for cid in test_case_ids if cid in node_index],
            dtype=np.int64,
        )
        missing = set(test_case_ids) - set(node_index.keys())
        if missing:
            print(f"[predict_event_classical] Aviso: {len(missing)} case_ids de teste "
                  f"não encontrados no grafo (ignorados).")

    seeds = [base_seed + i for i in range(n_repeats)]
    runs_per_model = {name: [] for name in models}

    for seed in seeds:
        seed_results = _run_single_split_classical(
            X, y, node_ids, models, test_size, n_splits, seed,
            fixed_test_idx=fixed_test_idx,
        )
        for name in models:
            if seed_results[name] is not None:
                runs_per_model[name].append(seed_results[name])

    summary = {}
    for name, runs in runs_per_model.items():
        if not runs:
            print(
                f"\n[{name}] Nenhuma execução válida (K provavelmente maior que o dataset).")
            continue

        cv_balanced_accuracies = [r["cv_balanced_accuracy"] for r in runs]
        cv_aucs = [r["cv_auc"] for r in runs]
        test_balanced_accuracies = [r["test_balanced_accuracy"] for r in runs]
        test_aucs = [r["test_auc"] for r in runs]

        summary[name] = {
            "cv_balanced_accuracy_mean": float(np.nanmean(cv_balanced_accuracies)),
            "cv_balanced_accuracy_std": float(np.nanstd(cv_balanced_accuracies)),
            "cv_auc_mean": float(np.nanmean(cv_aucs)),
            "cv_auc_std": float(np.nanstd(cv_aucs)),
            "test_balanced_accuracy_mean": float(np.mean(test_balanced_accuracies)),
            "test_balanced_accuracy_std": float(np.std(test_balanced_accuracies)),
            "test_auc_mean": float(np.mean(test_aucs)),
            "test_auc_std": float(np.std(test_aucs)),
            "predictions_per_seed": {
                seed: r["predictions"] for seed, r in zip(seeds, runs)
            },
        }

        print(f"\n[{name}] Teste ({len(runs)} seeds) - Balanced Accuracy: "
              f"{summary[name]['test_balanced_accuracy_mean']:.4f} "
              f"(+/- {summary[name]['test_balanced_accuracy_std']:.4f}) | "
              f"AUC: {summary[name]['test_auc_mean']:.4f} "
              f"(+/- {summary[name]['test_auc_std']:.4f})")

    return summary
