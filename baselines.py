"""Modelos clássicos de comparação (sem propagação em grafo).

Este módulo avalia alternativas tabulares — KNN, SVM com kernel RBF, MLP e
Random Forest — sobre as mesmas features e os mesmos splits usados pelas
GNNs em :mod:`graph`. O objetivo é isolar o efeito da propagação no grafo:
se uma GNN não supera um modelo tabular equivalente, o ganho não pode ser
atribuído à topologia.

As features vêm dos atributos ``genes`` dos nós do grafo, que já passaram
por imputação, ``log1p`` e padronização em
:func:`graph.build_patient_knn_graph`.
"""

import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def _graph_to_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray, list]:
    """Extrai features e alvo do grafo na mesma ordem de nós usada pelas GNNs.

    Manter a ordem é essencial para que ``train_test_split`` e
    ``StratifiedKFold`` gerem, com a mesma seed, exatamente os mesmos índices
    de treino/teste em :mod:`graph` e aqui.
    """
    node_ids = list(G.nodes())
    X = np.array([G.nodes[n]["genes"] for n in node_ids], dtype=np.float64)
    y = np.array([G.nodes[n]["event"] for n in node_ids], dtype=np.int64)
    return X, y, node_ids


def _build_classical_models(k_values: list[int]) -> dict:
    """Monta os modelos clássicos a avaliar.

    Cada ``K`` do KNN vira um modelo separado. ``class_weight='balanced'``
    é usado em SVM e Random Forest para compensar o desbalanceamento de
    classes, de forma análoga ao ``pos_weight`` das GNNs. KNN e MLP não têm
    esse parâmetro no scikit-learn.

    Os hiperparâmetros seguem os valores da branch principal e podem diferir
    do protocolo final do relatório (por exemplo, ``MLP`` aqui usa
    ``max_iter=1000`` e o relatório usa 2000).
    """
    models = {
        f"KNN_k{k}": (lambda k=k: KNeighborsClassifier(n_neighbors=k))
        for k in k_values
    }
    models["SVM"] = lambda: SVC(
        probability=True, class_weight="balanced", random_state=42
    )
    models["MLP"] = lambda: MLPClassifier(max_iter=1000, random_state=42)
    models["RandomForest"] = lambda: RandomForestClassifier(
        class_weight="balanced", random_state=42
    )
    return models


def _evaluate_classical(
    model, X: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Avalia um modelo clássico em ``idx``.

    Retorna ``(balanced_accuracy, roc_auc, probabilidades)``. Se o modelo não
    expuser ``predict_proba`` (ou a AUC falhar), a AUC vira ``NaN`` e as
    predições binárias são usadas como escore.
    """
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
) -> dict:
    """
    Mesma lógica de split (treino/teste + CV interna) usada em
    :func:`graph._run_single_split`, aplicada aos modelos clássicos.
    """
    all_idx = np.arange(len(y))
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

            # Proteção para K do KNN maior que o número de amostras do fold.
            if name.startswith("KNN") and int(name.split("_k")[1]) > len(
                train_sub_idx
            ):
                fold_balanced_accuracies.append(float("nan"))
                fold_aucs.append(float("nan"))
                continue

            model = build_model()
            model.fit(X[train_sub_idx], y[train_sub_idx])
            fold_acc, fold_auc, _ = _evaluate_classical(
                model, X, y, val_sub_idx
            )
            fold_balanced_accuracies.append(fold_acc)
            fold_aucs.append(fold_auc)

        # Modelo final: treinado com todo o conjunto de treino, avaliado no teste.
        if name.startswith("KNN") and int(name.split("_k")[1]) > len(train_idx):
            print(
                f"[{name}] K maior que o numero de pacientes de treino "
                f"({len(train_idx)}); seed {seed} ignorada para este modelo."
            )
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
) -> dict:
    """Avalia KNN, SVM, MLP e Random Forest na predição do evento.

    Usa exatamente os mesmos splits treino/teste e a mesma CV interna de
    :func:`graph.predict_event_gnn` (mesmas seeds, ``test_size`` e
    estratificação), além das mesmas features já tratadas. Assim, a
    diferença de desempenho pode ser atribuída ao modelo e, no caso das
    GNNs, à propagação no grafo.

    Args:
        G: grafo de pacientes com atributos ``genes`` e ``event``.
        k_values: valores de ``K`` para o KNN (cada um vira um modelo).
        test_size: fração reservada para teste.
        n_splits: número de folds da CV interna.
        n_repeats: número de seeds independentes.
        base_seed: primeira seed; as demais são incrementais.

    Returns:
        Dicionário ``{nome_do_modelo: métricas}`` com médias/desvios de
        balanced accuracy e ROC-AUC na CV e no teste, além das previsões por
        seed. Modelos inviáveis (K maior que o treino) são omitidos.
    """
    X, y, node_ids = _graph_to_arrays(G)
    models = _build_classical_models(k_values)

    seeds = [base_seed + i for i in range(n_repeats)]
    # Guarda as execuções por seed (e não apenas uma lista) para que o
    # dicionário de previsões não fique desalinhado quando alguma seed é
    # ignorada por causa de um K grande demais.
    runs_per_model: dict[str, dict[int, dict]] = {name: {} for name in models}

    for seed in seeds:
        seed_results = _run_single_split_classical(
            X, y, node_ids, models, test_size, n_splits, seed
        )
        for name in models:
            if seed_results[name] is not None:
                runs_per_model[name][seed] = seed_results[name]

    summary = {}
    for name, runs_by_seed in runs_per_model.items():
        if not runs_by_seed:
            print(
                f"\n[{name}] Nenhuma execução válida "
                "(K provavelmente maior que o dataset)."
            )
            continue

        runs = list(runs_by_seed.values())
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
            "test_auc_mean": float(np.nanmean(test_aucs)),
            "test_auc_std": float(np.nanstd(test_aucs)),
            "predictions_per_seed": {
                seed: r["predictions"] for seed, r in runs_by_seed.items()
            },
        }

        print(f"\n[{name}] Teste ({len(runs)} seeds) - Balanced Accuracy: "
              f"{summary[name]['test_balanced_accuracy_mean']:.4f} "
              f"(+/- {summary[name]['test_balanced_accuracy_std']:.4f}) | "
              f"AUC: {summary[name]['test_auc_mean']:.4f} "
              f"(+/- {summary[name]['test_auc_std']:.4f})")

    return summary
