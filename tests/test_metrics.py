import numpy as np

from evaluation.metrics import evaluate


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["precision"] == 1 and m["recall"] == 1 and m["f1"] == 1
    assert m["roc_auc"] == 1 and m["pr_auc"] == 1
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_confusion_matrix_counts_and_threshold():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.6, 0.2, 0.4, 0.7, 0.9])
    m = evaluate(y, p)  # default threshold 0.5
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 2)
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    m2 = evaluate(y, p, threshold=0.3)
    assert (m2["fn"], m2["tp"]) == (0, 3)


def test_random_scores_pr_auc_near_class_share():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.3).astype(int)
    m = evaluate(y, rng.random(5000))
    assert abs(m["pr_auc"] - 0.3) < 0.03
    assert abs(m["roc_auc"] - 0.5) < 0.03