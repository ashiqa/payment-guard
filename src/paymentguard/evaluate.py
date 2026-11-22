"""Operating-point evaluation; no threshold optimization on the test set."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve


def choose_threshold(y, scores, max_fpr: float):
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must lie in [0,1].")
    fpr, recall, thresholds = roc_curve(y, scores, drop_intermediate=False)
    # A finite value above one is an explicit review-none policy, including ties.
    thresholds = np.where(np.isinf(thresholds), np.nextafter(1.0, 2.0), thresholds)
    feasible = np.flatnonzero(fpr <= max_fpr)
    idx = max(feasible, key=lambda i: (recall[i], -fpr[i], thresholds[i]))
    return float(thresholds[idx])


def metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    both = np.unique(y).size == 2
    return {
        "rows": len(y), "fraud_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)) if both else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "review_rate": float(np.mean(scores >= threshold)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def reliability(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    result = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (scores >= low) & ((scores < high) if high < 1 else (scores <= high))
        if mask.any():
            result.append({"low": float(low), "high": float(high), "count": int(mask.sum()),
                           "mean_score": float(scores[mask].mean()), "fraud_rate": float(y[mask].mean())})
    return result
