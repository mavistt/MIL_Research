from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def _safe(metric, y_true, values, default=float("nan")):
    try:
        return float(metric(y_true, values))
    except ValueError:
        return default


def compute_metrics(y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5) -> dict:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = np.asarray(cm).reshape(-1)
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "auroc": _safe(roc_auc_score, y, p),
        "auprc": _safe(average_precision_score, y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": cm,
    }


def save_curves(y_true: Iterable[int], probabilities: Iterable[float], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    try:
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y, p):.4f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "roc_curve.png", dpi=160)
        plt.close()
    except ValueError:
        pass
    try:
        precision, recall, _ = precision_recall_curve(y, p)
        plt.figure()
        plt.plot(recall, precision, label=f"AUPRC={average_precision_score(y, p):.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "pr_curve.png", dpi=160)
        plt.close()
    except ValueError:
        pass
    cm = np.asarray(compute_metrics(y, p)["confusion_matrix"])
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, int(cm[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()


def dump_metrics(metrics: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)

