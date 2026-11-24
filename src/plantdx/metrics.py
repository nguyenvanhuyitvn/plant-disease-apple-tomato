"""Chỉ số đánh giá: accuracy, F1 macro, khoảng tin cậy bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def basic_metrics(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    present = sorted(set(y_true))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # F1 macro tính trên các lớp có mặt trong tập đánh giá
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t, p = y_true[idx], y_pred[idx]
        accs.append((t == p).mean())
        f1s.append(f1_score(t, p, labels=sorted(set(t)), average="macro", zero_division=0))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"acc_ci": np.percentile(accs, q).tolist(), "f1_ci": np.percentile(f1s, q).tolist()}

