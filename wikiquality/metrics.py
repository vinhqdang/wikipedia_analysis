"""Classification and ordinal metrics for quality-class prediction."""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "within_one": float((np.abs(y_true - y_pred) <= 1).mean()),
    }
