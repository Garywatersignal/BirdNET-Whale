from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix


def evaluate_model(model, test_embeddings, test_labels, num_classes=7) -> Dict:
    y_pred_proba = model.predict(test_embeddings, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(test_labels, y_pred)

    y_true_onehot = tf.one_hot(test_labels, depth=num_classes).numpy()
    auc_scores = []
    ap_scores = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
        except Exception:
            auc = float("nan")
        try:
            ap = average_precision_score(y_true_onehot[:, i], y_pred_proba[:, i])
        except Exception:
            ap = float("nan")
        auc_scores.append(auc)
        ap_scores.append(ap)

    macro_auc = float(np.nanmean(auc_scores))
    macro_ap = float(np.nanmean(ap_scores))

    return {
        "accuracy": float(accuracy),
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "per_class_auc": auc_scores,
        "per_class_ap": ap_scores,
        "confusion_matrix": confusion_matrix(test_labels, y_pred),
    }
