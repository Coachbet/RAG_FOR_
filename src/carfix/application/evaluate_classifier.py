"""Use case: evaluate text classifiers on a held-out test set (Session 2.1 metrics).

Accuracy alone hides weak classes, so we always report macro-F1, per-class precision / recall / F1 and the
confusion matrix (which systems get mixed up).
"""
from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from carfix.domain.models import LabelledText
from carfix.domain.ports import TextClassifier


def evaluate_classifier(clf: TextClassifier, examples: list[LabelledText], labels: list[str]) -> dict:
    y_true = [e.label for e in examples]
    y_pred = clf.predict([e.text for e in examples])
    return {
        "model": clf.name,
        "n": len(examples),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
        "errors": [{"text": e.text, "true": e.label, "pred": p}
                   for e, p in zip(examples, y_pred) if p != e.label],
    }
