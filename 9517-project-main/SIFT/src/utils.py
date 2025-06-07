# src/utils.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred, classes):
    """
    Evaluate model predictions using various classification metrics.
    Handles label mismatch issues and prints a complete report.
    Args:
        y_true: list of ground truth labels
        y_pred: list of predicted labels
        classes: full list of class names (indexed)
    """

    def _get_core_scores(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

    def _get_labels_and_names(y_true, classes):
        unique = sorted(set(y_true))
        return unique, [classes[i] for i in unique]

    def _generate_report(y_true, y_pred, labels, names):
        return classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=names,
            zero_division=0
        )

    # === Metric Computation ===
    scores = _get_core_scores(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    labels, names = _get_labels_and_names(y_true, classes)
    report = _generate_report(y_true, y_pred, labels, names)

    # === Output Report ===
    print("\n═══ Evaluation Summary ═══")
    print(f"✅ Accuracy       : {scores['accuracy']:.4f}")
    print(f"🎯 Precision      : {scores['precision']:.4f}")
    print(f"🔁 Recall         : {scores['recall']:.4f}")
    print(f"📊 F1-Score       : {scores['f1']:.4f}")

    print(f"\n🧩 Confusion Matrix:\n{cm}")
    print(f"\n📝 Classification Report:\n{report}")