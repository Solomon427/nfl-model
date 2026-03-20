import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_regression_predictions(y_true, y_pred) -> dict[str, float]:
    """
    Compute regression metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_classifier_predictions(
    y_true,
    y_pred,
    positive_class: int = 2,
) -> dict[str, float]:
    """
    Compute classification metrics.

    For multiclass tasks, recall_star is recall for the chosen positive class
    using labels=[positive_class].

    For binary tasks where labels are {0,1}, set positive_class=1.
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    recall_star = float(
        recall_score(
            y_true,
            y_pred,
            labels=[positive_class],
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "recall_star": recall_star,
    }