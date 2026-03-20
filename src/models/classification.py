import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.models.evaluation import evaluate_classifier_predictions
from src.models.split import time_split
from src.utils.config import MIN_POSITION_SIZE, RANDOM_STATE


def get_classification_models(include_advanced: bool = True) -> dict[str, object]:
    """
    Return classification models used in the notebook.
    """
    models = {
        "Logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_depth=8,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "NaiveBayes": GaussianNB(),
        "SVC": SVC(
            kernel="rbf",
            class_weight="balanced",
        ),
    }

    if include_advanced:
        try:
            import xgboost as xgb

            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
        except Exception:
            pass

        try:
            from catboost import CatBoostClassifier

            models["CatBoost"] = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                verbose=0,
            )
        except Exception:
            pass

    return models


def evaluate_classification_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    positive_class: int = 2,
) -> dict[str, float]:
    """
    Fit classifier and compute evaluation metrics.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_classifier_predictions(y_test, preds, positive_class=positive_class)


def benchmark_classification(
    datasets: dict[str, pd.DataFrame],
    label_col: str,
    positive_class: int,
    min_size: int = MIN_POSITION_SIZE,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Benchmark classification models across position datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Position -> dataframe with features + label_col + draft_year
    label_col : str
        Name of target label column
    positive_class : int
        Class used for recall_star
        - use 2 for quantile / multiclass
        - use 1 for binary star classification
    """
    results: list[dict] = []
    models = get_classification_models(include_advanced=include_advanced)

    for pos, df_pos in datasets.items():
        if len(df_pos) < min_size:
            continue

        train, test = time_split(df_pos)

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train.drop(columns=[label_col, "draft_year"])
        y_train = train[label_col]

        X_test = test.drop(columns=[label_col, "draft_year"])
        y_test = test[label_col]

        for model_name, model in models.items():
            try:
                metrics = evaluate_classification_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    positive_class=positive_class,
                )
                results.append(
                    {
                        "position": pos,
                        "model": model_name,
                        **metrics,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "position": pos,
                        "model": model_name,
                        "accuracy": None,
                        "f1_macro": None,
                        "f1_weighted": None,
                        "precision_macro": None,
                        "recall_macro": None,
                        "recall_star": None,
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(results)


def get_best_classification_models(
    results_df: pd.DataFrame,
    by: str = "f1_macro",
) -> pd.DataFrame:
    """
    Get best classification model per position.
    Higher is better for accuracy, f1_macro, recall_star.
    """
    clean = results_df.dropna(subset=[by]).copy()
    best = (
        clean.sort_values(by=by, ascending=False)
        .groupby("position")
        .first()
        .reset_index()
    )
    return best


def add_classification_combined_score(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add notebook-style combined classification score:
    score = 0.5 * f1_macro + 0.2 * accuracy + 0.3 * recall_star
    Higher is better.
    """
    out = results_df.copy()
    out["score"] = (
        out["f1_macro"] * 0.5
        + out["accuracy"] * 0.2
        + out["recall_star"] * 0.3
    )
    return out