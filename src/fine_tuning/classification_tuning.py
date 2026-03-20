import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.fine_tuning.grids import get_classification_param_grids
from src.models.classification import get_classification_models
from src.models.evaluation import evaluate_classifier_predictions
from src.models.split import time_split
from src.utils.config import MIN_POSITION_SIZE


def get_classification_tuning_models(
    model_names: list[str] | None = None,
    include_advanced: bool = True,
) -> dict[str, object]:
    """
    Return classification models selected for fine tuning.
    """
    models = get_classification_models(include_advanced=include_advanced)

    if model_names is None:
        return models

    return {name: model for name, model in models.items() if name in model_names}


def tune_classification_model(
    model,
    param_grid: dict,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
):
    """
    Run GridSearchCV for one classification model.
    """
    search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring = scoring,
        cv = cv,
        n_jobs = n_jobs,
        refit = True,
    )
    search.fit(X_train, y_train)

    return search


def evaluate_tuned_classification_model(
    search,
    X_test,
    y_test,
    positive_class: int = 2,
) -> dict[str, float]:
    """
    Evaluate the best tuned classifier on the test set.
    """
    preds = search.best_estimator_.predict(X_test)
    
    return evaluate_classifier_predictions(
        y_test,
        preds,
        positive_class=positive_class,
    )


def tune_classification_datasets(
    datasets: dict[str, pd.DataFrame],
    label_col: str,
    positive_class: int,
    model_names: list[str] | None = None,
    param_grids: dict[str, dict] | None = None,
    min_size: int = MIN_POSITION_SIZE,
    cv: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Fine tune classification models across position datasets.
    """
    all_models = get_classification_models(include_advanced=include_advanced)
    all_grids = (
        param_grids
        if param_grids is not None
        else get_classification_param_grids()
    )

    if model_names is None:
        selected_names = [name for name in all_models if name in all_grids]
    else:
        selected_names = [
            name
            for name in model_names
            if name in all_models and name in all_grids
        ]

    results: list[dict] = []

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

        for model_name in selected_names:
            model = all_models[model_name]
            grid = all_grids[model_name]

            try:
                search = tune_classification_model(
                    model = model,
                    param_grid = grid,
                    X_train = X_train,
                    y_train = y_train,
                    cv = cv,
                    scoring = scoring,
                    n_jobs = n_jobs,
                )

                metrics = evaluate_tuned_classification_model(
                    search,
                    X_test,
                    y_test,
                    positive_class = positive_class,
                )

                results.append(
                    {
                        "position": pos,
                        "model": model_name,
                        "label_col": label_col,
                        "positive_class": positive_class,
                        "best_params": search.best_params_,
                        "cv_best_score": float(search.best_score_),
                        **metrics,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "position": pos,
                        "model": model_name,
                        "label_col": label_col,
                        "positive_class": positive_class,
                        "best_params": None,
                        "cv_best_score": None,
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


def compare_tuned_vs_baseline_classification(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """
    Compare baseline and tuned classification results.
    """
    baseline_cols = ["position", "model", metric]
    tuned_cols = ["position", "model", metric]

    base = baseline_df[baseline_cols].copy().rename(
        columns = {
            "model": "baseline_model",
            metric: f"baseline_{metric}",
        }
    )

    tuned = tuned_df[tuned_cols].copy().rename(
        columns = {
            "model": "tuned_model",
            metric: f"tuned_{metric}",
        }
    )

    comparison = pd.merge(base, tuned, on="position", how="inner")
    comparison["improvement"] = (
        comparison[f"tuned_{metric}"] - comparison[f"baseline_{metric}"]
    )

    return comparison