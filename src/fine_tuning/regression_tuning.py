import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.models.evaluation import evaluate_regression_predictions
from src.models.regression import get_regression_models
from src.models.split import time_split
from src.utils.config import MIN_POSITION_SIZE
from src.fine_tuning.grids import get_regression_param_grids

# Regression tuning module.
# Contains functions to fine tune regression models across position datasets.

def get_regression_tuning_models(model_names: list[str] | None = None) -> dict[str, object]:
    """
    Return regression models selected for fine tuning.
    """
    models = get_regression_models()

    if model_names is None:
        return models

    return {name: model for name, model in models.items() if name in model_names}


def tune_regression_model(
    model,
    param_grid: dict,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
):
    """
    Run GridSearchCV for one regression model.
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


def evaluate_tuned_regression_model(search, X_test, y_test) -> dict[str, float]:
    """
    Evaluate the best tuned regression estimator on the test set.
    """
    preds = search.best_estimator_.predict(X_test)

    return evaluate_regression_predictions(y_test, preds)


def tune_regression_datasets(
    datasets: dict[str, pd.DataFrame],
    model_names: list[str] | None = None,
    param_grids: dict[str, dict] | None = None,
    min_size: int = MIN_POSITION_SIZE,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Fine tune regression models across position datasets.
    """
    all_models = get_regression_models()
    all_grids = param_grids if param_grids is not None else get_regression_param_grids()

    if model_names is None:
        selected_names = [name for name in all_models if name in all_grids]
    else:
        selected_names = [name for name in model_names if name in all_models and name in all_grids]

    results: list[dict] = []

    for pos, df_pos in datasets.items():
        if len(df_pos) < min_size:
            continue

        train, test = time_split(df_pos)

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train.drop(columns=["dr_av", "draft_year"])
        y_train = train["dr_av"]

        X_test = test.drop(columns=["dr_av", "draft_year"])
        y_test = test["dr_av"]

        for model_name in selected_names:
            model = all_models[model_name]
            grid = all_grids[model_name]

            try:
                search = tune_regression_model(
                    model=model,
                    param_grid=grid,
                    X_train=X_train,
                    y_train=y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                )

                metrics = evaluate_tuned_regression_model(search, X_test, y_test)

                results.append(
                    {
                        "position": pos,
                        "model": model_name,
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
                        "best_params": None,
                        "cv_best_score": None,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(results)


def compare_tuned_vs_baseline_regression(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    metric: str = "rmse",
) -> pd.DataFrame:
    """
    Compare baseline and tuned regression results.
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

    if metric in ["rmse", "mae"]:
        comparison["improvement"] = (
            comparison[f"baseline_{metric}"] - comparison[f"tuned_{metric}"]
        )
    else:
        comparison["improvement"] = (
            comparison[f"tuned_{metric}"] - comparison[f"baseline_{metric}"]
        )

    return comparison