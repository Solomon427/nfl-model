import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.models.evaluation import evaluate_regression_predictions
from src.models.split import time_split
from src.utils.config import MIN_POSITION_SIZE, RANDOM_STATE


def get_regression_models() -> dict[str, object]:
    """
    Return regression models used in the notebook.
    """
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=10),
        "SVR": SVR(),
    }


def evaluate_regression_model(model, X_train, y_train, X_test, y_test) -> dict[str, float]:
    """
    Fit model and compute regression metrics.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_regression_predictions(y_test, preds)


def benchmark_regression(
    datasets: dict[str, pd.DataFrame],
    min_size: int = MIN_POSITION_SIZE,
) -> pd.DataFrame:
    """
    Benchmark all regression models across position datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Position -> dataframe with features + dr_av + draft_year
    min_size : int
        Minimum dataset size to run

    Returns
    -------
    pd.DataFrame
        Results table.
    """
    results: list[dict] = []
    models = get_regression_models()

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

        for model_name, model in models.items():
            try:
                metrics = evaluate_regression_model(model, X_train, y_train, X_test, y_test)
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
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(results)


def get_best_regression_models(results_df: pd.DataFrame, by: str = "rmse") -> pd.DataFrame:
    """
    Get best regression model per position.
    Lower is better for rmse / mae.
    Higher is better for r2, so pass a sorted dataframe manually if needed.
    """
    clean = results_df.dropna(subset=[by]).copy()
    best = (
        clean.sort_values(by=by, ascending=True)
        .groupby("position")
        .first()
        .reset_index()
    )
    return best


def add_regression_combined_score(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add notebook-style combined score:
    score = 0.5 * rmse + 0.3 * mae - 10 * r2
    Lower is better.
    """
    out = results_df.copy()
    out["score"] = out["rmse"] * 0.5 + out["mae"] * 0.3 - out["r2"] * 10
    return out