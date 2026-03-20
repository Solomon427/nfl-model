# hyperparameter grids for tuning

print("USING REDUCED GRIDS")

REGRESSION_PARAM_GRIDS = {
    "Ridge": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "RF": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTrees": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GB": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
    },
    "HistGB": {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [None, 3, 5, 8],
        "max_iter": [100, 200, 300],
        "min_samples_leaf": [10, 20, 50],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 10, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
        "kernel": ["rbf", "linear"],
    },
}
""" (Took too long)
CLASSIFICATION_PARAM_GRIDS = {
    "Logistic": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "RF": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTrees": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GB": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
    },
    "HistGB": {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [None, 3, 5, 8],
        "max_iter": [100, 200, 300],
        "min_samples_leaf": [10, 20, 50],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 10, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "NaiveBayes": {
        "var_smoothing": [1e-12, 1e-10, 1e-9, 1e-8, 1e-7],
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
}
"""
CLASSIFICATION_PARAM_GRIDS = {
    "Logistic": {
        "C": [0.1, 1.0, 10.0],
    },
    "RF": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },
    "ExtraTrees": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },
    "GB": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    },
    "HistGB": {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, None],
        "max_iter": [100, 200],
    },
    "KNN": {
        "n_neighbors": [5, 10],
        "weights": ["uniform", "distance"],
    },
    "NaiveBayes": {
        "var_smoothing": [1e-10, 1e-9, 1e-8],
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf"],
        "gamma": ["scale"],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
    "CatBoost": {
        "iterations": [100, 200],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    },
}


def get_regression_param_grids() -> dict[str, dict]:
    """
    Return a copy of the regression parameter grids.
    """
    return {name: grid.copy() for name, grid in REGRESSION_PARAM_GRIDS.items()}


def get_classification_param_grids() -> dict[str, dict]:
    """
    Return a copy of the classification parameter grids.
    """
    return {name: grid.copy() for name, grid in CLASSIFICATION_PARAM_GRIDS.items()}


def get_selected_regression_param_grids(model_names: list[str]) -> dict[str, dict]:
    """
    Return parameter grids only for the requested regression models.
    """
    grids = get_regression_param_grids()
    
    return {name: grids[name] for name in model_names if name in grids}


def get_selected_classification_param_grids(model_names: list[str]) -> dict[str, dict]:
    """
    Return parameter grids only for the requested classification models.
    """
    grids = get_classification_param_grids() 

    return {name: grids[name] for name in model_names if name in grids}