import pandas as pd

from src.data.clean_data import fill_numeric_median, keep_existing_columns
from src.utils.config import FEATURE_SETS_REFINED


def av_to_class(av: float) -> int:
    """
    Fixed AV class:
    0 = low
    1 = average
    2 = star
    """
    if av < 5:
        return 0
    if av < 15:
        return 1
    return 2


def add_fixed_av_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fixed AV class column based on dr_av thresholds.
    """
    out = df.copy()
    out["av_class"] = out["dr_av"].apply(av_to_class)
    return out


def add_quantile_labels_per_position(
    df: pd.DataFrame,
    pos_col: str = "pos_fine",
    av_col: str = "dr_av",
    min_group_size: int = 50,
    label_col: str = "av_class_q",
) -> pd.DataFrame:
    """
    Add 3-class quantile labels separately within each position group.
    """
    out = df.copy()
    out[label_col] = -1

    for pos in out[pos_col].dropna().unique():
        idx = out[out[pos_col] == pos].index

        if len(idx) <= min_group_size:
            continue

        try:
            out.loc[idx, label_col] = pd.qcut(
                out.loc[idx, av_col],
                q=3,
                labels=[0, 1, 2],
                duplicates="drop",
            )
        except ValueError:
            # If qcut fails because of duplicate bins, leave as -1
            continue

    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(-1).astype(int)
    return out


def add_star_labels_per_position(
    df: pd.DataFrame,
    pos_col: str = "pos_fine",
    av_col: str = "dr_av",
    quantile: float = 0.7,
    min_group_size: int = 50,
    label_col: str = "star",
) -> pd.DataFrame:
    """
    Add binary label per position:
    1 = star if AV is at or above the given quantile within that position
    0 = otherwise
    """
    out = df.copy()
    out[label_col] = 0

    for pos in out[pos_col].dropna().unique():
        idx = out[out[pos_col] == pos].index

        if len(idx) <= min_group_size:
            continue

        threshold = out.loc[idx, av_col].quantile(quantile)
        out.loc[idx, label_col] = (out.loc[idx, av_col] >= threshold).astype(int)

    return out


def add_all_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fixed AV labels, per-position quantile labels, and binary star labels.
    """
    out = add_fixed_av_class(df)
    out = add_quantile_labels_per_position(out)
    out = add_star_labels_per_position(out)
    return out


def get_feature_sets_refined() -> dict[str, list[str]]:
    """
    Return refined position-specific feature sets.
    """
    return FEATURE_SETS_REFINED.copy()


def split_datasets_by_position(
    df: pd.DataFrame,
    pos_col: str = "pos_fine",
) -> dict[str, pd.DataFrame]:
    """
    Split dataframe into separate dataframes by fine position.
    """
    datasets: dict[str, pd.DataFrame] = {}
    for pos in df[pos_col].dropna().unique():
        subset = df[df[pos_col] == pos].copy()
        subset = subset.dropna(subset=["dr_av"])
        datasets[pos] = subset
    return datasets


def build_regression_datasets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    pos_col: str = "pos_fine",
    fill_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build cleaned regression datasets per position.
    Each dataset contains selected features + dr_av + draft_year.
    """
    feature_sets = feature_sets or get_feature_sets_refined()
    datasets = split_datasets_by_position(df, pos_col=pos_col)

    out: dict[str, pd.DataFrame] = {}
    for pos, data in datasets.items():
        if pos not in feature_sets:
            continue

        cols = keep_existing_columns(data, feature_sets[pos] + ["dr_av", "draft_year"])
        sub = data[cols].copy()

        if fill_missing:
            sub = fill_numeric_median(sub)

        out[pos] = sub

    return out


def build_quantile_classification_datasets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    pos_col: str = "pos_fine",
    label_col: str = "av_class_q",
    fill_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build cleaned multiclass quantile datasets per position.
    Each dataset contains selected features + av_class_q + draft_year.
    """
    feature_sets = feature_sets or get_feature_sets_refined()
    datasets = split_datasets_by_position(df, pos_col=pos_col)

    out: dict[str, pd.DataFrame] = {}
    for pos, data in datasets.items():
        if pos not in feature_sets:
            continue

        cols = keep_existing_columns(data, feature_sets[pos] + [label_col, "draft_year"])
        sub = data[cols].copy()
        sub = sub[sub[label_col] != -1].copy()

        if fill_missing:
            sub = fill_numeric_median(sub)

        out[pos] = sub

    return out


def build_binary_classification_datasets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    pos_col: str = "pos_fine",
    label_col: str = "star",
    fill_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build cleaned binary classification datasets per position.
    Each dataset contains selected features + star + draft_year.
    """
    feature_sets = feature_sets or get_feature_sets_refined()
    datasets = split_datasets_by_position(df, pos_col=pos_col)

    out: dict[str, pd.DataFrame] = {}
    for pos, data in datasets.items():
        if pos not in feature_sets:
            continue

        cols = keep_existing_columns(data, feature_sets[pos] + [label_col, "draft_year"])
        sub = data[cols].copy()
        sub = sub.dropna(subset=[label_col]).copy()

        if fill_missing:
            sub = fill_numeric_median(sub)

        out[pos] = sub

    return out