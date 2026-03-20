import pandas as pd

from src.utils.config import SPLIT_YEAR


def time_split(
    df: pd.DataFrame,
    year_col: str = "draft_year",
    split_year: int = SPLIT_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split.

    Train: year <= split_year
    Test:  year > split_year
    """
    train = df[df[year_col] <= split_year].copy()
    test = df[df[year_col] > split_year].copy()
    return train, test