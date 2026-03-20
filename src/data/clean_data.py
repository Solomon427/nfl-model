import pandas as pd

from src.utils.config import POSITION_GROUP_MAP, POSITION_FINE_MAP


def map_position(pos: str) -> str:
    """
    Map NFL position into broad groups.

    Returns one of:
    QB, SKILL, DEFENSE, OL, OTHER
    """
    if pd.isna(pos):
        return "OTHER"

    for group, positions in POSITION_GROUP_MAP.items():
        if pos in positions:
            return group
    return "OTHER"


def map_pos_fine(pos: str) -> str:
    """
    Map NFL position into fine-grained groups.

    Returns one of:
    QB, RB, WR, TE, OL, DL, LB, DB, OTHER
    """
    if pd.isna(pos):
        return "OTHER"

    for group, positions in POSITION_FINE_MAP.items():
        if pos in positions:
            return group
    return "OTHER"


def add_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add broad and fine position group columns.
    """
    out = df.copy()
    out["pos_group"] = out["pos_nfl"].apply(map_position)
    out["pos_fine"] = out["pos_nfl"].apply(map_pos_fine)
    return out


def fill_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with the median of each numeric column.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())
    return out


def keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Return only columns that actually exist in the dataframe.
    """
    return [col for col in columns if col in df.columns]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning used before modeling:
    - add position columns
    - drop rows missing dr_av
    """
    out = add_position_columns(df)
    out = out.dropna(subset=["dr_av"]).copy()
    return out