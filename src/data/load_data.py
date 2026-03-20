from pathlib import Path
import pandas as pd

from src.utils.config import FINAL_MERGED_DATA_PATH


def load_final_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the final merged NFL player dataset.

    Parameters
    ----------
    path : str | Path | None
        Optional override path. If None, uses FINAL_MERGED_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    file_path = Path(path) if path is not None else FINAL_MERGED_DATA_PATH
    return pd.read_csv(file_path)