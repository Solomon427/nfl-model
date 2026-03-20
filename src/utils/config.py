from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Main input file
FINAL_MERGED_DATA_PATH = PROCESSED_DATA_DIR / "final_merged_player_data.csv"

# Experiment settings
RANDOM_STATE = 42
SPLIT_YEAR = 2018
MIN_POSITION_SIZE = 100

# Common combine / metadata features
COMMON_FEATURES = [
    "ht",
    "wt",
    "forty",
    "bench",
    "vertical",
    "broad_jump",
    "cone",
    "shuttle",
    "age",
]

# Position maps
POSITION_GROUP_MAP = {
    "QB": ["QB"],
    "SKILL": ["RB", "WR", "TE"],
    "DEFENSE": ["CB", "S", "DB", "LB", "ILB", "OLB", "DT", "DE", "EDGE"],
    "OL": ["OT", "OG", "C"],
}

POSITION_FINE_MAP = {
    "QB": ["QB"],
    "RB": ["RB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "OL": ["OT", "OG", "C"],
    "DL": ["DT", "DE", "EDGE"],
    "LB": ["LB", "ILB", "OLB"],
    "DB": ["CB", "S"],
}

# Original broad feature groups
QB_FEATURES = [
    "qb_Cmp",
    "qb_Att",
    "qb_Cmp%",
    "qb_Yds",
    "qb_TD",
    "qb_Int",
    "qb_Y/A",
    "qb_AY/A",
    "qb_Y/G",
]

SKILL_FEATURES = [
    "rb_rush_Att",
    "rb_rush_Yds",
    "rb_rush_Y_per_Att",
    "rb_rush_TD",
    "rb_rec_Rec",
    "rb_rec_Yds",
    "rb_rec_Y_per_Rec",
    "rb_rec_TD",
    "rec_Rec",
    "rec_Yds",
    "rec_Y/R",
    "rec_TD",
    "rec_Y/G",
]

DEFENSE_FEATURES = [
    "Sk",
    "Solo",
    "Ast",
    "Comb",
    "TFL",
    "Int",
    "PD",
]

# Refined feature sets used in final experiments
FEATURE_SETS_REFINED = {
    "QB": [
        "qb_Cmp",
        "qb_Att",
        "qb_Cmp%",
        "qb_Yds",
        "qb_TD",
        "qb_Int",
        "qb_Y/A",
        "qb_AY/A",
        "qb_Y/G",
    ] + COMMON_FEATURES,
    "RB": [
        "rb_rush_Att",
        "rb_rush_Yds",
        "rb_rush_Y_per_Att",
        "rb_rush_TD",
        "rb_rec_Rec",
        "rb_rec_Yds",
        "rb_rec_Y_per_Rec",
        "rb_rec_TD",
    ] + COMMON_FEATURES,
    "WR": [
        "rec_Rec",
        "rec_Yds",
        "rec_Y/R",
        "rec_TD",
        "rec_Y/G",
    ] + COMMON_FEATURES,
    "TE": [
        "rec_Rec",
        "rec_Yds",
        "rec_Y/R",
        "rec_TD",
        "rec_Y/G",
    ] + COMMON_FEATURES,
    "DL": [
        "Sk",
        "TFL",
        "Solo",
    ] + COMMON_FEATURES,
    "LB": [
        "Solo",
        "Ast",
        "Comb",
        "TFL",
    ] + COMMON_FEATURES,
    "DB": [
        "Int",
        "PD",
        "Solo",
    ] + COMMON_FEATURES,
    "OL": COMMON_FEATURES,
}