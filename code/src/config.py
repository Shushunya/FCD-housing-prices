from pathlib import Path

import numpy as np

# ==============================================================================
# 1. PROJECT STRUCTURE & PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERM_DATA_DIR = DATA_DIR / "interm"  # Cleaned data but not the final dataset
PROCESSED_DATA_DIR = DATA_DIR / "processed"
API_DATA_DIR = DATA_DIR / "api"

MODELS_DIR = PROJECT_ROOT / "models"
URBAN_CLUSTER_DIR = MODELS_DIR / "urban_clusters"

# Raw data files
HOUSE_RAW_FILE = RAW_DATA_DIR / "house_pricing_raw.csv"
HOUSE_RAW_EXCEL_FILE = RAW_DATA_DIR / "Destaque_HAB_1T2025_EN.xlsx"
INCOME_RAW_FILE = RAW_DATA_DIR / "582.csv"
DENSITY_RAW_FILE = RAW_DATA_DIR / "density.csv"
AGE_RAW_FILE = RAW_DATA_DIR / "age-distribution.csv"

# API data files
WEATHER_FILE = API_DATA_DIR / "weather.csv"
SERVICES_FILE = API_DATA_DIR / "osm_services_counts.csv"

# Intermediate files
MUNICIPALITIES_FILE = INTERM_DATA_DIR / "municipalities.csv"
HOUSE_CLEAN_FILE = INTERM_DATA_DIR / "house_cleaned.csv"
INCOME_CLEAN_TOTAL = INTERM_DATA_DIR / "total_average_income_by_municipality.csv"
INCOME_CLEAN_EDU = INTERM_DATA_DIR / "average_income_by_education.csv"
DENSITY_CLEAN_FILE = INTERM_DATA_DIR / "density_by_municipality.csv"
AGE_CLEAN_FILE = INTERM_DATA_DIR / "age_distribution_by_municipality.csv"
WEATHER_QUARTER_FILE = INTERM_DATA_DIR / "weather_quarterly.csv"

# Processed files for modeling
MASTER_DF_FILE = PROCESSED_DATA_DIR / "all_raw_features.csv"
URBAN_CLUSTER_FILE = PROCESSED_DATA_DIR / "urban_cluster.csv"

# ==============================================================================
# 2. COLUMN DEFINITIONS & FEATURE GROUPS
# ==============================================================================
# Single Columns
MUNICIPALITY_COLUMN = "municipality"
INCOME_COLUMN = "avg_income"
POP_DENSITY_COLUMN = "people/km2"
YEAR_COLUMN = "year"

# Temporary Columns
COLUMN_MISSING_VALUES = "nan count"

# Time columns & values
YEARS = [2020, 2021, 2022, 2023]
TIME_COLUMNS = ["quarter_num", "quarter_ord"]

# Feature lists
SERVICES_COLUMNS = [
    "cinema",
    "college",
    "courthouse",
    "fire_station",
    "hospital",
    "kindergarten",
    "library",
    "mall",
    "museum",
    "pharmacy",
    "police",
    "post_office",
    "school",
    "station",
    "theatre",
    "university",
]

AGE_COLUMNS = ["< 5", "6 - 19", "20 - 34", "35 - 54", "55 - 64", "> 65"]

WEATHER_RAW_COLUMNS = [
    "total_sunshine_h",
    "mean_sunshine_h",
    "windspeed_mean_kmh",
    "total_precipitation_mm",
    "mean_precipitation_mm",
]
WEATHER_DAYS_COLUMNS = ["windy_days", "rainy_days", "sunny_days", "warm_days"]
WEATHER_COLUMNS = WEATHER_RAW_COLUMNS + WEATHER_DAYS_COLUMNS

# Composite feature sets
URBAN_FEATURES = [POP_DENSITY_COLUMN, INCOME_COLUMN] + SERVICES_COLUMNS
URBAN_PROFILE = [MUNICIPALITY_COLUMN] + URBAN_FEATURES

# ==============================================================================
# 3. MODELING HYPERPARAMETERS & CONSTANTS
# ==============================================================================
RANDOM_SEED = 42
SPLIT_SIZE = 0.2
MAX_ITER = 4000
ALPHAS = np.logspace(-6, 6, 40)

MODELS_TO_TRAIN = {
    "linear": "LinearRegression",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
    "gbm": "GradientBoostingRegressor",
    # "xgb": "XGBoost"
}

# ==============================================================================
# 4. EXPERIMENT CONFIGURATIONS
# ==============================================================================

# K-Means Clustering Configurations
CLUSTERING_CONFIGS = {
    # Experiment 1: Urbanization Tiers (All Data)
    "urban": {
        "features": URBAN_FEATURES,
        "model_dir": URBAN_CLUSTER_DIR,
        "cluster_profile": URBAN_PROFILE,
        "model_name_pattern": "urban_kmeans_{k}.joblib",
        "experiments": [
            {"n_clusters": k, "model_name": f"urban_kmeans_{k}.joblib"}
            for k in range(3, 7)
        ],
    },
    # Experiment 2: Urbanization Tiers (Removing Outliers: Porto/Lisboa)
    "urban_cut": {
        "features": URBAN_FEATURES,
        "model_dir": URBAN_CLUSTER_DIR,
        "cluster_profile": URBAN_PROFILE,
        "model_name_pattern": "urban_cut_kmeans_{k}.joblib",
        "exclude": ["Porto", "Lisboa"],
        "experiments": [
            {"n_clusters": k, "model_name": f"urban_cut_kmeans_{k}.joblib"}
            for k in range(3, 5)
        ],
    },
}

CLUSTER_LABELS = {
    "cluster_urban": {
        0: "Rural",
        1: "Suburbs",
        2: "Lisbon",
        3: "Porto",
        4: "Urban",
        5: "Historical Hub",
    }
}

# Regression Feature Selection
REGRESSION_FEATURES = {
    "numerical": [""],
    "categorical": ["cluster_urban", "cluster_age"],
}
