from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard paths for data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERM_DATA_DIR = DATA_DIR / "interm"  # Cleaned data but not the final dataset
PROCESSED_DATA_DIR = DATA_DIR / "processed"
API_DATA_DIR = DATA_DIR / "api"

# Raw data files
HOUSE_RAW_FILE = RAW_DATA_DIR / "house_pricing_raw.csv"
HOUSE_RAW_EXCEL_FILE = RAW_DATA_DIR / "Destaque_HAB_1T2025_EN.xlsx"
INCOME_RAW_FILE = RAW_DATA_DIR / "582.csv"
DENSITY_RAW_FILE = RAW_DATA_DIR / "density.csv"
AGE_RAW_FILE = RAW_DATA_DIR / "age-distribution.csv"

# API data files
WEATHER_FILE = API_DATA_DIR / "weather.csv"
SERVICES_FILE = API_DATA_DIR / "osm_services_counts.csv"

# interm files
MUNICIPALITIES_FILE = INTERM_DATA_DIR / "municipalities.csv"
HOUSE_CLEAN_FILE = INTERM_DATA_DIR / "house_cleaned.csv"
INCOME_CLEAN_TOTAL = INTERM_DATA_DIR / "total_average_income_by_municipality.csv"
INCOME_CLEAN_EDU = INTERM_DATA_DIR / "average_income_by_education.csv"
DENSITY_CLEAN_FILE = INTERM_DATA_DIR / "density_by_municipality.csv"
AGE_CLEAN_FILE = INTERM_DATA_DIR / "age_distribution_by_municipality.csv"
WEATHER_QUARTER_FILE = INTERM_DATA_DIR / "weather_quarterly.csv"

# processed files for modeling
MASTER_DF_FILE = PROCESSED_DATA_DIR / "all_raw_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"


# project constants
# eda
COLUMN_MISSING_VALUES = "nan count"
# modeling
RANDOM_SEED = 42
SPLIT_SIZE = 0.2
MAX_ITER = 4000
ALPHAS = np.logspace(-6, 6, 40)
# clustering

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

AGE_COLUMNS = ["< 5", "6 - 19", "20-34", "35 - 54", "> 55"]

WEATHER_RAW_COLUMNS = [
    "total_sunshine_h",
    "mean_sunshine_h",
    "windspeed_mean_kmh",
    "total_precipitation_mm",
    "mean_precipitation_mm",
]
WEATHER_DAYS_COLUMNS = ["windy_days", "rainy_days", "sunny_days", "warm_days"]
WEATHER_COLUMNS = WEATHER_RAW_COLUMNS + WEATHER_DAYS_COLUMNS


INCOME_COLUMN = "avg_income"
POP_DENSITY_COLUMN = "people/km2"
TIME_COLUMNS = ["quarter_num", "quarter_ord"]
YEAR_COLUMN = "year"


# CLUSTERING CONFIGURATION
CLUSTERING_CONFIG = {
    "urban": [POP_DENSITY_COLUMN, INCOME_COLUMN] + SERVICES_COLUMNS,
    "age": AGE_COLUMNS,
}

# CLUSTERING_PARAMS = {
#     "geo_features": {"n_clusters": 5, "algorithm": "kmeans"},
#     "house_features": {"n_clusters": 4, "algorithm": "kmeans"},
# }


# REGRESSION CONFIGURATION

REGRESSION_FEATURES = {
    # Numerical features to be Scaled
    "numerical": [
        "sqft_living",
        "sqft_lot",
        "bedrooms",
        "bathrooms",
        "yr_built",
        "floors",
        "view",
        "condition",
    ],
    # Categorical features to be One-Hot Encoded.
    # Logic: f"cluster_{key}" from CLUSTERING_CONFIG
    "categorical": ["cluster_urban", "cluster_age"],
}

# MODEL SELECTION
MODELS_TO_TRAIN = {
    "linear": "LinearRegression",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
    "gbm": "GradientBoostingRegressor",
    # "xgb": "XGBoost" # Uncomment if you install xgboost
}
