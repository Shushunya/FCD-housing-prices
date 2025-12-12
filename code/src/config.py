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
CLUSTERING_COLS = ["latitude", "longitude"]

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
