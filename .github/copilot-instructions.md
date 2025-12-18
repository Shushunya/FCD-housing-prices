# AI Coding Agent Instructions - Portuguese Housing Price Analysis

## Project Overview
This is a data science project analyzing Portuguese house pricing (2020-2023) across municipalities, incorporating urbanization clustering, weather data, demographics, and local services to predict housing prices per square meter. The project uses a centralized configuration system and notebook-driven workflows.

## Architecture & Data Flow

### Core Data Pipeline
1. **Raw data** (`data/raw/`) → **Intermediate** (`data/interm/`) → **Processed** (`data/processed/`)
2. Key datasets:
   - `house_pricing_raw.csv` - Raw housing prices
   - External APIs: weather data, OSM services
   - Demographics: age distribution, income, population density
3. Master dataset: `all_raw_features.csv` → enriched with clusters → `urban_cluster.csv`

### Configuration-Driven Design
**All paths, features, and experiments are defined in `code/src/config.py`**

- File paths: `HOUSE_RAW_FILE`, `MASTER_DF_FILE`, `URBAN_CLUSTER_FILE`, etc.
- Feature groups: `SERVICES_COLUMNS`, `AGE_COLUMNS`, `WEATHER_COLUMNS`, `URBAN_FEATURES`
- Constants: `RANDOM_SEED = 42`, `MUNICIPALITY_COLUMN`, `INCOME_COLUMN`, etc.
- Clustering experiments: `CLUSTERING_CONFIGS` dictionary defines features, exclusions, and k-range
- Always import from `src.config` - never hardcode paths or feature lists

### Module Structure
```
code/
├── src/
│   ├── config.py          # SINGLE SOURCE OF TRUTH for all configs
│   ├── data/
│   │   └── loader.py      # DataLoader class for filtering & loading
│   ├── features/
│   │   └── dimentionality.py  # UrbanizationPCA class
│   └── models/
│       └── clustering.py  # ClusteringEngine class
├── notebooks/             # Analysis notebooks
│   ├── clustering/
│   └── eda/
├── main.ipynb, eda.ipynb, modeling.ipynb  # Main workflows
```

## Key Patterns & Conventions

### 1. Import Pattern
```python
from src.config import (
    MASTER_DF_FILE,
    RANDOM_SEED,
    MUNICIPALITY_COLUMN,
    SERVICES_COLUMNS,
    URBAN_FEATURES
)
from src.data.loader import DataLoader
from src.models.clustering import ClusteringEngine
```

### 2. Feature Group Usage
When working with features, use predefined groups from config:
- `URBAN_FEATURES` = population density + income + all services
- `SERVICES_COLUMNS` = 16 amenity types (cinema, hospital, school, etc.)
- `AGE_COLUMNS` = 6 age buckets (note: format is `"< 5"`, `"6 - 19"`, etc.)
- `WEATHER_COLUMNS` = raw metrics + derived day counts

### 3. Clustering Workflow
Clustering uses **experiment-based configuration** in `CLUSTERING_CONFIGS`:
```python
engine = ClusteringEngine(experiment_name="urban", seed=RANDOM_SEED)
engine.load_and_preprocess(MASTER_DF_FILE)
engine.run_batch_training()  # Trains all k-values defined in config
```

Each experiment config includes:
- `features`: which columns to cluster on
- `experiments`: list of k-values to train
- `exclude`: optional municipalities to drop (e.g., outliers like Lisboa, Porto)

### 4. Data Loading with Filtering
Use `DataLoader` for consistent data access with exclusions:
```python
loader = DataLoader(MASTER_DF_FILE)
df = loader.load_dataset(
    features=config["features"],
    exclude_values=["Porto", "Lisboa"]
)
```

### 5. Target Variable Convention
The regression target is always **`log_price_sqm`** (log-transformed price per square meter).

### 6. Cross-Validation Strategy
Use **GroupKFold** with `groups=municipality` to prevent data leakage across municipalities:
```python
from sklearn.model_group import GroupKFold
gkf = GroupKFold(n_splits=5)
cross_validate(pipeline, X, y, cv=gkf, groups=groups)
```

### 7. Temporal Validation Pattern
The project uses temporal train/test splits:
- Training: 2019-2022 data
- Testing: 2023 data (holdout)
- See `modeling.ipynb` cells starting around line 312

## Development Workflows

### Environment Setup
```bash
# Project uses uv for dependency management
# Python 3.11+ required (see .python-version)
# Dependencies in pyproject.toml
```

### Notebook Execution Order
1. **Data preparation**: `code/notebooks/data_preparation.ipynb` - age bucket aggregation
2. **EDA**: `code/eda.ipynb` - exploratory analysis
3. **Clustering**: `code/notebooks/clustering/01_urbanization_clustering_all.ipynb`
4. **Modeling**: `code/modeling.ipynb` - regression models with temporal validation

### Adding New Features
1. Define column names in `src/config.py` under appropriate section
2. Add to relevant feature group (e.g., append to `SERVICES_COLUMNS`)
3. Update data loading logic in notebooks/scripts
4. Feature groups are concatenated: `feature_cols = numerical_features + categorical_features`

### Model Training Pattern
Models follow scikit-learn Pipeline convention:
```python
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([...])),
    ("regressor", model_class(**params))
])
pipeline.fit(X, y)
```

Trained models saved to `models/` as `.joblib` files.

## Project-Specific Quirks

### Column Naming
- Age columns have **spaces with comparison operators**: `"< 5"`, `"> 65"`, `"6 - 19"`
- Municipality column: always referenced as `MUNICIPALITY_COLUMN` from config
- Weather features: distinguish between `WEATHER_RAW_COLUMNS` (API) and `WEATHER_DAYS_COLUMNS` (derived)

### Presentation Mode
Classes like `DataLoader` and `ClusteringEngine` have `PRESENTATION_MODE` flag that suppresses logging:
```python
PRESENTATION_MODE = True  # Set to False for debugging
```

### Path Construction
Always use `Path` objects (from `pathlib`), not strings:
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

### Cluster Labels
Cluster interpretations defined in `CLUSTER_LABELS` dict in config:
```python
CLUSTER_LABELS = {
    "cluster_urban": {
        0: "Urban", 1: "Porto", 2: "Suburbs", 
        3: "Rural", 4: "Lisbon"
    }
}
```

## Testing & Validation

### Model Evaluation Metrics
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Track both CV performance and test performance separately

### Results Storage
- Cross-validation results: `models/cv_results.csv`
- Temporal validation: `models/temporal_validation_results.csv`
- Feature importance: `models/all_models_feature_importance.csv`

## Common Tasks

### Adding a New Clustering Experiment
1. Add to `CLUSTERING_CONFIGS` in `src/config.py`:
```python
"new_experiment": {
    "features": [...],
    "model_dir": MODELS_DIR / "experiment_name",
    "experiments": [{"n_clusters": k, "model_name": f"model_{k}.joblib"} for k in range(3, 6)]
}
```
2. Use `ClusteringEngine("new_experiment")` in notebook

### Modifying Feature Sets
Edit feature lists in `src/config.py` - they are imported and used throughout notebooks. Never define feature lists locally in notebooks.

### Adding New Model Types
Update `model_configs` dict in `modeling.ipynb` with sklearn-compatible regressors:
```python
model_configs = {
    "model_key": {"class": ModelClass, "params": {...}}
}
```

## Files to Check First
- `code/src/config.py` - All constants, paths, feature definitions
- `code/modeling.ipynb` - Current model training setup (attached)
- `data/processed/urban_cluster.csv` - Final dataset for modeling
- `README.MD` - Project structure overview
