# House pricing in Portugal (2020-2023) Data Science project

Contributors: Oleksandra Chernenko and Goncalo Marques 

## Project structure (draft)

house-pricing-project/
├── data/
│   ├── raw/                  # Original immutable datasets
│   ├── external/             # Data fetched from APIs (Weather, Maps)
│   ├── interm/               # Intermediate data
│   └── processed/            # Final datasets for modeling
├── models/                   # Saved .pkl/.joblib model files 
│   ├── cluster_model.pkl     # Saved K-Means model
│   └── housing_lin_reg.pkl   # Saved Linear Regression model
├── notebooks/
│   ├── eda.ipynb
│   ├── clustering.ipynb
│   └── modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── api_fetcher.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── clustering.py
│   └── models/               # Code to TRAIN the models
│       ├── train_model.py
│       └── predict_model.py
├── pyproject.toml
├── uv.lock
└── README.md
