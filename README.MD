# House pricing in Portugal (2020-2023) Data Science project

Contributors: Oleksandra Chernenko and Goncalo Marques 

## Project structure (draft)
house-pricing-project/
├── data/
│   ├── api/                  # Data fetched from APIs (Weather, Maps)
│   ├── interm/             # Intermediate datasets
│   ├── processed/               # Final datasets for modeling
│   └── raw/            # Original immutable datasets
├── models/                   # Saved .pkl/.joblib model files 
│   └── housing_lin_reg.pkl   # Saved Linear Regression model
├── code/
|   ├── notebooks/
|   ├── src/
|   |   ├── data/
|   |   ├── features/
|   |   ├── __init__.py
|   |   └── config.py
|   ├── clustering.ipynb
|   ├── eda.ipynb
|   ├── main.ipynb
|   └── modeling.ipynb
├── pyproject.toml
├── final_report.ipynb
├── uv.lock
├── .gitignore
├── .python-version
├── LICENSE.MD
└── README.md
