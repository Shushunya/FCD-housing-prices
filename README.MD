# House Pricing in Portugal (2020-2023)

A Data Science project exploring the real estate market trends in Portugal. This project covers the full lifecycle from data ingestion and cleaning to spatial clustering and predictive modeling.

## Contributors
* **Oleksandra Chernenko** (50% contribution)
* **Gonçalo Marques** (50% contribution)

---

## Project Overview
The goal of this project is to analyze the factors influencing house prices in Portugal between 2020 and 2023. We utilize external data (Weather and Maps APIs) to enrich the core dataset and apply machine learning techniques to predict property valuations.

### Key Features
* Exploratory Data Analysis (EDA): Deep dive into price trends by region and time.
* Clustering: Geospatial analysis to identify specific market segments.
* Predictive Modeling: Linear Regression and other algorithms to estimate house prices.

---

## Project Structure

```text
house-pricing-project/
├── code/
│   ├── notebooks/          # Exploratory and experimental notebooks
|   |   ├── clustering/
|   |   ├── eda/
|   |   └── modeling/
│   ├── src/                # Modular Python scripts
│   │   ├── data/           # Data processing scripts
│   │   ├── features/       # PCA
|   |   ├── models/         # Clustering logic
│   │   ├── __init__.py
│   │   └── config.py       # Global configurations and API keys
│   └── main.ipynb
├── data/
│   ├── api/                # Data fetched from APIs (Weather, Maps)
│   ├── interm/             # Intermediate datasets (cleaned/merged)
│   ├── processed/          # Final datasets ready for modeling
│   └── raw/                # Original immutable datasets
├── models/                 # Serialized model files
├── final_report.ipynb      # Final presentation of results and insights
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Lockfile for reproducible environment
├── .python-version         # Python version specification
├── .gitignore              # Files to be excluded from version control
├── LICENSE.md              # Project license
└── README.md               # Project documentation
```
---

## Getting Started

### Prerequisites
This project uses uv for Python package management and environment isolation.

### Installation
1. Clone the repository:
   git clone https://github.com/your-username/house-pricing-portugal.git
   cd house-pricing-portugal

2. Sync the environment and install dependencies:
   uv sync

3. Activate the virtual environment:
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows

---

## Usage
1. Data Ingestion: Run the scripts in code/src/data/ to fetch and update API data.
2. Analysis: Review code/eda.ipynb for statistical summaries and visualizations.
3. Training: Execute code/modeling.ipynb to train the regression models.
4. Results: The final conclusions and model performances are documented in final_report.ipynb.

---

## License
This project is licensed under the terms of the LICENSE.md file included in this repository.