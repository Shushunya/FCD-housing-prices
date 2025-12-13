import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A robust data loader for fetching, filtering, and validating urbanization datasets.
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"The file {self.filepath} was not found.")

    def load_dataset(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Loads the dataset, optionally filtering for specific features.
        """
        try:
            df = pd.read_csv(self.filepath)

            if features:
                df = self._select_features(df, features)

            logger.info(f"Successfully loaded dataset with {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def get_data_for_year(
        self, year: int, date_column: str = "year", features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filters data by year, then returns only the requested features.

        Args:
            year: The target year (int).
            date_column: The name of the column to filter by (default 'year').
            features: The list of columns to return. The 'year' column does NOT need to be in this list.
        """
        try:
            # 1. Load FULL dataset first (so we have the 'year' column available for filtering)
            df = pd.read_csv(self.filepath)

            # 2. Validation
            if date_column not in df.columns:
                raise ValueError(
                    f"Filtering column '{date_column}' not found in dataset."
                )

            # 3. Filter Rows
            filtered_df = df[df[date_column] == year].copy()

            if filtered_df.empty:
                logger.warning(f"No data found for year {year}.")
                # Return empty DF with requested columns (or all if none requested)
                cols = features if features else df.columns
                return pd.DataFrame(columns=cols)

            # 4. Select Columns (The clean-up step)
            # If 'features' is provided, we subset now.
            # This allows you to exclude 'year' from the final result.
            if features:
                filtered_df = self._select_features(filtered_df, features)

            logger.info(f"Retrieved {len(filtered_df)} rows for year {year}.")
            return filtered_df

        except Exception as e:
            logger.error(f"Failed to get data for year {year}: {e}")
            raise

    def _select_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Private helper to validate and select specific columns.
        """
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Requested features not found in dataset: {missing}")

        return df[features]
