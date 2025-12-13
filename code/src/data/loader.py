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

    def load_dataset(
        self,
        features: Optional[List[str]] = None,
        exclude_column: str = "municipality",
        exclude_values: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Loads the dataset, optionally filtering columns and excluding specific rows.

        Args:
            features: List of columns to keep.
            exclude_column: The column to check for exclusions (default: "municipality").
            exclude_values: List of values to drop (e.g., ["Lisboa", "Porto"]).
        """
        try:
            df = pd.read_csv(self.filepath)

            if exclude_values:
                if exclude_column not in df.columns:
                    raise ValueError(
                        f"Exclusion column '{exclude_column}' not found in dataset."
                    )

                initial_count = len(df)
                # Filter out rows where the column value is in the exclude list
                df = df[~df[exclude_column].isin(exclude_values)]

                dropped_count = initial_count - len(df)
                if dropped_count > 0:
                    logger.info(
                        f"Excluded {dropped_count} rows matching {exclude_values} in '{exclude_column}'."
                    )

            if features:
                df = self._select_features(df, features)

            logger.info(f"Successfully loaded dataset with {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def get_data_for_year(
        self,
        year: int,
        date_column: str = "year",
        features: Optional[List[str]] = None,
        exclude_column: str = "municipality",
        exclude_values: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Filters data by year, handles exclusions, then returns requested features.
        """
        try:
            df = pd.read_csv(self.filepath)

            if date_column not in df.columns:
                raise ValueError(f"Filtering column '{date_column}' not found.")

            df = df[df[date_column] == year].copy()

            if df.empty:
                logger.warning(f"No data found for year {year}.")
                cols = features if features else df.columns
                return pd.DataFrame(columns=cols)

            if exclude_values:
                if exclude_column not in df.columns:
                    raise ValueError(f"Exclusion column '{exclude_column}' not found.")

                df = df[~df[exclude_column].isin(exclude_values)]
                logger.info(
                    f"Year {year}: Excluded {len(exclude_values)} specific values from '{exclude_column}'."
                )

            if features:
                df = self._select_features(df, features)

            logger.info(f"Retrieved {len(df)} rows for year {year}.")
            return df

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
