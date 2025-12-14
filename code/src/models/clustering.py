import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Import configuration
from src.config import (
    CLUSTERING_CONFIGS,
    MASTER_DF_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    URBAN_CLUSTER_FILE,
)

PRESENTATION_MODE = True

# Configure Logging
log_level = logging.CRITICAL if PRESENTATION_MODE else logging.INFO

logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClusteringEngine:
    """
    A robust engine for performing, tuning, and analyzing unsupervised clustering.
    Driven by experiment configurations in config.py.
    """

    def __init__(self, experiment_name: str, seed: int = 42):
        """
        Args:
            experiment_name: Key from CLUSTERING_CONFIGS (e.g., 'urban', 'urban_cut').
            seed: Random seed for reproducibility.
        """
        if experiment_name not in CLUSTERING_CONFIGS:
            raise ValueError(
                f"Experiment '{experiment_name}' not found in CLUSTERING_CONFIGS keys."
            )

        self.experiment_name = experiment_name
        self.config = CLUSTERING_CONFIGS[experiment_name]
        self.seed = seed
        self.result_dir = PROCESSED_DATA_DIR

        self.model_dir = Path(self.config.get("model_dir", MODELS_DIR))
        self.features = self.config["features"]
        self.exclude_list = self.config.get("exclude", [])

        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.model = None
        self.labels: Optional[np.ndarray] = None
        self.data_scaled: Optional[np.ndarray] = None

        self.df_working: Optional[pd.DataFrame] = None
        self.original_df = pd.read_csv(MASTER_DF_FILE)

        logger.info(f"Initialized ClusteringEngine for experiment: '{experiment_name}'")
        logger.info(f"Target Directory: {self.model_dir}")

    def load_and_preprocess(self, original_df: pd.DataFrame) -> np.ndarray:
        """
        Prepares data specific to the experiment:
        1. Filters out excluded municipalities (e.g., Porto/Lisboa).
        2. Stores the filtered DF (with names) in self.df_working.
        3. Selects ONLY numeric features for scaling.
        """
        df = original_df.copy()

        if self.exclude_list:
            if "municipality" in df.columns:
                initial_count = len(df)
                df = df[~df["municipality"].isin(self.exclude_list)]
                dropped_count = initial_count - len(df)
                logger.info(f"Excluded {dropped_count} rows: {self.exclude_list}")
            else:
                logger.warning(
                    "Column 'municipality' not found. Cannot apply exclusions."
                )

        self.df_working = df.reset_index(drop=True)

        missing_cols = [c for c in self.features if c not in self.df_working.columns]
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        data_subset = self.df_working[self.features]

        self.data_scaled = self.scaler.fit_transform(data_subset)
        logger.info(f"Data processed. Shape: {self.data_scaled.shape}")

        return self.data_scaled

    def run_kmeans_tuning(self, max_k: int = 10) -> Tuple[plt.Figure, pd.DataFrame]:
        """Runs Elbow Method and Silhouette Analysis."""
        if self.data_scaled is None:
            raise ValueError("Data not processed. Call load_and_preprocess() first.")

        wcss = []
        sil_scores = []
        k_range = range(2, max_k + 1)

        logger.info(f"Starting K-Means tuning for k=2 to {max_k}...")

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            labels = km.fit_predict(self.data_scaled)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(self.data_scaled, labels))

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Elbow Plot
        ax1.plot(k_range, wcss, "bo--", markersize=8)
        ax1.set_title(f"Elbow Method ({self.experiment_name})")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("WCSS")
        ax1.grid(True, alpha=0.3)

        # Silhouette Plot
        ax2.plot(k_range, sil_scores, "go--", markersize=8)
        ax2.set_title(f"Silhouette Analysis ({self.experiment_name})")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        metrics_df = pd.DataFrame(
            {"k": k_range, "wcss": wcss, "silhouette": sil_scores}
        )
        return fig, metrics_df

    def run_batch_training(self):
        """
        Automatically runs and saves all models defined in the 'experiments' list
        of the configuration.
        """
        if self.data_scaled is None:
            raise ValueError("Data not processed.")

        experiments = self.config.get("experiments", [])
        if not experiments:
            logger.warning("No experiments list found in config.")
            return

        for exp in experiments:
            k = exp["n_clusters"]
            name = exp["model_name"]
            logger.info(f"Running batch training: k={k} -> {name}")
            self.train_model(n_clusters=k, model_name=name)

    def train_model(self, n_clusters: int, model_name: str = "kmeans.joblib"):
        """Trains a single KMeans model and saves it to the experiment directory."""
        self.model = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        self.labels = self.model.fit_predict(self.data_scaled)

        # Save model
        save_path = self.model_dir / model_name
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")

    def plot_saved_models_from_config(self, pca_data: np.ndarray) -> plt.Figure:
        """
        Loads models strictly defined in the config's 'experiments' list and plots them.
        """
        if self.data_scaled is None:
            raise ValueError("Data not processed. Run load_and_preprocess() first.")

        # Get list of experiments from config
        experiments = self.config.get("experiments", [])
        if not experiments:
            raise ValueError("No experiments found in config to plot.")

        n_plots = len(experiments)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, exp in enumerate(experiments):
            ax = axes[idx]
            k = exp["n_clusters"]
            filename = exp["model_name"]
            full_path = self.model_dir / filename

            if not full_path.exists():
                ax.text(
                    0.5, 0.5, f"File not found:\n{filename}", ha="center", va="center"
                )
                ax.set_axis_off()
                continue

            try:
                model = joblib.load(full_path)
                # Predict on current scaled data
                labels = model.predict(self.data_scaled)

                # Calculate score
                if len(set(labels)) > 1:
                    score = silhouette_score(self.data_scaled, labels)
                else:
                    score = 0.0

                sns.scatterplot(
                    x=pca_data[:, 0],
                    y=pca_data[:, 1],
                    hue=labels,
                    palette="viridis",
                    ax=ax,
                    legend="full",
                    s=80,
                    alpha=0.7,
                )
                ax.set_title(
                    f"K-means with k={k} clusters ({self.experiment_name})\nSilhouette: {score:.3f}"
                )
                ax.grid(True, linestyle="--", alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", color="red")

        # Hide unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig

    def get_cluster_profiles(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame with feature means and cluster counts.
        Uses self.df_working (the filtered data) and current labels.
        """
        if self.labels is None:
            raise ValueError("Model not trained or loaded. Run train_model first.")

        df_profile = self.df_working.copy()
        df_profile["Cluster"] = self.labels

        profile_cols = self.config.get("cluster_profile", self.features)
        valid_cols = [c for c in profile_cols if c in df_profile.columns]
        numeric_cols = df_profile[valid_cols].select_dtypes(include=np.number).columns

        means = df_profile.groupby("Cluster")[numeric_cols].mean()
        counts = df_profile["Cluster"].value_counts().sort_index()
        means["Count"] = counts

        # Reorder to put Count first
        cols = ["Count"] + [c for c in means.columns if c != "Count"]
        return means[cols].round(2)

    def get_cluster_members(self, name_column: str = "municipality") -> pd.DataFrame:
        """
        Returns a DataFrame listing all municipalities belonging to each cluster.
        """
        if self.labels is None:
            raise ValueError("Model not trained or loaded.")

        if name_column not in self.df_working.columns:
            raise ValueError(f"Column '{name_column}' not found in working DataFrame.")

        df_temp = self.df_working.copy()
        df_temp["Cluster"] = self.labels

        members_df = (
            df_temp.groupby("Cluster")[name_column].apply(list).to_frame(name="Members")
        )
        members_df["Count"] = members_df["Members"].apply(len)
        return members_df

    def save_labels_to_dataset(
        self,
        subset_df: pd.DataFrame,
        filepath: Union[str, Path],
        column_name: str = "cluster_label",
    ):
        """
        Saves the current cluster labels back to the master CSV file.

        CRITICAL: This relies on 'subset_df' having the same Index as the master dataset.
        Do not reset_index() on your data before passing it here.

        Args:
            subset_df: The DataFrame used for clustering (must preserve original index).
            filepath: Path to the master .csv file.
            column_name: The name of the new column (e.g., 'cluster_2022').
        """
        if self.labels is None:
            raise ValueError("Model not trained. Run train_final_model() first.")

        if len(subset_df) != len(self.labels):
            raise ValueError(
                f"Shape mismatch: Input data has {len(subset_df)} rows, but model generated {len(self.labels)} labels."
            )

        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"Master file not found at {target_path}")

        try:
            # 1. Load the Master Dataset
            # We assume this file corresponds to the source of subset_df
            logger.info(f"Loading master dataset from {target_path}...")
            master_df = pd.read_csv(target_path)

            # 2. Validation
            # Ensure the indices in our subset actually exist in the master file
            if not subset_df.index.isin(master_df.index).all():
                raise IndexError(
                    "Indices in the subset do not match the master dataset. Did you reset_index() somewhere?"
                )

            # 3. Update/Create the Column
            # If column doesn't exist, create it with NaNs (so we don't assume 0 for non-clustered rows)
            if column_name not in master_df.columns:
                master_df[column_name] = np.nan

            # 4. Map Labels to Correct Rows
            # This is the magic: .loc uses the Index to match rows, not the position.
            logger.info(
                f"Updating {len(self.labels)} rows in column '{column_name}'..."
            )
            master_df.loc[subset_df.index, column_name] = self.labels

            # 5. Save Overwrite
            master_df.to_csv(target_path, index=False)
            logger.info(f"Success! Master dataset updated.")

        except Exception as e:
            logger.error(f"Failed to update master dataset: {e}")
            raise
