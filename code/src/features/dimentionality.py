import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UrbanizationPCA:
    """
    A class to handle loading, preprocessing, and analyzing urbanization data via PCA.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pipeline: Optional[Pipeline] = None
        self.pca_result: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []

    def load_data(self, filepath: str, features: List[str]) -> pd.DataFrame:
        """Loads data from CSV and validates features exist."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"The file {filepath} was not found.")

        try:
            df = pd.read_csv(path)
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features in dataset: {missing_features}")

            self.feature_names = features
            # logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            return df[features]
        except Exception as e:
            # logger.error(f"Error loading data: {e}")
            raise

    def run_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the PCA pipeline (Scaling + PCA) and transforms the data.
        """
        if not self.feature_names:
            self.feature_names = data.columns.tolist()

        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=self.n_components))]
        )

        try:
            components = self.pipeline.fit_transform(data)
            cols = [f"PC{i+1}" for i in range(self.n_components)]
            self.pca_result = pd.DataFrame(components, columns=cols)

            # logger.info("PCA computation completed successfully.")
            return self.pca_result
        except Exception as e:
            # logger.error(f"PCA computation failed: {e}")
            raise

    def get_loadings(self, n_display: int = 10) -> pd.DataFrame:
        """
        Returns a DataFrame of feature loadings (coefficients) for the top n_display components.
        """
        if not self.pipeline:
            raise RuntimeError("Model has not been fitted.")

        pca = self.pipeline.named_steps["pca"]

        loadings = pca.components_.T
        n_pcs = min(loadings.shape[1], n_display)

        cols = [f"PC{i+1}" for i in range(n_pcs)]
        df_loadings = pd.DataFrame(
            loadings[:, :n_pcs], columns=cols, index=self.feature_names
        )

        return df_loadings

    def plot_component_details(self, n_components: int = 2) -> plt.Figure:
        """
        Plots the feature loadings for the specified components as sorted bar charts.
        This helps interpret exactly what each Principal Component represents.
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted. Run 'run_pca' first.")

        df_loadings = self.get_loadings(n_display=n_components)

        # Create subplots (one for each PC)
        fig, axes = plt.subplots(1, n_components, figsize=(15, 3 * n_components))
        if n_components == 1:
            axes = [axes]  # Handle single plot edge case

        for i, ax in enumerate(axes):
            pc_col = f"PC{i+1}"

            # Sort by coefficient magnitude for clearer reading
            subset = df_loadings[pc_col].sort_values(ascending=True)

            # Color code: Red for negative, Blue for positive
            colors = ["#d62728" if x < 0 else "#1f77b4" for x in subset.values]

            subset.plot(kind="barh", ax=ax, color=colors, width=0.3, alpha=0.8)

            # Aesthetics
            ax.set_title(f"Composition of {pc_col}", fontsize=14, pad=15)
            ax.set_xlabel("Feature Loading (Correlation)", fontsize=10)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            # ax.grid(axis="x", linestyle="--", alpha=0.3)

            # Optional: Add text summary on the plot
            stats = self.get_variance_stats()
            var_txt = f"{stats.get(f'pc{i+1}_var', 0):.1f}% Variance"
            ax.text(
                0.98,
                0.05,
                var_txt,
                transform=ax.transAxes,
                ha="right",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        return fig

    def get_variance_stats(self) -> Dict[str, float]:
        """Returns explained variance ratios without re-running PCA."""
        if not self.pipeline:
            raise RuntimeError("Model has not been fitted. Run 'run_pca' first.")

        pca_step = self.pipeline.named_steps["pca"]
        exp_ratio = pca_step.explained_variance_ratio_
        result = {f"pc{i+1}_var": exp_ratio[i] * 100 for i in range(len(exp_ratio))}
        result.update({"total_var": sum(exp_ratio) * 100})

        return result

    def plot_projection(self, title: str = "Urbanization Data Structure") -> plt.Figure:
        """
        Plots the first two principal components.
        """
        if self.pca_result is None:
            raise RuntimeError("No PCA results found. Run 'run_pca' first.")

        stats = self.get_variance_stats()

        fig, ax = plt.subplots(figsize=(15, 6))

        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=self.pca_result,
            ax=ax,
            s=100,
            alpha=0.7,
            edgecolor="k",
        )

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel(
            f'PC1: Infrastructure ({stats["pc1_var"]:.2f}% Variance)', fontsize=12
        )
        ax.set_ylabel(
            f'PC2: Wealth & Density ({stats["pc2_var"]:.2f}% Variance)', fontsize=12
        )
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add a text box for total variance
        textstr = f'Total Explained Variance: {stats["total_var"]:.2f}%'
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=props,
        )

        plt.tight_layout()
        return fig
