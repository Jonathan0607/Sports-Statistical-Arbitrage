"""Local and global TreeSHAP explainability for XGBoost residual models."""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_contract import (
    FEATURE_COLUMNS,
    extract_xgb_categories_from_json,
    prepare_xgb_lgb_frames,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ShapAnalyzer")


class ShapAnalyzer:
    """Computes native XGBoost TreeSHAP feature contributions with categorical ID support."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")

        self.model_path = model_path
        logger.info(f"Loading XGBoost model from {model_path}...")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.xgb_categories = extract_xgb_categories_from_json(model_path)

        booster = self.model.get_booster()
        self.model_feature_names: List[str] = list(booster.feature_names or FEATURE_COLUMNS)
        self.explainer = self._init_explainer()

    def _init_explainer(self) -> Any:
        logger.info("Initializing SHAP explainer...")
        try:
            return shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.warning(f"Fallback to native XGBoost TreeSHAP booster wrapper: {exc}")
            return self.model.get_booster()

    def _prepare_frame(
        self, feature_vector: Any, feature_names: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        active_names = list(feature_names or self.model_feature_names)
        if isinstance(feature_vector, pd.DataFrame):
            df = feature_vector.copy()
            active_names = list(feature_names or df.columns)
        else:
            arr = np.asarray(feature_vector, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if len(active_names) != arr.shape[1]:
                raise ValueError(
                    f"Shape of passed values is {arr.shape}, expected {len(active_names)} features"
                )
            df = pd.DataFrame(arr, columns=active_names)

        xgb_df, _ = prepare_xgb_lgb_frames(
            df,
            active_names,
            xgb_model=self.model,
            xgb_categories=self.xgb_categories,
        )
        return xgb_df, active_names

    def _compute_native_contribs(self, xgb_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        dmat = xgb.DMatrix(xgb_df, enable_categorical=True)
        contribs = np.asarray(
            self.model.get_booster().predict(dmat, pred_contribs=True), dtype=float
        )
        return contribs[:, :-1], contribs[:, -1]

    def analyze_prediction(
        self, feature_vector: Any, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculates local TreeSHAP contributions for a feature row or 1-row DataFrame."""
        xgb_df, active_names = self._prepare_frame(feature_vector, feature_names)
        shap_vals, base_vals = self._compute_native_contribs(xgb_df)
        row_shap = shap_vals[0]
        base_val = float(base_vals[0])

        contributions = {
            str(name): float(row_shap[idx]) for idx, name in enumerate(active_names)
        }
        return {
            "base_value": base_val,
            "contributions": contributions,
            "prediction": float(base_val + np.sum(row_shap)),
        }

    def generate_summary_plot(
        self,
        historical_feature_matrix: pd.DataFrame,
        output_path: str = "logs/shap_summary.png",
    ) -> None:
        """Computes global TreeSHAP feature importances and saves a Beeswarm plot."""
        logger.info(f"Computing global SHAP values for {len(historical_feature_matrix)} logs...")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        feature_names = [c for c in self.model_feature_names if c in historical_feature_matrix.columns]
        if not feature_names:
            feature_names = list(historical_feature_matrix.columns)

        xgb_df, _ = self._prepare_frame(historical_feature_matrix, feature_names)
        shap_vals, _ = self._compute_native_contribs(xgb_df)

        # Convert categorical columns to numeric codes for matplotlib colormap rendering
        plot_df = xgb_df.copy()
        for col in plot_df.columns:
            if isinstance(plot_df[col].dtype, pd.CategoricalDtype):
                plot_df[col] = plot_df[col].cat.codes.astype(float)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, plot_df, show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved SHAP Beeswarm summary plot to: {output_path}")


if __name__ == "__main__":
    points_model_path = (
        "models/wf_fold2/global_xpts_xgb.json"
        if os.path.exists("models/wf_fold2/global_xpts_xgb.json")
        else "models/global_xpts_xgb.json"
    )
    mock_row = [
        30.5, 20.0, 0.25, 2.0, 0.0, 112.5, 100.5, 107.0, 0.28,
        3.5, 55.0, 42.0, 4.5, 4.2, 1.0, 2.0, 4.0, 0.5,
        0.58, 12.0, 25.0, 1.0, 5.0, 1.1,
        201939.0, 1610612744.0, 1610612747.0, 0.0,
    ]
    analyzer = ShapAnalyzer(points_model_path)
    analysis = analyzer.analyze_prediction(np.array(mock_row), FEATURE_COLUMNS)
    print(f"Base={analysis['base_value']:.4f} Pred={analysis['prediction']:.4f}")
