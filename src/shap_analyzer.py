import os
import sys
import logging
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ShapAnalyzer")


class ShapAnalyzer:
    def __init__(self, model_path: str):
        """
        Initializes the ShapAnalyzer by loading an XGBoost model
        and preparing the SHAP TreeExplainer.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")
        
        logger.info(f"Loading XGBoost model from {model_path}...")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        logger.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def analyze_prediction(self, feature_vector: np.ndarray, feature_names: list) -> dict:
        """
        Calculates the SHAP values (local contribution) for a specific feature vector.
        """
        # Reshape input to 2D row array if 1D is passed
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Compute SHAP values
        shap_values_obj = self.explainer(feature_vector)
        
        # shap_values_obj.values contains the contributions. For a single row, index it.
        row_shap = shap_values_obj.values[0]
        base_val = float(shap_values_obj.base_values[0])

        contributions = {}
        for idx, name in enumerate(feature_names):
            contributions[name] = float(row_shap[idx])

        return {
            "base_value": base_val,
            "contributions": contributions,
            "prediction": float(base_val + np.sum(row_shap))
        }

    def generate_summary_plot(self, historical_feature_matrix: pd.DataFrame, output_path: str = "logs/shap_summary.png"):
        """
        Computes global feature importances across a historical matrix and saves a Beeswarm plot.
        """
        logger.info(f"Computing global SHAP values for {len(historical_feature_matrix)} logs...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Compute SHAP values across matrix
        shap_values = self.explainer(historical_feature_matrix)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, historical_feature_matrix, show=False)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved SHAP Beeswarm summary plot to: {output_path}")


if __name__ == "__main__":
    # Define Points model path
    points_model_path = "models/global_xpts.json"
    
    # Mock features vector representing:
    # [minutes_played_mean, pred_bayesian_stat, usage_rate_mean, days_rest, is_back_to_back,
    #  opp_def_rating, opp_pace_factor, opp_pos_pts_allowed, vacated_usage,
    #  roll_potential_assists, roll_touches, roll_defender_distance]
    mock_vector = np.array([30.5, 18.2, 0.225, 2.0, 0.0, 112.5, 100.5, 107.0, 0.28, 3.5, 55.0, 4.2])
    
    feature_columns = [
        "minutes_played_mean",
        "pred_bayesian_stat",
        "usage_rate_mean",
        "days_rest",
        "is_back_to_back",
        "opp_def_rating",
        "opp_pace_factor",
        "opp_pos_pts_allowed",
        "vacated_usage",
        "roll_potential_assists",
        "roll_touches",
        "roll_defender_distance",
    ]
    
    try:
        analyzer = ShapAnalyzer(points_model_path)
        analysis = analyzer.analyze_prediction(mock_vector, feature_columns)
        
        print("\n" + "=" * 60)
        print("🔍 XGBOOST POINTS MODEL - LOCAL SHAP EXPLANATION 🔍")
        print("=" * 60)
        print(f"Model Base (Expected) Value:  {analysis['base_value']:.4f}")
        print(f"Model Final Prediction:       {analysis['prediction']:.4f}")
        print("-" * 60)
        print(f"{'Feature Name':<25} | {'SHAP Contribution (pts)':<25}")
        print("-" * 60)
        for feat, val in analysis["contributions"].items():
            print(f"{feat:<25} | {val:+.4f}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"Failed to run SHAP analyzer demo: {e}", file=sys.stderr)
