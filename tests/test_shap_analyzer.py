import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shap_analyzer import ShapAnalyzer


def test_shap_analyzer_init():
    """Verify that ShapAnalyzer loads the Points model and explainer correctly."""
    points_model_path = "models/global_xpts.json"
    analyzer = ShapAnalyzer(points_model_path)
    assert analyzer.model is not None
    assert analyzer.explainer is not None


def test_shap_analyzer_prediction():
    """Test local feature contribution calculation output dict keys and values."""
    points_model_path = "models/global_xpts.json"
    analyzer = ShapAnalyzer(points_model_path)
    
    mock_vector = np.array([30.5, 18.2, 0.225, 2.0, 0.0, 112.5, 100.5, 107.0, 1.0, 3.5, 55.0, 4.2])
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
    
    analysis = analyzer.analyze_prediction(mock_vector, feature_columns)
    
    assert "base_value" in analysis
    assert "contributions" in analysis
    assert "prediction" in analysis
    assert len(analysis["contributions"]) == 12
    assert "pred_bayesian_stat" in analysis["contributions"]
    assert "roll_touches" in analysis["contributions"]
    assert abs(analysis["prediction"] - (analysis["base_value"] + sum(analysis["contributions"].values()))) < 1e-4


def test_shap_analyzer_summary_plot(tmp_path):
    """Test global feature importance Beeswarm summary plot generation."""
    points_model_path = "models/global_xpts.json"
    analyzer = ShapAnalyzer(points_model_path)
    
    # Create dummy matrix
    data = {
        "minutes_played_mean": [30.5, 25.0, 35.0],
        "pred_bayesian_stat": [18.2, 12.5, 22.0],
        "usage_rate_mean": [0.225, 0.180, 0.280],
        "days_rest": [2.0, 1.0, 3.0],
        "is_back_to_back": [0.0, 1.0, 0.0],
        "opp_def_rating": [112.5, 115.0, 110.0],
        "opp_pace_factor": [100.5, 98.0, 102.5],
        "opp_pos_pts_allowed": [107.0, 102.0, 112.0],
        "vacated_usage": [0.25, 0.0, 0.30],
        "roll_potential_assists": [3.5, 2.0, 5.0],
        "roll_touches": [55.0, 42.0, 68.0],
        "roll_defender_distance": [4.2, 3.8, 5.1],
    }
    df = pd.DataFrame(data)
    
    output_png = os.path.join(tmp_path, "shap_summary.png")
    analyzer.generate_summary_plot(df, output_png)
    
    assert os.path.exists(output_png)
