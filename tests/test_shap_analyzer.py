import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shap_analyzer import ShapAnalyzer
from src.train_global_model import FEATURE_COLUMNS


def _resolve_points_model_path() -> str:
    for candidate in (
        "models/wf_fold2/global_xpts_xgb.json",
        "models/global_xpts_xgb.json",
        "models/global_xpts.json",
    ):
        if os.path.exists(candidate):
            return candidate
    return "models/wf_fold2/global_xpts_xgb.json"


def test_shap_analyzer_init():
    """Verify that ShapAnalyzer loads the Points model and explainer correctly."""
    analyzer = ShapAnalyzer(_resolve_points_model_path())
    assert analyzer.model is not None
    assert analyzer.explainer is not None


def test_shap_analyzer_prediction():
    """Test local feature contribution calculation output dict keys and values."""
    analyzer = ShapAnalyzer(_resolve_points_model_path())

    mock_vector = np.array(
        [
            30.5, 20.0, 0.25, 2.0, 0.0, 112.5, 100.5, 107.0, 0.28,
            3.5, 55.0, 42.0, 4.5, 4.2, 1.0, 2.0, 4.0, 0.5,
            0.58, 12.0, 25.0, 1.0, 5.0, 1.1,
            201939.0, 1610612744.0, 1610612747.0, 0.0,
        ]
    )

    analysis = analyzer.analyze_prediction(mock_vector, FEATURE_COLUMNS)

    assert "base_value" in analysis
    assert "contributions" in analysis
    assert "prediction" in analysis
    assert len(analysis["contributions"]) == len(FEATURE_COLUMNS)
    assert "vacated_usage" in analysis["contributions"]
    assert "roll_touches" in analysis["contributions"]
    assert (
        abs(
            analysis["prediction"]
            - (analysis["base_value"] + sum(analysis["contributions"].values()))
        )
        < 1e-4
    )


def test_shap_analyzer_summary_plot(tmp_path):
    """Test global feature importance Beeswarm summary plot generation."""
    analyzer = ShapAnalyzer(_resolve_points_model_path())

    # Create dummy matrix
    data = {
        col: (
            ["201939", "2544", "1628983"]
            if col in ["player_id", "team_id", "opponent_team_id"]
            else [1.0, 2.0, 3.0]
        )
        for col in FEATURE_COLUMNS
    }
    df = pd.DataFrame(data)

    output_png = os.path.join(tmp_path, "shap_summary.png")
    analyzer.generate_summary_plot(df, output_png)

    assert os.path.exists(output_png)
