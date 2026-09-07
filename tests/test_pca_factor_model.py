import os
import pytest
import pandas as pd
import numpy as np
from src.models import (
    generate_mock_box_scores,
    build_fixed_width_matrix,
    fit_pca_factor_model,
    extract_game_factors
)

TEST_PLOT_PATH = "logs/test_pca_variance_plot.png"

@pytest.fixture
def clean_plot():
    # Remove plot if exists before test
    if os.path.exists(TEST_PLOT_PATH):
        try:
            os.remove(TEST_PLOT_PATH)
        except OSError:
            pass

    yield

    # Cleanup after test
    if os.path.exists(TEST_PLOT_PATH):
        try:
            os.remove(TEST_PLOT_PATH)
        except OSError:
            pass

def test_generate_mock_box_scores():
    df = generate_mock_box_scores(num_games=10)
    assert not df.empty
    # 10 games * 2 teams * 10 players/positions = 200 records
    assert len(df) == 200
    assert "game_id" in df.columns
    assert "team_abbrev" in df.columns
    assert "minutes_played" in df.columns

def test_build_fixed_width_matrix():
    mock_df = generate_mock_box_scores(num_games=5)
    matrix_df = build_fixed_width_matrix(mock_df)
    
    assert not matrix_df.empty
    assert len(matrix_df) == 5
    # columns should be 64 (16 players * 4 stats: min, pts, reb, ast)
    assert matrix_df.shape[1] == 64
    
    # Check that T1_P1_MIN exists
    assert "T1_P1_MIN" in matrix_df.columns
    assert "T2_P8_AST" in matrix_df.columns

def test_fit_pca_factor_model(clean_plot):
    mock_df = generate_mock_box_scores(num_games=20)
    matrix_df = build_fixed_width_matrix(mock_df)
    
    scaler, pca, n_components = fit_pca_factor_model(matrix_df, save_plot_path=TEST_PLOT_PATH)
    
    assert os.path.exists(TEST_PLOT_PATH)
    assert n_components > 0
    assert n_components <= 64
    
    # Verify we can extract factors
    sample_game = matrix_df.iloc[0].values
    factors = extract_game_factors(sample_game, scaler, pca, n_components=3)
    assert factors.shape == (1, 3)
