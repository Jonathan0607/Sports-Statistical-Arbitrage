import pytest
import pandas as pd
import numpy as np
from src.models import (
    calculate_positional_priors,
    apply_bayesian_shrinkage,
    get_positional_prior,
    get_default_priors
)
from scrapers.hfp_poller import UnifiedRestPoller

def test_calculate_positional_priors():
    # Construct mock historical box scores DataFrame
    box_scores = pd.DataFrame({
        'Minutes': [30.0, 36.0, 10.0, 20.0, 32.0],  # One row below 15 mins (will be filtered)
        'Points': [15.0, 20.0, 2.0, 8.0, 18.0],
        'Rebounds': [5.0, 8.0, 1.0, 4.0, 6.0],
        'Assists': [6.0, 5.0, 0.0, 2.0, 4.0],
        'Turnovers': [2.0, 3.0, 1.0, 1.0, 2.0],
        'Position': ['PG', 'C', 'SG', 'PG', 'C']
    })

    priors = calculate_positional_priors(box_scores)
    assert 'points' in priors
    assert 'rebounds' in priors
    assert 'assists' in priors
    assert 'turnovers' in priors

    # LeBron PG Per-36 Points: (15 / 30) * 36 = 18.0. PG 2: (8 / 20) * 36 = 14.4. Average = 16.2
    assert abs(priors['points']['PG'] - 16.2) < 1e-5
    # C 1 Per-36 Points: (20 / 36) * 36 = 20.0. C 2: (18 / 32) * 36 = 20.25. Average = 20.125
    assert abs(priors['points']['C'] - 20.125) < 1e-5

def test_apply_bayesian_shrinkage():
    # player with 0 mins: w = 0 -> posterior is exactly prior
    post_0 = apply_bayesian_shrinkage(player_stat_per_36=25.0, player_total_minutes=0.0, positional_prior=15.0, K=500)
    assert post_0 == 15.0

    # player with 100 mins (< 300 mins) and K=500: w = 100 / 600 = 1/6 -> posterior is shrunk
    post_100 = apply_bayesian_shrinkage(player_stat_per_36=25.0, player_total_minutes=100.0, positional_prior=15.0, K=500)
    expected_100 = (1/6) * 25.0 + (5/6) * 15.0
    assert abs(post_100 - expected_100) < 1e-5

    # player with >= 300 mins -> no shrinkage, returns player_stat_per_36 directly
    post_300 = apply_bayesian_shrinkage(player_stat_per_36=25.0, player_total_minutes=300.0, positional_prior=15.0, K=500)
    assert post_300 == 25.0

    post_500 = apply_bayesian_shrinkage(player_stat_per_36=25.0, player_total_minutes=500.0, positional_prior=15.0, K=500)
    assert post_500 == 25.0

    post_large = apply_bayesian_shrinkage(player_stat_per_36=25.0, player_total_minutes=4500.0, positional_prior=15.0, K=500)
    assert post_large == 25.0

def test_get_positional_prior():
    # Test fallbacks
    prior_pts = get_positional_prior('Points', 'PG')
    assert prior_pts == get_default_priors()['points']['PG']

    prior_rebs = get_positional_prior('rebounds', 'C')
    assert prior_rebs == get_default_priors()['rebounds']['C']

    # Test unknown position fallback to 'SG'
    prior_unknown = get_positional_prior('Points', 'XYZ')
    assert prior_unknown == get_default_priors()['points']['SG']

def test_poller_integration_no_regression():
    poller = UnifiedRestPoller()
    history_df = pd.DataFrame({
        'minutes_played': [30.0, 32.0, 28.0, 35.0, 33.0, 31.0],
        'points': [20, 25, 18, 28, 22, 21],
        'rebounds': [5, 6, 4, 7, 5, 5],
        'assists': [5, 4, 6, 5, 4, 5],
        'turnovers': [2, 3, 1, 4, 2, 2]
    })
    
    # Test fit and prediction with positional priors
    pred_no_id = poller.project_player_baseline(history_df, 'points')
    pred_with_id = poller.project_player_baseline(history_df, 'points', player_id='lebronjames')
    
    assert isinstance(pred_no_id, float)
    assert pred_no_id > 0.0
    assert isinstance(pred_with_id, float)
    assert pred_with_id > 0.0
