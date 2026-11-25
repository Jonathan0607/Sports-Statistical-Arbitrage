# tests/test_walk_forward.py
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.walk_forward_backtester import WalkForwardBacktester


def test_walk_forward_initialization():
    backtester = WalkForwardBacktester(target_keys=["pts", "reb", "ast"], max_box_scores=500)
    assert len(backtester.folds) == 2
    assert backtester.folds[0]["train_seasons"] == ["2023-24"]
    assert backtester.folds[0]["test_seasons"] == ["2024-25"]
    assert backtester.folds[1]["train_seasons"] == ["2023-24", "2024-25"]
    assert backtester.folds[1]["test_seasons"] == ["2025-26"]


def test_walk_forward_run_mocked():
    backtester = WalkForwardBacktester(target_keys=["pts", "reb"], max_box_scores=50, retrain=True)

    mock_summary = [
        {"Category": "Points", "Count": 50, "Meta MAE": 4.50, "EWMA MAE": 5.00, "Sim CRPS": 3.10},
        {"Category": "Rebounds", "Count": 50, "Meta MAE": 1.74, "EWMA MAE": 1.90, "Sim CRPS": 1.40},
        {"Category": "OVERALL", "Count": 100, "Meta MAE": 3.12, "EWMA MAE": 3.45, "Sim CRPS": 2.25},
    ]

    with patch("src.walk_forward_backtester.train_global_ensemble") as mock_train, \
         patch("src.walk_forward_backtester.HistoricalReplayCore") as mock_replay_cls:

        mock_replay = MagicMock()
        mock_replay.run_backtest.return_value = {"results": [], "summary": mock_summary}
        mock_replay_cls.return_value = mock_replay

        results = backtester.run()

        # Check train calls
        assert mock_train.call_count == 2
        # Fold 1 call args
        f1_call = mock_train.call_args_list[0]
        assert f1_call.kwargs["train_seasons"] == ["2023-24"]
        assert f1_call.kwargs["model_dir"] == "models/wf_fold1"

        # Fold 2 call args
        f2_call = mock_train.call_args_list[1]
        assert f2_call.kwargs["train_seasons"] == ["2023-24", "2024-25"]
        assert f2_call.kwargs["model_dir"] == "models/wf_fold2"

        # Check replay calls
        assert mock_replay_cls.call_count == 2
        assert mock_replay_cls.call_args_list[0].kwargs["model_dir"] == "models/wf_fold1"
        assert mock_replay_cls.call_args_list[1].kwargs["model_dir"] == "models/wf_fold2"

        # Check results structure
        assert "Fold 1" in results
        assert "Fold 2" in results
        overall_row = [r for r in results["Fold 1"]["summary"] if r["Category"] == "OVERALL"][0]
        assert overall_row["Meta MAE"] == 3.12


def test_walk_forward_tuning_modes():
    """Verify tuning modes correctly pass config_path to train_global_ensemble."""

    mock_summary = [
        {"Category": "OVERALL", "Count": 10, "Meta MAE": 1.0, "EWMA MAE": 1.0, "Sim CRPS": 1.0}
    ]
    
    with patch("src.walk_forward_backtester.train_global_ensemble") as mock_train, \
         patch("src.walk_forward_backtester.HistoricalReplayCore") as mock_replay_cls, \
         patch("src.walk_forward_backtester.os.makedirs"):
         
        mock_replay = MagicMock()
        mock_replay.run_backtest.return_value = {"results": [], "summary": mock_summary}
        mock_replay_cls.return_value = mock_replay
        
        # Test domain_priors
        backtester = WalkForwardBacktester(target_keys=["pts"], retrain=True, tuning_mode="domain_priors")
        backtester.run()
        
        assert mock_train.call_args_list[-2].kwargs["config_path"] == "config/domain_priors_hyperparameters.json"
        
        mock_train.reset_mock()
        mock_replay_cls.reset_mock()
        
        # Test nested_optuna
        with patch("src.tune_models.run_stat_regularization_sweep") as mock_sweep:
            backtester_nested = WalkForwardBacktester(target_keys=["pts"], retrain=True, tuning_mode="nested_optuna")
            backtester_nested.run()
            
            assert mock_sweep.call_count == 2 # 2 folds
            
            # Fold 1 inner sweep
            assert mock_sweep.call_args_list[0].kwargs["train_seasons"] == ["2023-24"]
            assert "models/wf_fold1/stat_hyperparameters.json" in mock_sweep.call_args_list[0].kwargs["output_path"]
            
            # Fold 1 train call
            assert "models/wf_fold1/stat_hyperparameters.json" in mock_train.call_args_list[0].kwargs["config_path"]
