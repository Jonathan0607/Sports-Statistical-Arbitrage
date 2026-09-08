# src/walk_forward_backtester.py
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_global_model import train_global_ensemble
from src.historical_replay_core import HistoricalReplayCore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WalkForwardBacktester")


class WalkForwardBacktester:
    """
    Executes an Expanding Window Walk-Forward Analysis across multiple NBA seasons:
    Fold 1: Train on ['2023-24'], Test on ['2024-25']
    Fold 2: Train on ['2023-24', '2024-25'], Test on ['2025-26']

    Eliminates lookahead bias by:
    1. Bounding training feature compilations and player discovery strictly to train_seasons.
    2. Isolating Optuna Negative Binomial dispersion (alpha) calibration to train_seasons residuals.
    3. Persisting isolated model artifacts per fold.
    4. Evaluating out-of-sample on unseen future seasons.
    """

    def __init__(
        self,
        target_keys: list = None,
        max_box_scores: int = None,
        retrain: bool = True,
        db_uri: str = None,
        tuning_mode: str = "domain_priors",
        inner_trials: int = 15,
    ):
        self.target_keys = target_keys or ["pts", "reb", "ast"]
        self.max_box_scores = max_box_scores
        self.retrain = retrain
        self.db_uri = db_uri
        self.tuning_mode = tuning_mode
        self.inner_trials = inner_trials
        self.folds = [
            {
                "fold_id": 1,
                "name": "Fold 1",
                "train_seasons": ["2023-24"],
                "test_seasons": ["2024-25"],
                "model_dir": "models/wf_fold1",
            },
            {
                "fold_id": 2,
                "name": "Fold 2",
                "train_seasons": ["2023-24", "2024-25"],
                "test_seasons": ["2025-26"],
                "model_dir": "models/wf_fold2",
            },
        ]

    def check_models_exist(self, model_dir: str) -> bool:
        """Checks whether all required model artifacts exist for target keys in model_dir."""
        for tk in self.target_keys:
            xgb_path = os.path.join(model_dir, f"global_x{tk}_xgb.json")
            compat_xgb = os.path.join(model_dir, f"global_x{tk}.json")
            lgb_path = os.path.join(model_dir, f"global_x{tk}_lgb.txt")
            meta_path = os.path.join(model_dir, f"global_x{tk}_meta.json")

            has_xgb = os.path.exists(xgb_path) or os.path.exists(compat_xgb)
            if not (has_xgb and os.path.exists(lgb_path) and os.path.exists(meta_path)):
                return False
        return True

    def train_fold(self, fold_info: dict):
        """Trains global ensemble models and calibrates alpha for the fold's training seasons."""
        logger.info(f"\n=======================================================")
        logger.info(f"🔨 TRAINING {fold_info['name']}: Seasons {fold_info['train_seasons']}")
        logger.info(f"Target metrics: {self.target_keys} -> Target dir: {fold_info['model_dir']}")
        logger.info(f"Tuning Mode: {self.tuning_mode}")
        logger.info(f"=======================================================")

        os.makedirs(fold_info["model_dir"], exist_ok=True)
        
        config_path = "config/stat_hyperparameters.json"
        if self.tuning_mode == "nested_optuna":
            config_path = os.path.join(fold_info["model_dir"], "stat_hyperparameters.json")
            logger.info(f"Running nested Optuna hyperparameter sweep on {fold_info['train_seasons']}")
            from src.tune_models import run_stat_regularization_sweep
            run_stat_regularization_sweep(
                stats=self.target_keys,
                n_trials=self.inner_trials,
                output_path=config_path,
                db_uri=self.db_uri,
                train_seasons=fold_info["train_seasons"],
            )
        elif self.tuning_mode == "reuse_optuna":
            config_path = os.path.join(fold_info["model_dir"], "stat_hyperparameters.json")
            logger.info(f"Reusing existing Optuna hyperparameter sweep config for {fold_info['train_seasons']}")
        elif self.tuning_mode == "domain_priors":
            config_path = "config/domain_priors_hyperparameters.json"

        train_global_ensemble(
            target_keys=self.target_keys,
            config_path=config_path,
            train_seasons=fold_info["train_seasons"],
            model_dir=fold_info["model_dir"],
        )

    def evaluate_fold(self, fold_info: dict) -> dict:
        """Evaluates trained fold models on out-of-sample test seasons."""
        logger.info(f"\n=======================================================")
        logger.info(f"🧪 EVALUATING {fold_info['name']}: Test Season(s) {fold_info['test_seasons']}")
        logger.info(f"Using weights from: {fold_info['model_dir']}")
        logger.info(f"=======================================================")

        replay = HistoricalReplayCore(
            db_uri=self.db_uri,
            model_dir=fold_info["model_dir"],
        )

        stat_categories = ["points", "rebounds", "assists"]
        eval_output = replay.run_backtest(
            test_seasons=fold_info["test_seasons"],
            stat_categories=stat_categories,
            max_box_scores=self.max_box_scores,
            return_summary=True,
        )

        return eval_output["summary"]

    def run(self) -> dict:
        """Executes full walk-forward pipeline across all expanding folds."""
        fold_results = {}

        for fold in self.folds:
            model_dir = fold["model_dir"]
            need_train = self.retrain or not self.check_models_exist(model_dir)

            if need_train:
                self.train_fold(fold)
            else:
                logger.info(f"Skipping training for {fold['name']} - found existing models in {model_dir}.")

            summary = self.evaluate_fold(fold)
            fold_results[fold["name"]] = {
                "fold_info": fold,
                "summary": summary,
            }

        self.display_results_table(fold_results)
        return fold_results

    def display_results_table(self, fold_results: dict):
        """Formats and prints markdown table comparing Meta MAE, EWMA MAE, and CRPS."""
        print("\n" + "#" * 105)
        print("📈  WALK-FORWARD ANALYSIS SUMMARY: EXPANDING MULTI-SEASON REGIME EVALUATION  📈")
        print("#" * 105)

        table_header = (
            f"| {'Fold':<8} | {'Test Season':<12} | {'Category':<12} | {'Count':<7} | "
            f"{'Meta MAE':<10} | {'EWMA MAE':<10} | {'MAE Imp (%)':<12} | {'Sim CRPS':<10} |"
        )
        separator = f"|{'-'*10}|{'-'*14}|{'-'*14}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*12}|"

        print(table_header)
        print(separator)

        for fold_name, f_data in fold_results.items():
            test_seasons_str = ",".join(f_data["fold_info"]["test_seasons"])
            summary = f_data["summary"]

            by_cat = summary.get("by_category", {})
            for cat_name in ["Points", "Rebounds", "Assists"]:
                if cat_name in by_cat:
                    c_data = by_cat[cat_name]
                    print(
                        f"| {fold_name:<8} | {test_seasons_str:<12} | {cat_name:<12} | "
                        f"{c_data['count']:<7} | {c_data['meta_mae']:<10.3f} | {c_data['ewma_mae']:<10.3f} | "
                        f"{c_data['mae_improvement_pct']:>+10.2f}% | {c_data['crps']:<10.3f} |"
                    )

            overall = summary.get("overall", {})
            if overall:
                print(
                    f"| {fold_name:<8} | {test_seasons_str:<12} | {'OVERALL':<12} | "
                    f"{overall.get('count', 0):<7} | {overall.get('meta_mae', 0.0):<10.3f} | "
                    f"{overall.get('ewma_mae', 0.0):<10.3f} | {overall.get('mae_improvement_pct', 0.0):>+10.2f}% | "
                    f"{overall.get('crps', 0.0):<10.3f} |"
                )
            print(separator)

        print("#" * 105 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Season Expanding Window Walk-Forward Backtester")
    parser.add_argument(
        "--max-box-scores",
        type=int,
        default=None,
        help="Maximum box scores to evaluate per fold (default: None, evaluate full season)",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip model retraining if fold models already exist in model_dir",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["pts", "reb", "ast"],
        help="Target metric keys to train and evaluate (default: pts reb ast)",
    )
    parser.add_argument(
        "--tuning-mode",
        type=str,
        default="domain_priors",
        choices=["nested_optuna", "domain_priors", "fixed_config", "reuse_optuna"],
        help="Hyperparameter tuning mode for Walk-Forward Analysis",
    )
    parser.add_argument(
        "--inner-trials",
        type=int,
        default=15,
        help="Number of nested Optuna trials if tuning_mode is nested_optuna",
    )

    args = parser.parse_args()

    backtester = WalkForwardBacktester(
        target_keys=args.targets,
        max_box_scores=args.max_box_scores,
        retrain=not args.no_retrain,
        tuning_mode=args.tuning_mode,
        inner_trials=args.inner_trials,
    )
    backtester.run()


if __name__ == "__main__":
    main()
