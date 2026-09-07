import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import ContextualFeatureEngine
from src.models import (
    load_all_positional_priors,
    get_positional_prior,
    apply_bayesian_shrinkage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HistoricalReplayCore")

class HistoricalReplayCore:
    def __init__(self, db_uri: str = None):
        self.db_uri = db_uri or os.getenv(
            "POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db"
        )
        if self.db_uri.startswith("postgres://"):
            self.db_uri = self.db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

        self.engine = sqlalchemy.create_engine(self.db_uri)
        self.feature_engine = ContextualFeatureEngine(db_uri=self.db_uri)

        self.market_target_map = {
            "Points": "pts",
            "Rebounds": "reb",
            "Assists": "ast",
            "3-Pointers Made": "fg3m",
            "Steals": "stl",
            "Blocked Shots": "blk",
            "Turnovers": "tov",
        }

        self.stat_to_market = {
            "points": "Points",
            "rebounds": "Rebounds",
            "assists": "Assists",
            "three_pointers_made": "3-Pointers Made",
            "steals": "Steals",
            "blocks": "Blocked Shots",
            "turnovers": "Turnovers",
        }

        self.loaded_models = {}
        self.loaded_ensembles = {}
        self._load_ensemble_models()
        self._load_priors()

    def _load_ensemble_models(self):
        loaded_count = 0
        for market, target_key in self.market_target_map.items():
            xgb_path = f"models/global_x{target_key}_xgb.json"
            lgb_path = f"models/global_x{target_key}_lgb.txt"
            meta_path = f"models/global_x{target_key}_meta.json"

            ensemble_entry = {"xgb": None, "lgb": None, "meta": None}

            if os.path.exists(xgb_path):
                try:
                    bst = xgb.XGBRegressor()
                    bst.load_model(xgb_path)
                    ensemble_entry["xgb"] = bst
                    self.loaded_models[market] = bst
                except Exception as e:
                    logger.error(f"Failed to load XGBoost model for {market}: {e}")

            if os.path.exists(lgb_path):
                try:
                    lgb_booster = lgb.Booster(model_file=lgb_path)
                    ensemble_entry["lgb"] = lgb_booster
                except Exception as e:
                    logger.error(f"Failed to load LightGBM model for {market}: {e}")

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta_data = json.load(f)
                    ensemble_entry["meta"] = meta_data
                except Exception as e:
                    logger.error(f"Failed to load Meta-Learner for {market}: {e}")

            if ensemble_entry["xgb"] is not None:
                self.loaded_ensembles[market] = ensemble_entry
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} Meta-Ensemble model clusters.")

    def _load_priors(self):
        try:
            self.cached_priors = load_all_positional_priors(self.db_uri)
            logger.info("Loaded positional priors successfully.")
        except Exception as e:
            logger.error(f"Failed to load positional priors: {e}")
            self.cached_priors = {}

    def evaluate_prediction_accuracy(self, player_id: str, player_name: str, stat_category: str, game_time: str, actual_outcome_value: float) -> dict:
        stat_col = stat_category.lower().strip()
        if stat_col in self.stat_to_market:
            market_key = self.stat_to_market[stat_col]
        else:
            market_key = stat_category.strip()
            matching = [k for k, v in self.stat_to_market.items() if v.lower() == market_key.lower()]
            if matching:
                stat_col = matching[0]
            else:
                return None

        if market_key not in self.loaded_ensembles:
            return None

        history_query = """
            SELECT g.game_id, gm.start_time, gm.home_team_id, gm.away_team_id,
                   g.minutes_played, g.points, g.rebounds, g.assists, g.turnovers,
                   g.steals, g.blocks, g.three_pointers_made, g.usage_rate
            FROM player_game_logs g
            JOIN games gm ON g.game_id = gm.game_id
            WHERE g.player_id = :player_id
              AND gm.start_time < :game_time
            ORDER BY gm.start_time ASC;
        """
        try:
            with self.engine.connect() as conn:
                history_df = pd.read_sql(sqlalchemy.text(history_query), conn, params={"player_id": player_id, "game_time": game_time})
        except Exception as e:
            return None

        if len(history_df) < 3:
            return None

        try:
            ewma_baseline = float(history_df[stat_col].ewm(alpha=0.3, adjust=False).mean().iloc[-1])
            history_df["usage_rate"] = (
                (history_df["points"] + 1.5 * history_df["assists"] + history_df["turnovers"])
                / (history_df["minutes_played"] * 2.0 + 1e-5)
            ).clip(0.05, 0.45)
            roll_min = float(history_df["minutes_played"].ewm(alpha=0.3, adjust=False).mean().iloc[-1])
            roll_usg = float(history_df["usage_rate"].ewm(alpha=0.3, adjust=False).mean().iloc[-1])
            roll_stat = float(history_df[stat_col].ewm(alpha=0.3, adjust=False).mean().iloc[-1])

            position = self.feature_engine.player_positions.get(player_id, "SG")
            prior_per_36 = get_positional_prior(stat_col, position, cached_priors=self.cached_priors)

            total_minutes = float(history_df["minutes_played"].sum())
            roll_stat_per_36 = (roll_stat / (roll_min + 1e-5)) * 36.0
            pred_bayesian_per_36 = apply_bayesian_shrinkage(roll_stat_per_36, total_minutes, prior_per_36, K=500.0)
            pred_bayesian_stat = (pred_bayesian_per_36 / 36.0) * roll_min

            active_feats = self.feature_engine.build_active_features(player_id, target_date=game_time)
            vacated_usage_val = active_feats.get("vacated_usage", active_feats.get("teammate_sidelined", 0.0))
            roll_pot_ast = active_feats.get("roll_potential_assists", 0.0)
            roll_touches = active_feats.get("roll_touches", 0.0)
            roll_def_dist = active_feats.get("roll_defender_distance", 4.5)

            pred_features = np.array([[
                roll_min, pred_bayesian_stat, roll_usg, active_feats["days_rest"], active_feats["is_back_to_back"],
                active_feats["opp_def_rating"], active_feats["opp_pace_factor"], active_feats["opp_pos_pts_allowed"],
                vacated_usage_val, roll_pot_ast, roll_touches, roll_def_dist
            ]], dtype=np.float32)

            ensemble = self.loaded_ensembles.get(market_key, {})
            model_xgb = ensemble.get("xgb")
            model_lgb = ensemble.get("lgb")
            meta_data = ensemble.get("meta")

            xgb_residual = float(model_xgb.predict(pred_features)[0]) if model_xgb else 0.0
            lgb_residual = float(model_lgb.predict(pred_features)[0]) if model_lgb else xgb_residual

            if meta_data and "weights" in meta_data and "intercept" in meta_data:
                w = meta_data["weights"]
                intercept = meta_data["intercept"]
                meta_residual = intercept + (w[0] * xgb_residual) + (w[1] * lgb_residual)
            else:
                meta_residual = (xgb_residual + lgb_residual) / 2.0

            meta_prediction = max(0.001, float(ewma_baseline + meta_residual))
            
            # --- HYBRID EXTENSION: Negative Binomial Simulation for CRPS ---
            alpha = meta_data.get('alpha', 0.1) if meta_data else 0.1
            mu = meta_prediction
            var_i = mu + alpha * (mu ** 2)
            n_nb = 1.0 / alpha
            p_nb = mu / var_i
            
            # Generate exactly 10,000 array instances just like simulator
            sims = np.random.negative_binomial(n=n_nb, p=p_nb, size=10000)
            sims = np.sort(sims)
            
            n_iters = len(sims)
            mae = np.mean(np.abs(sims - actual_outcome_value))
            diffs = sims[1:] - sims[:-1]
            weights = np.arange(1, n_iters) * np.arange(n_iters - 1, 0, -1)
            expected_diff = np.sum(diffs * weights) / (n_iters * n_iters)
            crps_val = float(mae - expected_diff)

            return {
                "player_id": player_id,
                "player_name": player_name,
                "stat_category": market_key,
                "game_time": game_time,
                "actual_value": float(actual_outcome_value),
                "meta_prediction": float(meta_prediction),
                "xgboost_prediction": max(0.001, float(ewma_baseline + xgb_residual)),
                "ewma_baseline": float(ewma_baseline),
                "crps": crps_val
            }
        except Exception as e:
            return None

    def run_backtest(self, start_date: str, end_date: str, stat_categories: list = None) -> list:
        logger.info(f"Starting Hybrid Architecture Replay from {start_date} to {end_date}...")
        if stat_categories is None:
            stat_categories = ["points", "rebounds", "assists"]

        active_stats = []
        for cat in stat_categories:
            cat_lower = cat.lower().strip()
            if cat_lower in self.stat_to_market:
                active_stats.append(cat_lower)
            else:
                matching = [k for k, v in self.stat_to_market.items() if v.lower() == cat_lower]
                if matching:
                    active_stats.append(matching[0])

        query_logs = """
            SELECT g.game_id, g.player_id, p.full_name AS player_name, gm.start_time,
                   g.points, g.rebounds, g.assists, g.steals, g.blocks, g.turnovers, g.three_pointers_made
            FROM player_game_logs g
            JOIN games gm ON g.game_id = gm.game_id
            JOIN players p ON g.player_id = p.master_player_id
            WHERE gm.start_time >= :start_date AND gm.start_time <= :end_date
            ORDER BY gm.start_time ASC;
        """
        with self.engine.connect() as conn:
            logs_df = pd.read_sql(sqlalchemy.text(query_logs), conn, params={"start_date": start_date, "end_date": end_date})

        total_logs = len(logs_df)
        logger.info(f"Loaded {total_logs} box scores for evaluation.")
        if total_logs == 0:
            return []

        results = []
        for idx, row in logs_df.iterrows():
            player_id = row["player_id"]
            player_name = row["player_name"]
            game_time = str(row["start_time"])

            for stat_col in active_stats:
                if stat_col in row and row[stat_col] is not None:
                    actual_value = float(row[stat_col])
                    res = self.evaluate_prediction_accuracy(player_id, player_name, stat_col, game_time, actual_value)
                    if res is not None:
                        results.append(res)

            if (idx + 1) % 500 == 0:
                logger.info(f"Processed {idx + 1}/{total_logs} box scores ({len(results)} evaluations).")

        res_df = pd.DataFrame(results)

        y_true = res_df["actual_value"].values
        y_meta = res_df["meta_prediction"].values
        y_ewma = res_df["ewma_baseline"].values
        y_crps = res_df["crps"].values

        overall_meta_mae = float(np.mean(np.abs(y_true - y_meta)))
        overall_ewma_mae = float(np.mean(np.abs(y_true - y_ewma)))
        overall_crps = float(np.mean(y_crps))

        print("\n" + "=" * 90)
        print("📊  HYBRID ARCHITECTURE: META-ENSEMBLE MAE + NEGATIVE BINOMIAL CRPS  📊")
        print("=" * 90)
        print(f"Time Range:           {start_date} to {end_date}")
        print(f"Total Box Scores:     {total_logs}")
        print(f"Total Predictions:    {len(results)}")
        print("-" * 90)
        print(f"{'Category':<16} | {'Count':<6} | {'Meta MAE':<9} | {'EWMA MAE':<9} | {'Simulated CRPS':<15}")
        print("-" * 90)

        for cat, group in res_df.groupby("stat_category"):
            g_true = group["actual_value"].values
            g_meta = group["meta_prediction"].values
            g_ewma = group["ewma_baseline"].values
            g_crps = group["crps"].values

            cat_meta_mae = float(np.mean(np.abs(g_true - g_meta)))
            cat_ewma_mae = float(np.mean(np.abs(g_true - g_ewma)))
            cat_crps = float(np.mean(g_crps))

            print(
                f"{cat:<16} | {len(group):<6} | {cat_meta_mae:<9.3f} | {cat_ewma_mae:<9.3f} | {cat_crps:<15.3f}"
            )

        print("-" * 90)
        print(
            f"{'OVERALL':<16} | {len(results):<6} | {overall_meta_mae:<9.3f} | {overall_ewma_mae:<9.3f} | {overall_crps:<15.3f}"
        )
        print("=" * 90)
        
        return results

if __name__ == "__main__":
    replay = HistoricalReplayCore()
    replay.run_backtest(
        start_date="2025-11-15 00:00:00",
        end_date="2025-11-17 23:59:59",
        stat_categories=["points", "rebounds", "assists"],
    )
