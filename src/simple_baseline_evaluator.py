# src/simple_baseline_evaluator.py
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SimpleBaselineEvaluator")


class SimpleBaselineEvaluator:
    """
    Pure Statistical Time-Series Baseline Evaluator.
    Completely bypasses tree-based machine learning (XGBoost, LightGBM)
    and evaluates:
    1. 5-Game Simple Moving Average (SMA) Baseline
    2. Exponentially Weighted Moving Average (EWMA) with alpha = 0.3
    3. Multiplicative Context-Adjusted EWMA:
       mu_hat = EWMA * (Opponent Pace / League Pace) * (Opponent Def Rating / League Def Rating)
    """

    def __init__(
        self,
        db_uri: str = None,
        alpha: float = 0.3,
        league_pace: float = 99.0,
        league_def_rating: float = 115.0,
    ):
        load_dotenv()
        self.db_uri = db_uri or os.getenv(
            "POSTGRES_URI", "postgresql+psycopg2://postgres:password@localhost:5432/sports_db"
        )
        if self.db_uri.startswith("postgres://"):
            self.db_uri = self.db_uri.replace("postgres://", "postgresql+psycopg2://", 1)

        self.engine = sqlalchemy.create_engine(self.db_uri)
        self.alpha = alpha
        self.league_pace = league_pace
        self.league_def_rating = league_def_rating

        self.target_stats = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
        }

        self.player_history = {}
        self.matchup_map = {}
        self.players_meta = {}

    def load_data(self):
        """
        Pre-loads all player game logs, player metadata, and team matchup metrics
        into memory for high-performance vectorized time-series evaluation.
        """
        logger.info("Pre-loading historical game logs and matchup metrics from database...")
        with self.engine.connect() as conn:
            logs_df = pd.read_sql(
                """
                SELECT g.game_id, g.player_id, p.full_name AS player_name, p.current_team_id,
                       gm.home_team_id, gm.away_team_id, gm.start_time,
                       g.points, g.rebounds, g.assists
                FROM player_game_logs g
                JOIN games gm ON g.game_id = gm.game_id
                JOIN players p ON g.player_id = p.master_player_id
                ORDER BY gm.start_time ASC;
                """,
                conn,
            )

            metrics_df = pd.read_sql(
                """
                SELECT team_id, game_id, rolling_def_rating, rolling_pace_factor
                FROM team_matchup_metrics;
                """,
                conn,
            )

        logs_df["start_time"] = pd.to_datetime(logs_df["start_time"], utc=True)
        self.logs_df = logs_df

        # Build fast O(1) matchup dictionary: (team_id, game_id) -> (def_rating, pace_factor)
        self.matchup_map = {}
        for r in metrics_df.itertuples():
            if pd.notna(r.rolling_def_rating) and pd.notna(r.rolling_pace_factor):
                self.matchup_map[(int(r.team_id), int(r.game_id))] = (
                    float(r.rolling_def_rating),
                    float(r.rolling_pace_factor),
                )

        # Build player historical game logs index
        self.player_history = {}
        for pid, group in logs_df.groupby("player_id"):
            self.player_history[pid] = group.sort_values("start_time").reset_index(drop=True)

        logger.info(
            f"Loaded {len(logs_df)} game logs across {len(self.player_history)} players and {len(self.matchup_map)} matchup metrics."
        )

    def evaluate_point_in_time(
        self,
        player_id: str,
        game_id: int,
        game_time: pd.Timestamp,
        stat_col: str,
        actual_value: float,
        player_team_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> dict:
        """
        Calculates point-in-time predictions for a single player box score:
        1. 5-Game SMA
        2. Pure EWMA (alpha = 0.3)
        3. EWMA + Multiplicative Context (Pace & Def Rating)
        """
        p_hist = self.player_history.get(player_id)
        if p_hist is None:
            return None

        # Strictly point-in-time: games prior to game_time
        prior_df = p_hist[p_hist["start_time"] < game_time]
        if len(prior_df) < 5:
            # Need at least 5 games for a fair 5-game SMA baseline
            return None

        # 1. 5-Game Simple Moving Average (SMA)
        naive_sma = float(prior_df.tail(5)[stat_col].mean())

        # 2. Exponentially Weighted Moving Average (EWMA)
        ewma_raw = float(prior_df[stat_col].ewm(alpha=self.alpha, adjust=True).mean().iloc[-1])

        # 3. Multiplicative Context Adjustment
        # Determine opponent team id
        curr_team = int(player_team_id) if pd.notna(player_team_id) else int(home_team_id)
        opp_team = int(away_team_id) if curr_team == int(home_team_id) else int(home_team_id)

        # Retrieve opponent defensive rating & pace factor
        opp_def, opp_pace = self.matchup_map.get((opp_team, int(game_id)), (self.league_def_rating, self.league_pace))
        if pd.isna(opp_def) or opp_def <= 0:
            opp_def = self.league_def_rating
        if pd.isna(opp_pace) or opp_pace <= 0:
            opp_pace = self.league_pace

        pace_scaler = opp_pace / self.league_pace
        def_scaler = opp_def / self.league_def_rating

        ewma_context = max(0.001, ewma_raw * pace_scaler * def_scaler)

        return {
            "player_id": player_id,
            "game_id": game_id,
            "game_time": game_time,
            "stat_category": stat_col,
            "actual_value": float(actual_value),
            "naive_sma": float(naive_sma),
            "ewma_raw": float(ewma_raw),
            "ewma_context": float(ewma_context),
            "opp_pace": float(opp_pace),
            "opp_def": float(opp_def),
            "pace_scaler": float(pace_scaler),
            "def_scaler": float(def_scaler),
        }

    def run_evaluation(
        self,
        start_date: str = "2026-03-01 00:00:00",
        end_date: str = "2026-04-10 23:59:59",
        stat_categories: list = None,
    ) -> pd.DataFrame:
        """
        Executes chronological evaluation over the evaluation window and prints
        a comparison report.
        """
        if not self.player_history:
            self.load_data()

        if stat_categories is None:
            stat_categories = ["points", "rebounds", "assists"]

        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        eval_window = self.logs_df[
            (self.logs_df["start_time"] >= start_dt) & (self.logs_df["start_time"] <= end_dt)
        ].copy()

        logger.info(
            f"Evaluating pure statistical baselines from {start_date} to {end_date} ({len(eval_window)} box scores)..."
        )

        results = []
        for _, row in eval_window.iterrows():
            pid = row["player_id"]
            gid = int(row["game_id"])
            gtime = row["start_time"]
            curr_team_id = row["current_team_id"]
            home_team_id = int(row["home_team_id"])
            away_team_id = int(row["away_team_id"])

            for stat_col in stat_categories:
                if stat_col in row and pd.notna(row[stat_col]):
                    actual_val = float(row[stat_col])
                    res = self.evaluate_point_in_time(
                        player_id=pid,
                        game_id=gid,
                        game_time=gtime,
                        stat_col=stat_col,
                        actual_value=actual_val,
                        player_team_id=curr_team_id,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                    )
                    if res is not None:
                        res["player_name"] = row["player_name"]
                        results.append(res)

        if not results:
            logger.warning("No evaluations generated.")
            return pd.DataFrame()

        res_df = pd.DataFrame(results)

        # ------------------------------------------------------------------
        # Print Terminal Comparison Table
        # ------------------------------------------------------------------
        print("\n" + "=" * 96)
        print("📊  PURE STATISTICAL BASELINE EVALUATION: EWMA & CONTEXT VS 5-GAME SMA  📊")
        print("=" * 96)
        print(f"Time Range:            {start_date} to {end_date}")
        print(f"Total Box Scores:      {len(eval_window)}")
        print(f"Total Stat Forecasts:  {len(res_df)}")
        print(f"EWMA Smoothing Alpha:  {self.alpha}")
        print(f"League Neutral Scalers: Pace={self.league_pace}, DefRating={self.league_def_rating}")
        print("-" * 96)
        print(
            f"{'Category':<12} | {'Count':<6} | {'SMA-5 MAE':<9} | {'EWMA MAE':<9} | {'EWMA+Ctx MAE':<12} | {'EWMA vs SMA Δ':<14} | {'Winner':<10}"
        )
        print("-" * 96)

        for cat, group in res_df.groupby("stat_category"):
            y_true = group["actual_value"].values
            y_sma = group["naive_sma"].values
            y_ewma = group["ewma_raw"].values
            y_ctx = group["ewma_context"].values

            sma_mae = float(np.mean(np.abs(y_true - y_sma)))
            ewma_mae = float(np.mean(np.abs(y_true - y_ewma)))
            ctx_mae = float(np.mean(np.abs(y_true - y_ctx)))

            delta_ewma_pct = ((ewma_mae - sma_mae) / (sma_mae + 1e-9)) * 100.0 if sma_mae > 0 else 0.0
            winner = "EWMA ✅" if ewma_mae < sma_mae else "SMA-5 ❌"

            print(
                f"{cat.capitalize():<12} | {len(group):<6} | {sma_mae:<9.3f} | {ewma_mae:<9.3f} | {ctx_mae:<12.3f} | {delta_ewma_pct:>+6.2f}% ({ewma_mae - sma_mae:>+6.3f}) | {winner:<10}"
            )

        print("-" * 96)
        y_all_true = res_df["actual_value"].values
        y_all_sma = res_df["naive_sma"].values
        y_all_ewma = res_df["ewma_raw"].values
        y_all_ctx = res_df["ewma_context"].values

        all_sma_mae = float(np.mean(np.abs(y_all_true - y_all_sma)))
        all_ewma_mae = float(np.mean(np.abs(y_all_true - y_all_ewma)))
        all_ctx_mae = float(np.mean(np.abs(y_all_true - y_all_ctx)))

        all_sma_rmse = float(np.sqrt(np.mean((y_all_true - y_all_sma) ** 2)))
        all_ewma_rmse = float(np.sqrt(np.mean((y_all_true - y_all_ewma) ** 2)))
        all_ctx_rmse = float(np.sqrt(np.mean((y_all_true - y_all_ctx) ** 2)))

        overall_delta_pct = ((all_ewma_mae - all_sma_mae) / (all_sma_mae + 1e-9)) * 100.0 if all_sma_mae > 0 else 0.0
        overall_winner = "EWMA ✅" if all_ewma_mae < all_sma_mae else "SMA-5 ❌"

        print(
            f"{'OVERALL':<12} | {len(res_df):<6} | {all_sma_mae:<9.3f} | {all_ewma_mae:<9.3f} | {all_ctx_mae:<12.3f} | {overall_delta_pct:>+6.2f}% ({all_ewma_mae - all_sma_mae:>+6.3f}) | {overall_winner:<10}"
        )
        print("=" * 96)

        print("\n" + "-" * 96)
        print("📈  RMSE (ROOT MEAN SQUARED ERROR) COMPARISON  📈")
        print("-" * 96)
        print(f"5-Game SMA Overall RMSE:              {all_sma_rmse:.4f}")
        print(f"Pure EWMA (alpha=0.3) Overall RMSE:   {all_ewma_rmse:.4f} (Delta: {all_ewma_rmse - all_sma_rmse:+.4f})")
        print(f"EWMA + Context Adjusted Overall RMSE: {all_ctx_rmse:.4f} (Delta: {all_ctx_rmse - all_sma_rmse:+.4f})")
        print("=" * 96 + "\n")

        return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pure Statistical Baseline Models (EWMA & Context vs SMA-5)")
    parser.add_argument("--start-date", type=str, default="2026-03-01 00:00:00", help="Evaluation start date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-date", type=str, default="2026-04-10 23:59:59", help="Evaluation end date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--alpha", type=float, default=0.3, help="EWMA smoothing factor alpha (default: 0.3)")
    args = parser.parse_args()

    evaluator = SimpleBaselineEvaluator(alpha=args.alpha)
    evaluator.run_evaluation(
        start_date=args.start_date,
        end_date=args.end_date,
        stat_categories=["points", "rebounds", "assists"],
    )
