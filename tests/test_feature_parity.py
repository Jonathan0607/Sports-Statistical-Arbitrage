"""Tests for unified feature parity contract, temporal leakage invariance, and evaluator equivalence."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_fast import evaluate_dataframe_vectorized, load_models
from src.feature_contract import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    build_online_feature_row,
    compute_causal_player_features,
    compute_target_rolling_features,
    predict_meta_ensemble,
)
from src.features import ContextualFeatureEngine
from src.historical_replay_core import evaluate_dataframe_sequential
from src.models import get_default_priors
from src.train_global_model import (
    FEATURE_COLUMNS as TRAIN_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION as TRAIN_SCHEMA_VERSION,
    train_global_ensemble,
)


def _make_synthetic_player_history(
    player_id: str = "1629029",
    team_id: int = 7,
    opp_ids: tuple = (2, 5, 11, 14, 20),
    n_games_s1: int = 12,
    n_games_s2: int = 15,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 1000 * int(abs(hash(player_id)) % 100 + 1)

    for season, n_games, start_str in [
        ("2024-25", n_games_s1, "2024-11-01 19:00:00"),
        ("2025-26", n_games_s2, "2025-10-25 19:00:00"),
    ]:
        dates = pd.date_range(start_str, periods=n_games, freq="2D", tz="UTC")
        for idx, dt in enumerate(dates):
            game_id += 1
            opp = int(opp_ids[idx % len(opp_ids)])
            is_home = idx % 2 == 0
            mins = float(np.clip(rng.normal(31.0, 4.0), 10.0, 42.0))
            fga = int(np.clip(rng.poisson(16), 3, 30))
            fta = int(np.clip(rng.poisson(5), 0, 15))
            pts = int(np.clip(rng.poisson(22), 2, 48))
            reb = int(np.clip(rng.poisson(6), 0, 18))
            ast = int(np.clip(rng.poisson(5), 0, 16))
            tov = int(np.clip(rng.poisson(2.5), 0, 8))
            rows.append(
                {
                    "game_id": game_id,
                    "start_time": dt,
                    "season": season,
                    "player_id": str(player_id),
                    "team_id": team_id,
                    "home_team_id": team_id if is_home else opp,
                    "away_team_id": opp if is_home else team_id,
                    "opponent_team_id": opp,
                    "is_home_game": 1.0 if is_home else 0.0,
                    "opp_def_rating": float(110.0 + (opp % 7)),
                    "opp_pace_factor": float(97.0 + (opp % 5)),
                    "opp_pos_pts_allowed": float(20.0 + (opp % 4)),
                    "pace_differential": float((opp % 5) - 2.0),
                    "vacated_usage": 0.25 if idx % 6 == 0 else 0.0,
                    "minutes_played": mins,
                    "points": pts,
                    "rebounds": reb,
                    "assists": ast,
                    "turnovers": tov,
                    "steals": int(rng.poisson(1.1)),
                    "blocks": int(rng.poisson(0.6)),
                    "three_pointers_made": int(rng.poisson(2.3)),
                    "field_goals_attempted": fga,
                    "free_throws_attempted": fta,
                    "plus_minus": float(rng.normal(2.0, 9.0)),
                    "potential_assists": float(ast * 1.8),
                    "touches": float(mins * 2.1),
                    "passes_made": float(mins * 1.4),
                    "time_of_possession_sec": float(mins * 4.5),
                    "avg_defender_distance_ft": float(np.clip(rng.normal(4.4, 0.5), 2.5, 7.0)),
                    "shorthanded_points": float(rng.poisson(2.0)),
                }
            )
    return pd.DataFrame(rows)


class TestTemporalLeakagePerturbation:
    """1. Perturbing ground-truth stats at T and T+1..T+k must not alter game T features."""

    def test_contract_schema_shared(self):
        assert FEATURE_SCHEMA_VERSION == TRAIN_SCHEMA_VERSION
        assert FEATURE_COLUMNS == TRAIN_FEATURE_COLUMNS

    def test_perturbation_invariance_causal_pipeline(self):
        # Start with a fringe/rotation player profile in earlier games, then explode to 48 min / 999 pts at T..end
        df = _make_synthetic_player_history(player_id="2544", n_games_s1=10, n_games_s2=12, seed=11)
        df.loc[:14, "minutes_played"] = 14.0
        df.loc[:14, "points"] = 6
        df.loc[:14, "assists"] = 1
        df.loc[:14, "turnovers"] = 1

        target_idx = 15  # Game T in second season
        priors = get_default_priors("NBA")

        clean_feat = compute_causal_player_features(df.copy(), league="NBA")
        clean_pts = compute_target_rolling_features(clean_feat, "points", position="SF", cached_priors=priors)

        # Corrupt ALL ground-truth stats at game T and all future games T+1..end
        perturbed = df.copy()
        perturbed.loc[target_idx:, "minutes_played"] = 48.0
        perturbed.loc[target_idx:, "points"] = 999
        perturbed.loc[target_idx:, "rebounds"] = 60
        perturbed.loc[target_idx:, "assists"] = 45
        perturbed.loc[target_idx:, "turnovers"] = 20
        perturbed.loc[target_idx:, "steals"] = 15
        perturbed.loc[target_idx:, "blocks"] = 15
        perturbed.loc[target_idx:, "three_pointers_made"] = 30
        perturbed.loc[target_idx:, "field_goals_attempted"] = 80
        perturbed.loc[target_idx:, "free_throws_attempted"] = 40
        perturbed.loc[target_idx:, "plus_minus"] = -65.0
        perturbed.loc[target_idx:, "potential_assists"] = 70.0
        perturbed.loc[target_idx:, "touches"] = 250.0
        perturbed.loc[target_idx:, "passes_made"] = 200.0
        perturbed.loc[target_idx:, "time_of_possession_sec"] = 900.0
        perturbed.loc[target_idx:, "avg_defender_distance_ft"] = 0.5
        perturbed.loc[target_idx:, "shorthanded_points"] = 55.0

        pert_feat = compute_causal_player_features(perturbed, league="NBA")
        pert_pts = compute_target_rolling_features(pert_feat, "points", position="SF", cached_priors=priors)

        numeric_check_cols = [
            c for c in FEATURE_COLUMNS if c not in ("player_id", "team_id", "opponent_team_id")
        ] + ["tier_alpha", "bayesian_roll_stat", "roll_fga", "roll_fta", "roll_expected_points"]

        clean_vec = clean_pts.loc[target_idx, numeric_check_cols].to_numpy(dtype=float)
        pert_vec = pert_pts.loc[target_idx, numeric_check_cols].to_numpy(dtype=float)

        assert np.all(np.isfinite(clean_vec))
        assert np.allclose(clean_vec, pert_vec, rtol=0.0, atol=1e-12)
        assert clean_pts.loc[target_idx, "player_tier"] == pert_pts.loc[target_idx, "player_tier"]

    def test_perturbation_invariance_contextual_feature_engine(self):
        df = _make_synthetic_player_history(player_id="1629029", n_games_s1=8, n_games_s2=8, seed=22)
        target_idx = 11

        with patch("sqlalchemy.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_create.return_value = mock_engine

            with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
                fe = ContextualFeatureEngine(league="NBA")
                fe.player_teams = {"1629029": 7}
                fe.player_positions = {"1629029": "PG"}
                fe.get_co_stars = MagicMock(return_value=[])

                clean_df = fe.build_player_feature_matrix("1629029", df.copy())

                perturbed = df.copy()
                perturbed.loc[target_idx:, "points"] = 999
                perturbed.loc[target_idx:, "minutes_played"] = 48.0
                perturbed.loc[target_idx:, "assists"] = 50
                perturbed.loc[target_idx:, "rebounds"] = 40
                perturbed.loc[target_idx:, "plus_minus"] = 99.0
                perturbed.loc[target_idx:, "shorthanded_points"] = 88.0

                pert_df = fe.build_player_feature_matrix("1629029", perturbed)

        check_cols = [
            "roll_min",
            "roll_usg",
            "roll_potential_assists",
            "roll_touches",
            "roll_passes_made",
            "roll_time_of_possession_sec",
            "roll_defender_distance",
            "roll_shorthanded_points",
            "roll_plus_minus",
            "roll_ts_pct",
            "roll_reb_rate",
            "roll_ast_rate",
            "roll_usg_efficiency",
            "player_tier",
            "tier_alpha",
            "pace_differential",
            "vacated_usage",
        ]
        assert np.allclose(
            clean_df.loc[target_idx, check_cols].to_numpy(dtype=float),
            pert_df.loc[target_idx, check_cols].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )

    def test_vacated_usage_strict_dnp_no_in_game_injury_leak(self):
        """Teammate early injury (mins > 0 and < 5) must NOT count as vacated_usage; only DNP (0.0) counts."""
        df = _make_synthetic_player_history(player_id="1629029", n_games_s1=2, n_games_s2=0, seed=33).drop(
            columns=["vacated_usage"]
        )
        g1, g2 = int(df.loc[0, "game_id"]), int(df.loc[1, "game_id"])

        with patch("sqlalchemy.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            # Game g1: co_star played 2.0 mins (early exit/ejection -> NOT knowable pre-game -> 0.0)
            # Game g2: co_star played 0.0 mins / absent (DNP -> 0.28)
            mock_conn.execute.side_effect = [
                MagicMock(fetchall=MagicMock(return_value=[])),
                MagicMock(fetchall=MagicMock(return_value=[(g1, "co_star_a", 2.0)])),
            ]
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_create.return_value = mock_engine

            with patch.object(ContextualFeatureEngine, "_load_player_info", return_value=None):
                fe = ContextualFeatureEngine(league="NBA")
                fe.player_teams = {"1629029": 7}
                fe.player_positions = {"1629029": "PG"}
                fe.get_co_stars = MagicMock(return_value=[{"player_id": "co_star_a", "avg_usg": 0.28}])

                out_df = fe.build_player_feature_matrix("1629029", df.copy())

        assert abs(float(out_df.loc[out_df["game_id"] == g1, "vacated_usage"].iloc[0]) - 0.0) < 1e-12
        assert abs(float(out_df.loc[out_df["game_id"] == g2, "vacated_usage"].iloc[0]) - 0.28) < 1e-12

    def test_get_co_stars_historical_team_on_target_date(self):
        """Traded player with mutable current_team_id=14 must resolve to Team 7 pre-trade and Team 14 post-trade."""
        import sqlalchemy

        eng = sqlalchemy.create_engine("sqlite:///:memory:")
        with eng.begin() as conn:
            conn.execute(sqlalchemy.text("CREATE TABLE teams (team_id INT PRIMARY KEY, abbreviation TEXT)"))
            conn.execute(sqlalchemy.text("CREATE TABLE players (master_player_id TEXT PRIMARY KEY, position TEXT, current_team_id INT)"))
            conn.execute(sqlalchemy.text("CREATE TABLE games (game_id INT PRIMARY KEY, home_team_id INT, away_team_id INT, start_time TEXT)"))
            conn.execute(sqlalchemy.text("CREATE TABLE player_game_logs (game_id INT, player_id TEXT, minutes_played REAL, usage_rate REAL)"))

            conn.execute(sqlalchemy.text("INSERT INTO teams VALUES (7, 'DAL'), (14, 'LAL'), (2, 'BOS'), (5, 'DEN')"))
            # Star '1629029' already updated to current_team_id=14 (LAL), teammate '202681' on DAL (7)
            conn.execute(sqlalchemy.text("INSERT INTO players VALUES ('1629029', 'PG', 14), ('202681', 'PG', 7), ('2544', 'SF', 14)"))

            # Pre-trade games (Jan 2025): 1629029 and 202681 play for Team 7 (DAL)
            pre_games = [
                (1, 7, 2, "2025-01-01 19:00:00"),
                (2, 5, 7, "2025-01-03 19:00:00"),
                (3, 7, 14, "2025-01-05 19:00:00"),
                (4, 2, 7, "2025-01-07 19:00:00"),
                (5, 7, 5, "2025-01-09 19:00:00"),
            ]
            # Post-trade games (Mar 2025): 1629029 traded to Team 14 (LAL), while 202681 stays on Team 7 (DAL)
            post_games = [
                (6, 14, 2, "2025-03-01 19:00:00"),
                (7, 5, 14, "2025-03-03 19:00:00"),
                (8, 14, 5, "2025-03-05 19:00:00"),
                (9, 2, 14, "2025-03-07 19:00:00"),
                (10, 14, 7, "2025-03-09 19:00:00"),
            ]
            for g in pre_games + post_games:
                conn.execute(sqlalchemy.text("INSERT INTO games VALUES (:gid, :h, :a, :st)"), {"gid": g[0], "h": g[1], "a": g[2], "st": g[3]})
                if g[0] <= 5:
                    conn.execute(sqlalchemy.text("INSERT INTO player_game_logs VALUES (:gid, '1629029', 36.0, 0.34)"), {"gid": g[0]})
                    conn.execute(sqlalchemy.text("INSERT INTO player_game_logs VALUES (:gid, '202681', 34.0, 0.29)"), {"gid": g[0]})
                else:
                    conn.execute(sqlalchemy.text("INSERT INTO player_game_logs VALUES (:gid, '1629029', 36.0, 0.34)"), {"gid": g[0]})
                    conn.execute(sqlalchemy.text("INSERT INTO player_game_logs VALUES (:gid, '2544', 35.0, 0.31)"), {"gid": g[0]})
                    if g[0] == 10:
                        conn.execute(sqlalchemy.text("INSERT INTO player_game_logs VALUES (10, '202681', 34.0, 0.29)"))

        fe = ContextualFeatureEngine.__new__(ContextualFeatureEngine)
        fe.engine = eng
        fe.league = "NBA"
        fe.player_teams = {"1629029": 14, "202681": 7, "2544": 14}
        fe.player_positions = {"1629029": "PG", "202681": "PG", "2544": "SF"}

        dal_pre = {c["player_id"] for c in fe.get_co_stars(team_id=7, exclude_player_id="202681", target_date="2025-01-15")}
        lal_pre = {c["player_id"] for c in fe.get_co_stars(team_id=14, exclude_player_id="2544", target_date="2025-01-15")}
        dal_post = {c["player_id"] for c in fe.get_co_stars(team_id=7, exclude_player_id="202681", target_date="2025-03-15")}
        lal_post = {c["player_id"] for c in fe.get_co_stars(team_id=14, exclude_player_id="2544", target_date="2025-03-15")}

        assert "1629029" in dal_pre
        assert "1629029" not in lal_pre
        assert "1629029" not in dal_post
        assert "1629029" in lal_post


class TestBatchOnlineParity:
    """2. 50 historical player-game snapshots replayed via offline batch vs online inference."""

    def test_50_snapshots_batch_vs_online_parity(self):
        models = load_models("models/wf_fold2")
        assert "pts" in models and "xgb" in models["pts"] and "lgb" in models["pts"]

        priors = get_default_priors("NBA")
        players = [("2544", 14, "SF"), ("201939", 10, "PG"), ("203999", 8, "C")]
        player_dfs = [
            _make_synthetic_player_history(player_id=pid, team_id=tid, n_games_s1=10, n_games_s2=22, seed=100 + i)
            for i, (pid, tid, _) in enumerate(players)
        ]
        pos_map = {pid: pos for pid, _, pos in players}

        batch_rows = []
        online_rows = []
        snapshots_checked = 0

        for df_p in player_dfs:
            pid = str(df_p["player_id"].iloc[0])
            pos = pos_map[pid]
            feat_df = compute_causal_player_features(df_p, league="NBA")
            stat_df = compute_target_rolling_features(
                feat_df, stat_col="points", position=pos, cached_priors=priors, league="NBA"
            )

            # Sample second-season games (where prior games in season exist) up to 50 total snapshots
            s2_indices = stat_df.index[stat_df["season"] == "2025-26"].tolist()[2:]
            for idx in s2_indices:
                if snapshots_checked >= 50:
                    break
                batch_row = stat_df.loc[idx]
                history_slice = df_p.iloc[:idx].copy()
                target_ctx = {
                    c: df_p.iloc[idx][c]
                    for c in [
                        "game_id",
                        "start_time",
                        "season",
                        "home_team_id",
                        "away_team_id",
                        "team_id",
                        "opponent_team_id",
                        "days_rest",
                        "is_back_to_back",
                        "is_home_game",
                        "opp_def_rating",
                        "opp_pace_factor",
                        "opp_pos_pts_allowed",
                        "pace_differential",
                        "vacated_usage",
                    ]
                    if c in df_p.columns
                }
                online_row, active_cols = build_online_feature_row(
                    player_id=pid,
                    history_df=history_slice,
                    target_context=target_ctx,
                    stat_col="points",
                    position=pos,
                    cached_priors=priors,
                    league="NBA",
                )
                assert active_cols == FEATURE_COLUMNS
                batch_rows.append(batch_row)
                online_rows.append(online_row)
                snapshots_checked += 1

        assert snapshots_checked == 50

        batch_df = pd.DataFrame(batch_rows).reset_index(drop=True)
        online_df = pd.DataFrame(online_rows).reset_index(drop=True)

        numeric_cols = [c for c in FEATURE_COLUMNS if c not in ("player_id", "team_id", "opponent_team_id")] + [
            "bayesian_roll_stat"
        ]
        assert np.allclose(
            batch_df[numeric_cols].to_numpy(dtype=float),
            online_df[numeric_cols].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        assert (batch_df["player_id"].astype(str) == online_df["player_id"].astype(str)).all()
        assert (batch_df["team_id"].astype(int) == online_df["team_id"].astype(int)).all()
        assert (batch_df["opponent_team_id"].astype(int) == online_df["opponent_team_id"].astype(int)).all()

        preds_batch = predict_meta_ensemble(
            batch_df,
            FEATURE_COLUMNS,
            xgb_model=models["pts"]["xgb"],
            lgb_model=models["pts"]["lgb"],
            meta_data=models["pts"]["meta"],
            ewma_baseline=batch_df["bayesian_roll_stat"].to_numpy(dtype=float),
            xgb_categories=models["pts"].get("xgb_categories"),
        )
        preds_online = predict_meta_ensemble(
            online_df,
            FEATURE_COLUMNS,
            xgb_model=models["pts"]["xgb"],
            lgb_model=models["pts"]["lgb"],
            meta_data=models["pts"]["meta"],
            ewma_baseline=online_df["bayesian_roll_stat"].to_numpy(dtype=float),
            xgb_categories=models["pts"].get("xgb_categories"),
        )

        for key in ("meta_prediction", "xgb_residual", "lgb_residual", "ewma_baseline"):
            assert np.allclose(preds_batch[key], preds_online[key], rtol=0.0, atol=1e-6)


class TestRowByRowVsVectorizedEvaluatorParity:
    """3. evaluate_fast.py vectorized evaluator must match single-row sequential replay on >= 200 rows."""

    def test_200_row_vectorized_vs_sequential_parity(self):
        models = load_models("models/wf_fold2")
        priors = get_default_priors("NBA")

        # 10 players x 25 test-season games -> 240 valid evaluated rows per stat (720 total predictions)
        player_specs = [
            ("2544", 14, "SF"),
            ("201939", 10, "PG"),
            ("203999", 8, "C"),
            ("1629029", 7, "PG"),
            ("1628369", 2, "SF"),
            ("203507", 17, "PF"),
            ("201142", 24, "PF"),
            ("1630162", 18, "SG"),
            ("1628983", 21, "PG"),
            ("203954", 23, "C"),
        ]
        player_dfs = [
            _make_synthetic_player_history(player_id=pid, team_id=tid, n_games_s1=8, n_games_s2=25, seed=200 + i)
            for i, (pid, tid, _) in enumerate(player_specs)
        ]
        pos_map = {pid: pos for pid, _, pos in player_specs}

        vec_df, vec_summary = evaluate_dataframe_vectorized(
            player_dfs=player_dfs,
            models=models,
            test_seasons=["2025-26"],
            player_positions=pos_map,
            cached_priors=priors,
            league="NBA",
            compute_crps=False,
        )
        seq_df, seq_summary = evaluate_dataframe_sequential(
            player_dfs=player_dfs,
            models=models,
            test_seasons=["2025-26"],
            player_positions=pos_map,
            cached_priors=priors,
            league="NBA",
            compute_crps=False,
        )

        # Ensure >= 200 rows per stat category
        pts_vec = vec_df[vec_df["stat_category"] == "pts"].reset_index(drop=True)
        assert len(pts_vec) >= 200
        assert len(vec_df) == len(seq_df)

        sort_cols = ["stat_category", "player_id", "start_time", "game_id"]
        vec_sorted = vec_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        seq_sorted = seq_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        assert (vec_sorted["player_id"] == seq_sorted["player_id"]).all()
        assert (vec_sorted["stat_category"] == seq_sorted["stat_category"]).all()

        for col in ("meta_prediction", "ewma_baseline", "xgb_residual", "lgb_residual", "actual_value"):
            assert np.allclose(
                vec_sorted[col].to_numpy(dtype=float),
                seq_sorted[col].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-6,
            )

        vec_overall = [r for r in vec_summary if r["Category"] == "OVERALL"][0]
        seq_overall = [r for r in seq_summary if r["Category"] == "OVERALL"][0]
        assert abs(vec_overall["Meta MAE"] - seq_overall["Meta MAE"]) < 1e-12
        assert abs(vec_overall["EWMA MAE"] - seq_overall["EWMA MAE"]) < 1e-12


class TestTrainSeasonsLeakageRegression:
    """Regression test verifying train_global_ensemble strictly isolates training seasons."""

    def test_train_global_ensemble_filters_future_seasons(self, tmp_path):
        # Mix 2023-24 (train) and 2024-25 (future test) rows for two players
        p1 = _make_synthetic_player_history("2544", team_id=14, n_games_s1=15, n_games_s2=15, seed=1)
        p1["season"] = ["2023-24"] * 15 + ["2024-25"] * 15
        p2 = _make_synthetic_player_history("201939", team_id=10, n_games_s1=15, n_games_s2=15, seed=2)
        p2["season"] = ["2023-24"] * 15 + ["2024-25"] * 15

        # Expected clean training rows if strictly restricted to 2023-24:
        # Each player has 15 rows in 2023-24; row 0 dropped due to shift(1) NaN -> 14 rows x 2 = 28 samples.
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(fetchall=MagicMock(return_value=[("2544",), ("201939",)])),
            MagicMock(scalar=MagicMock(return_value="2024-04-15T00:00:00Z")),
            MagicMock(fetchall=MagicMock(return_value=[("2544", "SF"), ("201939", "PG")])),
        ]
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        mock_fe = MagicMock()
        mock_fe.build_player_feature_matrix.side_effect = [p1.copy(), p2.copy()]

        with patch("src.train_global_model.sqlalchemy.create_engine", return_value=mock_engine), \
             patch("src.train_global_model.ContextualFeatureEngine", return_value=mock_fe):
            train_global_ensemble(
                target_keys=["pts"],
                train_seasons=["2023-24"],
                model_dir=str(tmp_path),
                league="NBA",
            )

        import json
        meta = json.load(open(tmp_path / "global_xpts_meta.json"))
        assert meta["train_seasons"] == ["2023-24"]
        assert meta["training_samples"] == 28
        assert meta["feature_schema_version"] == FEATURE_SCHEMA_VERSION
