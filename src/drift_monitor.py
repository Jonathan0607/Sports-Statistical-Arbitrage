"""Production rolling calibration, covariate drift detection, and automated circuit breaker.

Replaces the legacy WNBA-only ad-hoc script while preserving optional Discord webhook alerting
and CLI execution. Supports SQLite (including in-memory/temp files for tests) and PostgreSQL
via SQLAlchemy Core.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

logger = logging.getLogger("DriftMonitor")

DEFAULT_MONITORED_FEATURES: Tuple[str, ...] = (
    "roll_min",
    "roll_stat",
    "roll_usg",
    "days_rest",
    "opp_def_rating",
    "opp_pace_factor",
    "opp_pos_pts_allowed",
    "vacated_usage",
    "pace_differential",
    "roll_stat_var",
    "roll_residual",
    "roll_ts_pct",
    "roll_touches",
)

DEFAULT_REFERENCE_STATS_PATH = os.getenv(
    "DRIFT_REFERENCE_STATS_PATH", "models/registry/reference_stats.json"
)
DEFAULT_TELEMETRY_DB_URL = os.getenv(
    "DRIFT_TELEMETRY_URI", "sqlite:///logs/prediction_telemetry.sqlite"
)


class DriftAction(str, Enum):
    HEALTHY = "HEALTHY"
    WARN_RETRAIN = "WARN_RETRAIN"
    CIRCUIT_BREAK_FALLBACK_EWMA = "CIRCUIT_BREAK_FALLBACK_EWMA"


@dataclass
class DriftThresholds:
    """Configurable alert and circuit-breaker thresholds."""

    min_settled_samples: int = 30
    min_covariate_samples: int = 30
    warn_ece: float = 0.06
    critical_ece: float = 0.12
    warn_brier: float = 0.26
    critical_brier: float = 0.32
    warn_log_loss: float = 0.72
    critical_log_loss: float = 0.85
    warn_mae_degradation: float = 0.02
    critical_mae_degradation: float = 0.15
    warn_psi: float = 0.10
    critical_psi: float = 0.25
    warn_ks_pvalue: float = 0.01
    critical_ks_pvalue: float = 0.001
    ks_critical_min_stat: float = 0.20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_utc(ts: Optional[Any]) -> str:
    if ts is None:
        return _utc_now().isoformat()
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return _utc_now().isoformat()
    return parsed.to_pydatetime().astimezone(timezone.utc).isoformat()


def compute_ece(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """Computes 10-bin Expected Calibration Error (ECE) over predicted hit probabilities."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[valid], 0.0, 1.0)
    y = np.clip(y[valid], 0.0, 1.0)
    n = len(p)
    if n == 0:
        return 0.0, []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_out: List[Dict[str, Any]] = []
    for i in range(n_bins):
        low, high = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        count = int(np.sum(mask))
        if count == 0:
            bins_out.append(
                {
                    "bin_low": round(low, 2),
                    "bin_high": round(high, 2),
                    "count": 0,
                    "mean_prob": None,
                    "empirical_rate": None,
                    "gap": 0.0,
                }
            )
            continue
        mean_prob = float(np.mean(p[mask]))
        emp_rate = float(np.mean(y[mask]))
        gap = abs(emp_rate - mean_prob)
        ece += (count / n) * gap
        bins_out.append(
            {
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": count,
                "mean_prob": round(mean_prob, 6),
                "empirical_rate": round(emp_rate, 6),
                "gap": round(gap, 6),
            }
        )
    return float(ece), bins_out


def compute_brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    if not np.any(valid):
        return 0.0
    return float(np.mean((np.clip(p[valid], 0.0, 1.0) - np.clip(y[valid], 0.0, 1.0)) ** 2))


def compute_log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(p) & np.isfinite(y)
    if not np.any(valid):
        return 0.0
    pc = np.clip(p[valid], eps, 1.0 - eps)
    yc = np.clip(y[valid], 0.0, 1.0)
    return float(-np.mean(yc * np.log(pc) + (1.0 - yc) * np.log(1.0 - pc)))


def compute_psi(
    reference_counts_or_probs: Sequence[float],
    bin_edges: Sequence[float],
    current_values: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """Computes Population Stability Index (PSI) for a numeric feature against reference bins."""
    vals = np.asarray(current_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or len(bin_edges) < 2:
        return 0.0

    edges = np.asarray(bin_edges, dtype=float).copy()
    # Expand outer edges so extreme OOD test values fall into end buckets cleanly
    edges[0] = -np.inf
    edges[-1] = np.inf
    curr_counts, _ = np.histogram(vals, bins=edges)
    curr_props = curr_counts.astype(float) / max(1, int(np.sum(curr_counts)))

    ref_arr = np.asarray(reference_counts_or_probs, dtype=float)
    ref_sum = float(np.sum(ref_arr))
    ref_props = ref_arr / ref_sum if ref_sum > 0 else np.full_like(curr_props, 1.0 / len(curr_props))

    # Align lengths defensively
    m = min(len(ref_props), len(curr_props))
    ref_props = np.clip(ref_props[:m], eps, None)
    curr_props = np.clip(curr_props[:m], eps, None)
    ref_props = ref_props / np.sum(ref_props)
    curr_props = curr_props / np.sum(curr_props)

    psi = float(np.sum((curr_props - ref_props) * np.log(curr_props / ref_props)))
    return max(0.0, psi)


def build_reference_stats(
    df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_MONITORED_FEATURES,
    n_bins: int = 10,
    max_sample_size: int = 1000,
    seed: int = 42,
    output_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds serialized reference distribution quantiles and subsamples for PSI + KS drift checks."""
    rng = np.random.default_rng(seed)
    feature_stats: Dict[str, Any] = {}

    for col in features:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(vals, quantiles)
        # Ensure strictly increasing bin edges when distributions are tied/zero-heavy
        unique_edges = np.unique(edges)
        if len(unique_edges) < 3:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if abs(vmax - vmin) < 1e-9:
                edges = np.linspace(vmin - 0.5, vmax + 0.5, n_bins + 1)
            else:
                edges = np.linspace(vmin, vmax, n_bins + 1)
        else:
            # Perturb ties monotonically if some quantiles collapse
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-6

        hist_edges = edges.copy()
        hist_edges[0] = -np.inf
        hist_edges[-1] = np.inf
        counts, _ = np.histogram(vals, bins=hist_edges)
        props = (counts.astype(float) / max(1, int(np.sum(counts)))).tolist()

        if len(vals) > max_sample_size:
            idx = rng.choice(len(vals), size=max_sample_size, replace=False)
            sample = vals[idx].tolist()
        else:
            sample = vals.tolist()

        feature_stats[col] = {
            "count": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "bin_edges": [float(x) for x in edges],
            "bin_proportions": [float(x) for x in props],
            "sample": [float(x) for x in sample],
        }

    payload: Dict[str, Any] = {
        "generated_at": _utc_now().isoformat(),
        "n_rows": int(len(df)),
        "metadata": metadata or {},
        "features": feature_stats,
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
    return payload


class DriftMonitor:
    """Rolling telemetry logger, calibration & covariate drift analyzer, and EWMA circuit breaker."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        reference_stats_path: Optional[str] = DEFAULT_REFERENCE_STATS_PATH,
        reference_stats: Optional[Dict[str, Any]] = None,
        thresholds: Optional[DriftThresholds] = None,
        discord_webhook_url: Optional[str] = None,
    ):
        self.db_url = db_url or DEFAULT_TELEMETRY_DB_URL
        if self.db_url.startswith("sqlite:///") and not self.db_url.endswith(":memory:"):
            sqlite_path = self.db_url[len("sqlite:///") :]
            parent = os.path.dirname(os.path.abspath(sqlite_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        engine_kwargs: Dict[str, Any] = {}
        if self.db_url.startswith("sqlite:///:memory:"):
            from sqlalchemy.pool import StaticPool

            engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine_kwargs["poolclass"] = StaticPool

        self.engine = sqlalchemy.create_engine(self.db_url, **engine_kwargs)
        self.metadata = sqlalchemy.MetaData()
        self.telemetry_table = sqlalchemy.Table(
            "prediction_telemetry",
            self.metadata,
            sqlalchemy.Column("prediction_id", sqlalchemy.String(64), primary_key=True),
            sqlalchemy.Column("timestamp", sqlalchemy.String(64), nullable=False, index=True),
            sqlalchemy.Column("model_version", sqlalchemy.String(64), nullable=False),
            sqlalchemy.Column("feature_schema_version", sqlalchemy.String(32), nullable=False),
            sqlalchemy.Column("player_id", sqlalchemy.String(64), nullable=False, index=True),
            sqlalchemy.Column("stat_key", sqlalchemy.String(32), nullable=False, index=True),
            sqlalchemy.Column("predicted_mean", sqlalchemy.Float, nullable=False),
            sqlalchemy.Column("ewma_baseline", sqlalchemy.Float, nullable=False),
            sqlalchemy.Column("predicted_prob", sqlalchemy.Float, nullable=True),
            sqlalchemy.Column("line", sqlalchemy.Float, nullable=True),
            sqlalchemy.Column("feature_snapshot_json", sqlalchemy.Text, nullable=False),
            sqlalchemy.Column("realized_value", sqlalchemy.Float, nullable=True),
            sqlalchemy.Column("realized_hit", sqlalchemy.Integer, nullable=True),
        )
        self.metadata.create_all(self.engine)

        self.thresholds = thresholds or DriftThresholds()
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_DRIFT_WEBHOOK_URL") or os.getenv("DISCORD_WEBHOOK_URL")

        self.reference_stats_path = reference_stats_path
        self.reference_stats: Dict[str, Any] = reference_stats or self._load_reference_stats(reference_stats_path)
        self._last_report: Optional[Dict[str, Any]] = None
        self._manual_fallback_override: Optional[bool] = None

    @staticmethod
    def _load_reference_stats(path: Optional[str]) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {"features": {}}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "features" in data:
                return data
        except Exception as exc:
            logger.warning(f"Failed loading drift reference stats from {path}: {exc}")
        return {"features": {}}

    def set_manual_fallback(self, enabled: Optional[bool]) -> None:
        self._manual_fallback_override = enabled
        self._last_report = None

    def log_predictions(self, records: List[Dict[str, Any]]) -> List[str]:
        if not records:
            return []
        rows: List[Dict[str, Any]] = []
        ids: List[str] = []
        for rec in records:
            pid_rec = str(rec.get("prediction_id") or uuid.uuid4().hex)
            ids.append(pid_rec)
            snap = rec.get("feature_snapshot")
            if isinstance(snap, str):
                snap_json = snap
            else:
                snap_json = json.dumps(snap or {}, default=str)
            realized_hit = rec.get("realized_hit")
            if realized_hit is not None:
                realized_hit = int(bool(realized_hit))
            rows.append(
                {
                    "prediction_id": pid_rec,
                    "timestamp": _to_iso_utc(rec.get("timestamp")),
                    "model_version": str(rec.get("model_version", "unknown")),
                    "feature_schema_version": str(rec.get("feature_schema_version", "1.0.0")),
                    "player_id": str(rec.get("player_id", "")),
                    "stat_key": str(rec.get("stat_key", "pts")),
                    "predicted_mean": float(rec["predicted_mean"]),
                    "ewma_baseline": float(rec["ewma_baseline"]),
                    "predicted_prob": (
                        float(rec["predicted_prob"])
                        if rec.get("predicted_prob") is not None
                        else None
                    ),
                    "line": float(rec["line"]) if rec.get("line") is not None else None,
                    "feature_snapshot_json": snap_json,
                    "realized_value": (
                        float(rec["realized_value"])
                        if rec.get("realized_value") is not None
                        else None
                    ),
                    "realized_hit": realized_hit,
                }
            )

        with self.engine.begin() as conn:
            conn.execute(self.telemetry_table.insert(), rows)
        self._last_report = None
        return ids

    def log_prediction(
        self,
        *,
        model_version: str,
        feature_schema_version: str,
        player_id: str,
        stat_key: str,
        predicted_mean: float,
        ewma_baseline: float,
        predicted_prob: Optional[float] = None,
        line: Optional[float] = None,
        feature_snapshot: Optional[Dict[str, Any]] = None,
        realized_value: Optional[float] = None,
        realized_hit: Optional[int] = None,
        timestamp: Optional[Any] = None,
        prediction_id: Optional[str] = None,
    ) -> str:
        return self.log_predictions(
            [
                {
                    "prediction_id": prediction_id,
                    "timestamp": timestamp,
                    "model_version": model_version,
                    "feature_schema_version": feature_schema_version,
                    "player_id": player_id,
                    "stat_key": stat_key,
                    "predicted_mean": predicted_mean,
                    "ewma_baseline": ewma_baseline,
                    "predicted_prob": predicted_prob,
                    "line": line,
                    "feature_snapshot": feature_snapshot or {},
                    "realized_value": realized_value,
                    "realized_hit": realized_hit,
                }
            ]
        )[0]

    def record_settlement(
        self,
        prediction_id: str,
        realized_value: float,
        realized_hit: Optional[bool] = None,
    ) -> bool:
        with self.engine.begin() as conn:
            row = conn.execute(
                sqlalchemy.select(self.telemetry_table.c.line).where(
                    self.telemetry_table.c.prediction_id == prediction_id
                )
            ).fetchone()
            if row is None:
                return False
            hit_val = realized_hit
            if hit_val is None and row[0] is not None:
                hit_val = bool(float(realized_value) > float(row[0]))
            conn.execute(
                self.telemetry_table.update()
                .where(self.telemetry_table.c.prediction_id == prediction_id)
                .values(
                    realized_value=float(realized_value),
                    realized_hit=int(bool(hit_val)) if hit_val is not None else None,
                )
            )
            self._last_report = None
            return True

    def fetch_recent_predictions(
        self,
        window_days: int = 7,
        max_rows: Optional[int] = None,
        settled_only: bool = False,
        reference_now: Optional[datetime] = None,
    ) -> pd.DataFrame:
        now_dt = reference_now or _utc_now()
        cutoff_iso = (now_dt - timedelta(days=max(1, int(window_days)))).astimezone(timezone.utc).isoformat()
        stmt = (
            sqlalchemy.select(self.telemetry_table)
            .where(self.telemetry_table.c.timestamp >= cutoff_iso)
            .order_by(self.telemetry_table.c.timestamp.desc())
        )
        if settled_only:
            stmt = stmt.where(self.telemetry_table.c.realized_value.isnot(None))
        if max_rows is not None and max_rows > 0:
            stmt = stmt.limit(int(max_rows))

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return pd.DataFrame([dict(r) for r in rows])

    def evaluate_drift(
        self,
        window_days: int = 7,
        max_rows: int = 2000,
        reference_now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Evaluates rolling calibration & covariate drift and resolves circuit-breaker action."""
        df = self.fetch_recent_predictions(
            window_days=window_days, max_rows=max_rows, reference_now=reference_now
        )
        total_predictions = int(len(df))

        settled_df = (
            df[df["realized_value"].notna()].copy() if not df.empty else pd.DataFrame()
        )
        settled_count = int(len(settled_df))

        calibration: Dict[str, Any] = {
            "sample_count": settled_count,
            "ece": None,
            "brier_score": None,
            "log_loss": None,
            "meta_mae": None,
            "ewma_mae": None,
            "mae_delta": None,
            "mae_improvement_pct": None,
            "bins": [],
        }

        if settled_count > 0:
            actuals = pd.to_numeric(settled_df["realized_value"], errors="coerce").to_numpy(dtype=float)
            meta_preds = pd.to_numeric(settled_df["predicted_mean"], errors="coerce").to_numpy(dtype=float)
            ewma_preds = pd.to_numeric(settled_df["ewma_baseline"], errors="coerce").to_numpy(dtype=float)

            meta_mae = float(np.mean(np.abs(actuals - meta_preds)))
            ewma_mae = float(np.mean(np.abs(actuals - ewma_preds)))
            mae_delta = float(meta_mae - ewma_mae)
            mae_imp_pct = float(((ewma_mae - meta_mae) / ewma_mae) * 100.0) if ewma_mae > 1e-9 else 0.0
            calibration["meta_mae"] = round(meta_mae, 6)
            calibration["ewma_mae"] = round(ewma_mae, 6)
            calibration["mae_delta"] = round(mae_delta, 6)
            calibration["mae_improvement_pct"] = round(mae_imp_pct, 4)

            prob_mask = settled_df["predicted_prob"].notna() & settled_df["realized_hit"].notna()
            if bool(prob_mask.any()):
                probs = pd.to_numeric(settled_df.loc[prob_mask, "predicted_prob"], errors="coerce").to_numpy(dtype=float)
                hits = pd.to_numeric(settled_df.loc[prob_mask, "realized_hit"], errors="coerce").to_numpy(dtype=float)
                ece_val, bin_details = compute_ece(probs, hits, n_bins=10)
                brier_val = compute_brier_score(probs, hits)
                ll_val = compute_log_loss(probs, hits)
                calibration["ece"] = round(ece_val, 6)
                calibration["brier_score"] = round(brier_val, 6)
                calibration["log_loss"] = round(ll_val, 6)
                calibration["bins"] = bin_details

        # Covariate drift (PSI + two-sample KS p-value)
        ref_features = (self.reference_stats or {}).get("features", {})
        cov_details: Dict[str, Any] = {}
        psi_values: List[float] = []
        ks_pvalues: List[float] = []
        ks_stats: List[float] = []

        if total_predictions >= self.thresholds.min_covariate_samples and ref_features:
            snapshots: List[Dict[str, Any]] = []
            for raw in df["feature_snapshot_json"].tolist():
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    if isinstance(parsed, dict):
                        snapshots.append(parsed)
                except Exception:
                    continue
            snap_df = pd.DataFrame(snapshots) if snapshots else pd.DataFrame()
            for feat_name, ref_info in ref_features.items():
                if feat_name not in snap_df.columns:
                    continue
                curr_vals = pd.to_numeric(snap_df[feat_name], errors="coerce").dropna().to_numpy(dtype=float)
                curr_vals = curr_vals[np.isfinite(curr_vals)]
                if len(curr_vals) < self.thresholds.min_covariate_samples:
                    continue
                psi_val = compute_psi(
                    ref_info.get("bin_proportions", []),
                    ref_info.get("bin_edges", []),
                    curr_vals,
                )
                ref_sample = np.asarray(ref_info.get("sample", []), dtype=float)
                if len(ref_sample) >= 5:
                    ks_res = stats.ks_2samp(ref_sample, curr_vals)
                    ks_stat = float(ks_res.statistic)
                    ks_pval = float(ks_res.pvalue)
                else:
                    ks_stat = 0.0
                    ks_pval = 1.0

                psi_values.append(psi_val)
                ks_pvalues.append(ks_pval)
                ks_stats.append(ks_stat)
                cov_details[feat_name] = {
                    "psi": round(psi_val, 6),
                    "ks_stat": round(ks_stat, 6),
                    "ks_pvalue": float(ks_pval),
                    "current_mean": round(float(np.mean(curr_vals)), 6),
                    "reference_mean": round(float(ref_info.get("mean", 0.0)), 6),
                    "sample_count": int(len(curr_vals)),
                }

        covariate_summary = {
            "features_checked": len(cov_details),
            "max_psi": round(max(psi_values), 6) if psi_values else None,
            "mean_psi": round(float(np.mean(psi_values)), 6) if psi_values else None,
            "min_ks_pvalue": float(min(ks_pvalues)) if ks_pvalues else None,
            "max_ks_stat": round(max(ks_stats), 6) if ks_stats else None,
            "features": cov_details,
        }

        # Evaluate circuit breaker gates
        action = DriftAction.HEALTHY
        reasons: List[str] = []
        th = self.thresholds

        def _escalate(target: DriftAction, reason: str) -> None:
            nonlocal action
            reasons.append(reason)
            if target == DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA:
                action = DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA
            elif target == DriftAction.WARN_RETRAIN and action == DriftAction.HEALTHY:
                action = DriftAction.WARN_RETRAIN

        if settled_count >= th.min_settled_samples:
            ece_val = calibration["ece"]
            brier_val = calibration["brier_score"]
            ll_val = calibration["log_loss"]
            mae_delta = calibration["mae_delta"]

            if ece_val is not None:
                if ece_val >= th.critical_ece:
                    _escalate(
                        DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA,
                        f"ECE={ece_val:.4f} >= critical_ece={th.critical_ece:.4f}",
                    )
                elif ece_val >= th.warn_ece:
                    _escalate(
                        DriftAction.WARN_RETRAIN,
                        f"ECE={ece_val:.4f} >= warn_ece={th.warn_ece:.4f}",
                    )

            if brier_val is not None:
                if brier_val >= th.critical_brier:
                    _escalate(
                        DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA,
                        f"Brier={brier_val:.4f} >= critical_brier={th.critical_brier:.4f}",
                    )
                elif brier_val >= th.warn_brier:
                    _escalate(
                        DriftAction.WARN_RETRAIN,
                        f"Brier={brier_val:.4f} >= warn_brier={th.warn_brier:.4f}",
                    )

            if ll_val is not None:
                if ll_val >= th.critical_log_loss:
                    _escalate(
                        DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA,
                        f"LogLoss={ll_val:.4f} >= critical_log_loss={th.critical_log_loss:.4f}",
                    )
                elif ll_val >= th.warn_log_loss:
                    _escalate(
                        DriftAction.WARN_RETRAIN,
                        f"LogLoss={ll_val:.4f} >= warn_log_loss={th.warn_log_loss:.4f}",
                    )

            if mae_delta is not None:
                if mae_delta >= th.critical_mae_degradation:
                    _escalate(
                        DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA,
                        f"Meta-EWMA MAE delta={mae_delta:+.4f} >= critical={th.critical_mae_degradation:.4f}",
                    )
                elif mae_delta >= th.warn_mae_degradation:
                    _escalate(
                        DriftAction.WARN_RETRAIN,
                        f"Meta-EWMA MAE delta={mae_delta:+.4f} >= warn={th.warn_mae_degradation:.4f}",
                    )

        for feat_name, stats_dict in cov_details.items():
            psi_val = stats_dict["psi"]
            ks_pval = stats_dict["ks_pvalue"]
            ks_stat = stats_dict["ks_stat"]
            if psi_val >= th.critical_psi:
                _escalate(
                    DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA,
                    f"Covariate '{feat_name}' PSI={psi_val:.4f} >= critical_psi={th.critical_psi:.4f}",
                )
            elif ks_pval <= th.critical_ks_pvalue and (psi_val >= th.warn_psi or ks_stat >= th.ks_critical_min_stat):
                _escalate(
                    DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA,
                    f"Covariate '{feat_name}' KS p={ks_pval:.2e} <= critical={th.critical_ks_pvalue:.1e} (ks={ks_stat:.3f}, psi={psi_val:.3f})",
                )
            elif psi_val >= th.warn_psi or ks_pval <= th.warn_ks_pvalue:
                _escalate(
                    DriftAction.WARN_RETRAIN,
                    f"Covariate '{feat_name}' drift warning (PSI={psi_val:.4f}, ks_p={ks_pval:.2e})",
                )

        fallback_active = action == DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA
        if self._manual_fallback_override is not None:
            fallback_active = bool(self._manual_fallback_override)
            if fallback_active and action != DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA:
                action = DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA
                reasons.append("Manual EWMA fallback override active")

        report = {
            "timestamp": _utc_now().isoformat(),
            "window_days": int(window_days),
            "total_predictions": total_predictions,
            "settled_predictions": settled_count,
            "action": action.value,
            "fallback_active": bool(fallback_active),
            "reasons": reasons,
            "calibration": calibration,
            "covariate_drift": covariate_summary,
            "thresholds": th.to_dict(),
        }
        self._last_report = report
        return report

    def current_status(self, refresh: bool = False) -> Tuple[DriftAction, bool]:
        if refresh or self._last_report is None:
            rep = self.evaluate_drift()
        else:
            rep = self._last_report
        action = DriftAction(rep["action"])
        fallback = bool(rep["fallback_active"])
        if self._manual_fallback_override is not None:
            fallback = bool(self._manual_fallback_override)
            if fallback:
                action = DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA
        return action, fallback

    def send_discord_alert(self, report: Dict[str, Any]) -> bool:
        """Sends an embedded alert payload to Discord when WARN_RETRAIN or CIRCUIT_BREAK triggers."""
        if not self.discord_webhook_url or report.get("action") == DriftAction.HEALTHY.value:
            return False

        action = report.get("action", "WARN_RETRAIN")
        color = 15158332 if action == DriftAction.CIRCUIT_BREAK_FALLBACK_EWMA.value else 16776960
        cal = report.get("calibration", {})
        cov = report.get("covariate_drift", {})
        reasons_text = "\n".join(f"- {r}" for r in report.get("reasons", [])[:8]) or "Threshold exceeded"

        payload = {
            "username": "Quant Ops Drift Circuit Breaker",
            "embeds": [
                {
                    "title": f"Drift Monitor Alert: {action}",
                    "description": reasons_text,
                    "color": color,
                    "fields": [
                        {
                            "name": "Window Samples",
                            "value": f"Total: `{report.get('total_predictions', 0)}` | Settled: `{report.get('settled_predictions', 0)}`",
                            "inline": True,
                        },
                        {
                            "name": "ECE / Brier",
                            "value": f"ECE: `{cal.get('ece')}` | Brier: `{cal.get('brier_score')}`",
                            "inline": True,
                        },
                        {
                            "name": "MAE (Meta vs EWMA)",
                            "value": f"Meta: `{cal.get('meta_mae')}` | EWMA: `{cal.get('ewma_mae')}`",
                            "inline": True,
                        },
                        {
                            "name": "Covariate PSI / KS",
                            "value": f"Max PSI: `{cov.get('max_psi')}` | Min KS p: `{cov.get('min_ks_pvalue')}`",
                            "inline": True,
                        },
                    ],
                }
            ],
        }
        try:
            req = urllib.request.Request(
                self.discord_webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "QuantOps/2.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except Exception as exc:
            logger.error(f"Failed sending Discord drift alert: {exc}")
            return False


def send_discord_drift_alert(
    edge_pct: float,
    meta_mae: float,
    naive_mae: float,
    sample_count: int,
    webhook_url: Optional[str] = None,
) -> bool:
    """Backward-compatible Discord alert helper for legacy monitoring scripts."""
    target_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not target_url:
        return False
    payload = {
        "username": "Quant Ops Monitor",
        "embeds": [
            {
                "title": "Model Calibration Drift Warning",
                "description": (
                    f"14-Day Meta-Ensemble edge vs baseline slipped to **{edge_pct:+.2f}%** "
                    f"(Meta MAE: `{meta_mae:.3f}`, EWMA MAE: `{naive_mae:.3f}`, Samples: `{sample_count}`)."
                ),
                "color": 15158332,
            }
        ],
    }
    try:
        req = urllib.request.Request(
            target_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "QuantOps/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as exc:
        logger.error(f"Failed sending Discord alert: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate production calibration and covariate drift")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument("--reference-stats", type=str, default=DEFAULT_REFERENCE_STATS_PATH)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--notify-discord", action="store_true")
    args = parser.parse_args()

    monitor = DriftMonitor(db_url=args.db_url, reference_stats_path=args.reference_stats)
    report = monitor.evaluate_drift(window_days=args.window_days)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            f.write(rendered + "\n")
    if args.notify_discord and report["action"] != DriftAction.HEALTHY.value:
        monitor.send_discord_alert(report)


if __name__ == "__main__":
    main()
