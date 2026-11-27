"""Closed-loop Kalshi Live/Sandbox Performance Scorecard and Markdown/JSON Reporter."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sqlalchemy

from src.drift_monitor import DriftMonitor
from src.oms.reconciler import SettlementReconciler


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    return float(val)


def _sample_confidence_banner(settled_trades: int) -> Tuple[str, str]:
    if settled_trades < 50:
        return (
            "WARM-UP / INSUFFICIENT SAMPLE",
            f"Only `{settled_trades}` settled live/sandbox trades (< 50 threshold); edge and ROI CIs remain noisy.",
        )
    if settled_trades < 200:
        return (
            "PRELIMINARY",
            f"`{settled_trades}` settled trades (50–199); early directional signal—continue accumulation to >= 200 trades.",
        )
    return (
        "STATISTICALLY MEANINGFUL",
        f"`{settled_trades}` settled trades (>= 200); sufficient sample size for production calibration & edge assessment.",
    )


def _compute_ece_and_buckets(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    if len(probs) == 0:
        return None, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    buckets: List[Dict[str, Any]] = []
    n = float(len(probs))
    for idx in range(n_bins):
        lo, hi = float(edges[idx]), float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_p = float(np.mean(probs[mask]))
            emp_y = float(np.mean(outcomes[mask]))
            ece += (cnt / n) * abs(mean_p - emp_y)
            buckets.append(
                {
                    "bucket": f"[{lo:.1f}, {hi:.1f}{']' if idx == n_bins - 1 else ')'}",
                    "count": cnt,
                    "mean_pred": round(mean_p, 4),
                    "empirical_hit": round(emp_y, 4),
                }
            )
    return float(ece), buckets


def _bootstrap_ci(
    values: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    seed: int = 42,
    as_pct_ratio: bool = False,
) -> Optional[Tuple[float, float]]:
    n = len(values)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if as_pct_ratio and weights is not None:
            denom = float(np.sum(weights[idx]))
            stats[b] = (100.0 * float(np.sum(values[idx])) / denom) if denom > 1e-12 else 0.0
        else:
            stats[b] = float(np.mean(values[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


def _infer_contract_hit(row: Dict[str, Any]) -> bool:
    """Infers binary traded-contract win (1) vs loss (0) cleanly from gross P&L."""
    pnl = _safe_float(row.get("realized_pnl"))
    fees = _safe_float(row.get("fees_paid"))
    gross_pnl = pnl + fees
    return bool(gross_pnl > 0.0)


def _aligned_market_prob(row: Dict[str, Any]) -> Optional[float]:
    mid = row.get("market_mid_at_entry")
    if mid is None:
        return None
    val = float(mid)
    if str(row.get("side", "")).strip().lower() == "ask":
        return float(np.clip(1.0 - val, 1e-6, 1.0 - 1e-6))
    return float(np.clip(val, 1e-6, 1.0 - 1e-6))


def _trade_capital_risked(row: Dict[str, Any]) -> float:
    cnt = int(row.get("filled_count") or 0)
    if cnt <= 0:
        return 0.0
    side = str(row.get("side", "bid")).strip().lower()
    avg_px = row.get("avg_fill_price")
    if avg_px is not None:
        cost_per = float(avg_px) if side == "bid" else (1.0 - float(avg_px))
    else:
        cost_per = float(row.get("effective_cost") or 0.0)
    return float(max(0.0, cnt * cost_per))


def _adverse_slippage_cents(row: Dict[str, Any]) -> Optional[float]:
    avg_px = row.get("avg_fill_price")
    if avg_px is None or row.get("limit_price") is None:
        return None
    limit_px = float(row["limit_price"])
    fill_px = float(avg_px)
    side = str(row.get("side", "bid")).strip().lower()
    diff = (fill_px - limit_px) if side == "bid" else (limit_px - fill_px)
    return float(diff * 100.0)


def _format_opt(val: Optional[float], fmt: str = ".4f", suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}{suffix}"


class PerformanceReporter:
    """Computes live/sandbox OMS scorecard metrics and writes Markdown + JSON reports."""

    def __init__(
        self,
        drift_monitor: DriftMonitor,
        reconciler: Optional[SettlementReconciler] = None,
    ):
        self.drift_monitor = drift_monitor
        self.reconciler = reconciler or SettlementReconciler(drift_monitor)
        self.engine = self.reconciler.engine
        self.orders_table = self.reconciler.orders_table

    def _fetch_orders(self, env: str = "demo") -> List[Dict[str, Any]]:
        target_env = (env or "demo").strip().lower()
        with self.engine.connect() as conn:
            rows = conn.execute(sqlalchemy.select(self.orders_table)).mappings().all()
        filtered: List[Dict[str, Any]] = []
        for r in rows:
            row = dict(r)
            row_env = (row.get("env") or "demo").strip().lower()
            if row_env == target_env:
                filtered.append(row)
        return filtered

    @staticmethod
    def _calibration_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        paired = [
            r
            for r in rows
            if r.get("model_prob") is not None and _aligned_market_prob(r) is not None
        ]
        if not paired:
            return {
                "sample_count": 0,
                "model_brier": None,
                "market_brier": None,
                "brier_skill_score": None,
                "model_log_loss": None,
                "market_log_loss": None,
                "delta_log_loss": None,
                "model_ece": None,
                "market_ece": None,
                "reliability_buckets": [],
            }

        p_mod = np.clip(np.array([float(r["model_prob"]) for r in paired]), 1e-6, 1.0 - 1e-6)
        p_mkt = np.array([float(_aligned_market_prob(r)) for r in paired], dtype=np.float64)
        y = np.array([1.0 if _infer_contract_hit(r) else 0.0 for r in paired], dtype=np.float64)

        brier_mod = float(np.mean((p_mod - y) ** 2))
        brier_mkt = float(np.mean((p_mkt - y) ** 2))
        bss = float(1.0 - brier_mod / brier_mkt) if brier_mkt > 1e-12 else None

        ll_mod = float(-np.mean(y * np.log(p_mod) + (1.0 - y) * np.log(1.0 - p_mod)))
        ll_mkt = float(-np.mean(y * np.log(p_mkt) + (1.0 - y) * np.log(1.0 - p_mkt)))

        ece_mod, mod_buckets = _compute_ece_and_buckets(p_mod, y, n_bins=10)
        ece_mkt, _ = _compute_ece_and_buckets(p_mkt, y, n_bins=10)

        return {
            "sample_count": len(paired),
            "model_brier": brier_mod,
            "market_brier": brier_mkt,
            "brier_skill_score": bss,
            "model_log_loss": ll_mod,
            "market_log_loss": ll_mkt,
            "delta_log_loss": float(ll_mod - ll_mkt),
            "model_ece": ece_mod,
            "market_ece": ece_mkt,
            "reliability_buckets": mod_buckets,
        }

    def build_metrics(self, env: str = "demo", recent_limit: int = 25) -> Dict[str, Any]:
        all_rows = self._fetch_orders(env=env)
        executed = [
            r
            for r in all_rows
            if r["status"] in ("submitted", "partially_filled", "filled", "settled")
        ]
        settled = [
            r
            for r in executed
            if r["status"] == "settled" and r["realized_pnl"] is not None
        ]
        settled.sort(key=lambda r: (str(r.get("settled_at") or ""), str(r.get("created_at") or "")))

        rejected = [r for r in all_rows if r["status"] == "rejected"]
        dry_run_all = [r for r in all_rows if r["status"] == "dry_run"]
        dry_run_settled = [
            r
            for r in dry_run_all
            if r.get("settled_at") is not None and r.get("realized_pnl") is not None
        ]

        total_submitted = len(executed)
        filled_orders = sum(1 for r in executed if int(r.get("filled_count") or 0) > 0)
        fill_rate_pct = (100.0 * filled_orders / total_submitted) if total_submitted > 0 else 0.0

        settled_trades = len(settled)
        wins = sum(1 for r in settled if _infer_contract_hit(r))
        win_rate_pct = (100.0 * wins / settled_trades) if settled_trades > 0 else 0.0

        risked_arr = np.array([_trade_capital_risked(r) for r in settled], dtype=np.float64)
        pnl_arr = np.array([_safe_float(r.get("realized_pnl")) for r in settled], dtype=np.float64)
        fees_arr = np.array([_safe_float(r.get("fees_paid")) for r in settled], dtype=np.float64)
        gross_arr = pnl_arr + fees_arr
        expected_ev_usd = float(
            sum(
                _safe_float(r.get("expected_ev")) * int(r.get("filled_count") or 0)
                for r in settled
            )
        )

        capital_risked_usd = float(np.sum(risked_arr)) if settled_trades > 0 else 0.0
        total_fees_usd = float(np.sum(fees_arr)) if settled_trades > 0 else 0.0
        net_pnl_usd = float(np.sum(pnl_arr)) if settled_trades > 0 else 0.0
        gross_pnl_usd = float(np.sum(gross_arr)) if settled_trades > 0 else 0.0
        net_roi_pct = (100.0 * net_pnl_usd / capital_risked_usd) if capital_risked_usd > 1e-12 else 0.0
        ev_gap_usd = float(net_pnl_usd - expected_ev_usd)

        # Max Drawdown ($ and % from peak cumulative P&L)
        cum = 0.0
        peak = 0.0
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        for pnl_val in pnl_arr:
            cum += float(pnl_val)
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd_usd:
                max_dd_usd = dd
            if peak > 1e-9:
                dd_pct = 100.0 * dd / peak
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

        clv_vals = np.array(
            [float(r["clv_bps"]) for r in settled if r.get("clv_bps") is not None],
            dtype=np.float64,
        )
        mean_clv_bps = float(np.mean(clv_vals)) if len(clv_vals) > 0 else None

        roi_ci = _bootstrap_ci(
            pnl_arr, weights=risked_arr, n_resamples=1000, seed=42, as_pct_ratio=True
        )
        clv_ci = _bootstrap_ci(clv_vals, n_resamples=1000, seed=42, as_pct_ratio=False)

        banner_title, banner_detail = _sample_confidence_banner(settled_trades)
        calibration = self._calibration_block(settled)

        # Execution microstructure by taker vs maker
        by_mode: Dict[str, Dict[str, Any]] = {}
        for mode in sorted({str(r.get("execution_mode") or "unknown") for r in executed}):
            m_exec = [r for r in executed if str(r.get("execution_mode") or "unknown") == mode]
            m_set = [r for r in settled if str(r.get("execution_mode") or "unknown") == mode]
            m_filled = sum(1 for r in m_exec if int(r.get("filled_count") or 0) > 0)
            m_fill_rate = (100.0 * m_filled / len(m_exec)) if m_exec else 0.0
            slips = [s for s in (_adverse_slippage_cents(r) for r in m_exec) if s is not None]
            m_contracts = sum(int(r.get("filled_count") or 0) for r in m_exec)
            m_fees = sum(_safe_float(r.get("fees_paid")) for r in m_exec)
            m_clvs = [float(r["clv_bps"]) for r in m_set if r.get("clv_bps") is not None]
            m_pnl = sum(_safe_float(r.get("realized_pnl")) for r in m_set)
            m_risk = sum(_trade_capital_risked(r) for r in m_set)
            by_mode[mode] = {
                "orders": len(m_exec),
                "settled": len(m_set),
                "fill_rate_pct": m_fill_rate,
                "avg_slippage_cents": float(np.mean(slips)) if slips else None,
                "avg_fee_per_contract_cents": (100.0 * m_fees / m_contracts) if m_contracts > 0 else None,
                "mean_clv_bps": float(np.mean(m_clvs)) if m_clvs else None,
                "net_pnl_usd": m_pnl,
                "roi_pct": (100.0 * m_pnl / m_risk) if m_risk > 1e-12 else 0.0,
            }

        # Breakdown by stat_key & side
        by_stat_side: List[Dict[str, Any]] = []
        stat_side_keys = sorted({(str(r["stat_key"]), str(r["side"])) for r in settled})
        for stat_key, side in stat_side_keys:
            sub = [r for r in settled if str(r["stat_key"]) == stat_key and str(r["side"]) == side]
            sub_wins = sum(1 for r in sub if _infer_contract_hit(r))
            sub_clv = [float(r["clv_bps"]) for r in sub if r.get("clv_bps") is not None]
            sub_cal = self._calibration_block(sub)
            by_stat_side.append(
                {
                    "stat_key": stat_key,
                    "side": side,
                    "settled": len(sub),
                    "win_rate_pct": 100.0 * sub_wins / len(sub),
                    "mean_clv_bps": float(np.mean(sub_clv)) if sub_clv else None,
                    "model_brier": sub_cal["model_brier"],
                    "market_brier": sub_cal["market_brier"],
                    "net_pnl_usd": sum(_safe_float(r.get("realized_pnl")) for r in sub),
                }
            )

        # Breakdown by model_version
        by_model_version: List[Dict[str, Any]] = []
        versions = sorted({str(r.get("model_version") or "unknown") for r in settled})
        for mv in versions:
            sub = [r for r in settled if str(r.get("model_version") or "unknown") == mv]
            sub_wins = sum(1 for r in sub if _infer_contract_hit(r))
            sub_clv = [float(r["clv_bps"]) for r in sub if r.get("clv_bps") is not None]
            sub_cal = self._calibration_block(sub)
            sub_pnl = sum(_safe_float(r.get("realized_pnl")) for r in sub)
            sub_risk = sum(_trade_capital_risked(r) for r in sub)
            by_model_version.append(
                {
                    "model_version": mv,
                    "settled": len(sub),
                    "win_rate_pct": 100.0 * sub_wins / len(sub),
                    "mean_clv_bps": float(np.mean(sub_clv)) if sub_clv else None,
                    "model_brier": sub_cal["model_brier"],
                    "net_pnl_usd": sub_pnl,
                    "roi_pct": (100.0 * sub_pnl / sub_risk) if sub_risk > 1e-12 else 0.0,
                }
            )

        # Guardrail rejections & Drift status
        rej_counts: Dict[str, int] = {}
        for r in rejected:
            reason = str(r.get("rejection_reason") or "unspecified")
            rej_counts[reason] = rej_counts.get(reason, 0) + 1

        drift_rep = self.drift_monitor.evaluate_drift()

        # Recent settled ledger (latest 25 descending)
        recent_settled: List[Dict[str, Any]] = []
        for r in reversed(settled[-recent_limit:]):
            recent_settled.append(
                {
                    "settled_at": r.get("settled_at") or r.get("created_at"),
                    "ticker": r["ticker"],
                    "stat_key": r["stat_key"],
                    "side": r["side"],
                    "execution_mode": r["execution_mode"],
                    "model_prob": float(r["model_prob"]),
                    "fill_price": r.get("avg_fill_price") if r.get("avg_fill_price") is not None else r["limit_price"],
                    "closing_mid_price": r.get("closing_mid_price"),
                    "clv_bps": r.get("clv_bps"),
                    "fees_paid": _safe_float(r.get("fees_paid")),
                    "hit": _infer_contract_hit(r),
                    "realized_pnl": _safe_float(r.get("realized_pnl")),
                }
            )

        # Dry-Run shadow summary
        dry_wins = sum(1 for r in dry_run_settled if _infer_contract_hit(r))
        dry_pnl = sum(_safe_float(r.get("realized_pnl")) for r in dry_run_settled)
        dry_clvs = [float(r["clv_bps"]) for r in dry_run_settled if r.get("clv_bps") is not None]
        dry_cal = self._calibration_block(dry_run_settled)

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "env": (env or "demo").strip().lower(),
            "confidence_banner": {"status": banner_title, "detail": banner_detail},
            "topline": {
                "total_submitted_orders": total_submitted,
                "filled_orders": filled_orders,
                "fill_rate_pct": fill_rate_pct,
                "settled_trades": settled_trades,
                "win_rate_pct": win_rate_pct,
                "capital_risked_usd": capital_risked_usd,
                "gross_pnl_usd": gross_pnl_usd,
                "total_fees_paid_usd": total_fees_usd,
                "net_realized_pnl_usd": net_pnl_usd,
                "net_roi_pct": net_roi_pct,
                "expected_ev_usd": expected_ev_usd,
                "ev_realization_gap_usd": ev_gap_usd,
                "max_drawdown_usd": max_dd_usd,
                "max_drawdown_pct": max_dd_pct,
                "mean_clv_bps": mean_clv_bps,
                "net_roi_ci95_pct": roi_ci,
                "mean_clv_ci95_bps": clv_ci,
            },
            "calibration_head_to_head": calibration,
            "by_execution_mode": by_mode,
            "by_stat_and_side": by_stat_side,
            "by_model_version": by_model_version,
            "guardrails_and_drift": {
                "drift_action": drift_rep.get("action"),
                "fallback_active": bool(drift_rep.get("fallback_active", False)),
                "rolling_ece": (drift_rep.get("calibration") or {}).get("ece"),
                "max_psi": (drift_rep.get("covariate_drift") or {}).get("max_psi"),
                "total_rejected_orders": len(rejected),
                "rejection_breakdown": rej_counts,
            },
            "recent_settled_trades": recent_settled,
            "dry_run_shadow": {
                "total_dry_run_orders": len(dry_run_all),
                "settled_shadow_trades": len(dry_run_settled),
                "shadow_win_rate_pct": (100.0 * dry_wins / len(dry_run_settled)) if dry_run_settled else 0.0,
                "shadow_mean_clv_bps": float(np.mean(dry_clvs)) if dry_clvs else None,
                "shadow_net_pnl_usd": dry_pnl,
                "shadow_model_brier": dry_cal["model_brier"],
                "shadow_market_brier": dry_cal["market_brier"],
            },
        }

    def render_markdown(self, metrics: Dict[str, Any]) -> str:
        t = metrics["topline"]
        cal = metrics["calibration_head_to_head"]
        banner = metrics["confidence_banner"]
        gd = metrics["guardrails_and_drift"]
        shadow = metrics["dry_run_shadow"]

        roi_ci = t["net_roi_ci95_pct"]
        clv_ci = t["mean_clv_ci95_bps"]
        roi_ci_str = f"[{roi_ci[0]:+.2f}%, {roi_ci[1]:+.2f}%]" if roi_ci else "N/A (<2 settled)"
        clv_ci_str = f"[{clv_ci[0]:+.1f}, {clv_ci[1]:+.1f}] bps" if clv_ci else "N/A (<2 CLV samples)"

        lines: List[str] = [
            f"# Kalshi Closed-Loop Performance Scorecard (`env={metrics['env']}`)",
            "",
            f"*Generated at:* `{metrics['generated_at_utc']}`",
            "",
            f"> **Sample-Size Confidence: {banner['status']}** — {banner['detail']}",
            "",
            "## 1. Top-Line Financial & Sample-Size Scorecard",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Total Submitted Orders | `{t['total_submitted_orders']}` |",
            f"| Filled Orders | `{t['filled_orders']}` (`{t['fill_rate_pct']:.1f}%` fill rate) |",
            f"| Settled Trades | `{t['settled_trades']}` (`{t['win_rate_pct']:.1f}%` win rate) |",
            f"| Capital Risked | `${t['capital_risked_usd']:,.2f}` |",
            f"| Gross P&L | `${t['gross_pnl_usd']:+,.2f}` |",
            f"| Total Fees Paid | `${t['total_fees_paid_usd']:,.2f}` |",
            f"| **Net Realized P&L** | **`${t['net_realized_pnl_usd']:+,.2f}`** |",
            f"| **Net ROI** | **`{t['net_roi_pct']:+.2f}%`** (95% CI: `{roi_ci_str}`) |",
            f"| Expected EV vs. Realized Gap | `${t['expected_ev_usd']:+,.2f}` vs. `${t['net_realized_pnl_usd']:+,.2f}` (gap `${t['ev_realization_gap_usd']:+,.2f}`) |",
            f"| Max Drawdown (Peak Cum P&L) | `${t['max_drawdown_usd']:,.2f}` (`{t['max_drawdown_pct']:.2f}%`) |",
            f"| **Mean CLV** | **`{_format_opt(t['mean_clv_bps'], '+.1f', ' bps')}`** (95% CI: `{clv_ci_str}`) |",
            "",
            "## 2. Model vs. Kalshi Market Calibration Head-to-Head",
            "",
            "| Calibration Metric | Model | Kalshi Market Mid | Skill / Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| Settled Paired Samples | `{cal['sample_count']}` | `{cal['sample_count']}` | — |",
            f"| Brier Score (lower better) | `{_format_opt(cal['model_brier'])}` | `{_format_opt(cal['market_brier'])}` | BSS (`>0` beats mkt): `{_format_opt(cal['brier_skill_score'], '+.4f')}` |",
            f"| Binary Log-Loss (lower better) | `{_format_opt(cal['model_log_loss'])}` | `{_format_opt(cal['market_log_loss'])}` | Delta (`<0` beats mkt): `{_format_opt(cal['delta_log_loss'], '+.4f')}` |",
            f"| 10-Bin ECE | `{_format_opt(cal['model_ece'])}` | `{_format_opt(cal['market_ece'])}` | — |",
            "",
        ]

        if cal["reliability_buckets"]:
            lines.extend([
                "### Reliability Buckets (Model Prob vs. Realized Hit Rate)",
                "",
                "| Prob Bucket | Count | Mean Pred | Empirical Hit |",
                "| :--- | :---: | :---: | :---: |",
            ])
            for b in cal["reliability_buckets"]:
                lines.append(
                    f"| `{b['bucket']}` | `{b['count']}` | `{b['mean_pred']:.4f}` | `{b['empirical_hit']:.4f}` |"
                )
            lines.append("")

        lines.extend([
            "## 3. Execution Microstructure & Segment Breakdown",
            "",
            "### By Execution Mode",
            "",
            "| Mode | Orders | Fill Rate | Avg Slippage (¢) | Avg Fee/Contract (¢) | Mean CLV (bps) | Net P&L ($) | ROI (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        if metrics["by_execution_mode"]:
            for mode, row in metrics["by_execution_mode"].items():
                lines.append(
                    f"| `{mode}` | `{row['orders']}` | `{row['fill_rate_pct']:.1f}%` | `{_format_opt(row['avg_slippage_cents'], '+.2f')}` | `{_format_opt(row['avg_fee_per_contract_cents'], '.2f')}` | `{_format_opt(row['mean_clv_bps'], '+.1f')}` | `${row['net_pnl_usd']:+,.2f}` | `{row['roi_pct']:+.2f}%` |"
                )
        else:
            lines.append("| — | `0` | `0.0%` | `N/A` | `N/A` | `N/A` | `$+0.00` | `+0.00%` |")
        lines.append("")

        lines.extend([
            "### By Stat Category & Side",
            "",
            "| Stat | Side | Settled | Win Rate | Mean CLV (bps) | Model Brier | Market Brier | Net P&L ($) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        if metrics["by_stat_and_side"]:
            for r in metrics["by_stat_and_side"]:
                lines.append(
                    f"| `{r['stat_key']}` | `{r['side']}` | `{r['settled']}` | `{r['win_rate_pct']:.1f}%` | `{_format_opt(r['mean_clv_bps'], '+.1f')}` | `{_format_opt(r['model_brier'])}` | `{_format_opt(r['market_brier'])}` | `${r['net_pnl_usd']:+,.2f}` |"
                )
        else:
            lines.append("| — | — | `0` | `0.0%` | `N/A` | `N/A` | `N/A` | `$+0.00` |")
        lines.append("")

        lines.extend([
            "### By Model Version",
            "",
            "| Model Version | Settled | Win Rate | Mean CLV (bps) | Model Brier | Net P&L ($) | ROI (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        if metrics["by_model_version"]:
            for r in metrics["by_model_version"]:
                lines.append(
                    f"| `{r['model_version']}` | `{r['settled']}` | `{r['win_rate_pct']:.1f}%` | `{_format_opt(r['mean_clv_bps'], '+.1f')}` | `{_format_opt(r['model_brier'])}` | `${r['net_pnl_usd']:+,.2f}` | `{r['roi_pct']:+.2f}%` |"
                )
        else:
            lines.append("| — | `0` | `0.0%` | `N/A` | `N/A` | `$+0.00` | `+0.00%` |")
        lines.append("")

        lines.extend([
            "## 4. Risk Guardrails & Drift Health Snapshot",
            "",
            f"- **Drift Action:** `{gd['drift_action']}` | **Fallback Active:** `{gd['fallback_active']}` | **Rolling ECE:** `{_format_opt(gd['rolling_ece'])}` | **Max PSI:** `{_format_opt(gd['max_psi'])}`",
            f"- **Total Guardrail/API Rejections (`status=rejected`):** `{gd['total_rejected_orders']}`",
            "",
            "| Rejection Reason | Count |",
            "| :--- | :---: |",
        ])
        if gd["rejection_breakdown"]:
            for reason, cnt in sorted(gd["rejection_breakdown"].items()):
                lines.append(f"| `{reason}` | `{cnt}` |")
        else:
            lines.append("| *(none)* | `0` |")
        lines.append("")

        lines.extend([
            "## 5. Recent Settled Trade Ledger & Dry-Run Shadow Summary",
            "",
            "### Recent Settled Live/Sandbox Trades (Latest 25)",
            "",
            "| Settled UTC | Ticker | Stat | Side | Mode | Model Prob | Fill Px | Close Mid | CLV (bps) | Fees ($) | Hit | Realized P&L ($) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        if metrics["recent_settled_trades"]:
            for tr in metrics["recent_settled_trades"]:
                lines.append(
                    f"| `{str(tr['settled_at'])[:19]}` | `{tr['ticker']}` | `{tr['stat_key']}` | `{tr['side']}` | `{tr['execution_mode']}` | `{tr['model_prob']:.4f}` | `{tr['fill_price']:.4f}` | `{_format_opt(tr['closing_mid_price'])}` | `{_format_opt(tr['clv_bps'], '+.1f')}` | `${tr['fees_paid']:.2f}` | `{'WIN' if tr['hit'] else 'LOSS'}` | `${tr['realized_pnl']:+,.2f}` |"
                )
        else:
            lines.append("| — | *(no settled trades yet)* | — | — | — | — | — | — | — | — | — | — |")
        lines.append("")

        lines.extend([
            "### Dry-Run Shadow Orders (Isolated from Live P&L)",
            "",
            "| Total Dry-Run Orders | Settled Shadow Trades | Shadow Win Rate | Shadow Mean CLV (bps) | Shadow Model vs Market Brier | Counterfactual Net P&L ($) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| `{shadow['total_dry_run_orders']}` | `{shadow['settled_shadow_trades']}` | `{shadow['shadow_win_rate_pct']:.1f}%` | `{_format_opt(shadow['shadow_mean_clv_bps'], '+.1f')}` | `{_format_opt(shadow['shadow_model_brier'])}` vs `{_format_opt(shadow['shadow_market_brier'])}` | `${shadow['shadow_net_pnl_usd']:+,.2f}` |",
            "",
        ])
        return "\n".join(lines)

    def write_report(
        self,
        *,
        env: str = "demo",
        md_path: Optional[str] = "docs/sandbox_performance.md",
        json_path: Optional[str] = "benchmarks/sandbox_performance.json",
    ) -> Dict[str, Any]:
        metrics = self.build_metrics(env=env)
        markdown = self.render_markdown(metrics)
        if md_path:
            os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
            with open(md_path, "w") as f:
                f.write(markdown)
        if json_path:
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
        return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kalshi OMS Closed-Loop Performance Scorecard")
    parser.add_argument("--env", choices=["demo", "prod"], default="demo")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--output", default="docs/sandbox_performance.md")
    parser.add_argument("--json-out", default="benchmarks/sandbox_performance.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    monitor = DriftMonitor(db_url=args.db_url) if args.db_url else DriftMonitor()
    reporter = PerformanceReporter(monitor)
    summary = reporter.write_report(env=args.env, md_path=args.output, json_path=args.json_out)
    print(json.dumps(summary["topline"], indent=2))
