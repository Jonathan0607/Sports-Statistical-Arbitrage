"""Deterministic non-ML risk guardrails for Kalshi OMS order execution."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

from src.drift_monitor import DriftMonitor
from src.oms.kalshi_client import KalshiClient


class RiskDecision(NamedTuple):
    approved: bool
    reason: str
    adjusted_count: int


@dataclass
class RiskLimits:
    max_quote_age_ms: int = 5000
    min_price: float = 0.05
    max_price: float = 0.95
    max_spread: float = 0.08
    max_order_notional_usd: float = 5.00
    max_market_exposure_usd: float = 10.00
    max_total_exposure_usd: float = 50.00
    max_daily_loss_usd: float = 15.00
    kill_switch_file: str = ".oms_kill_switch"


class RiskGuardrails:
    """Enforces deterministic hard safety limits outside the ML/Copula pipeline."""

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        drift_monitor: Optional[DriftMonitor] = None,
    ):
        self.limits = limits or RiskLimits()
        self.drift_monitor = drift_monitor
        self.inference_failsafe_tripped: bool = False

    def is_kill_switch_active(self) -> bool:
        env_flag = os.getenv("OMS_KILL_SWITCH", "").strip().lower() in ("1", "true", "yes", "on")
        file_flag = bool(self.limits.kill_switch_file and os.path.exists(self.limits.kill_switch_file))
        return env_flag or file_flag

    def _is_drift_fallback_active(self, drift_report: Optional[Dict[str, Any]] = None) -> bool:
        if drift_report is not None:
            return bool(drift_report.get("fallback_active", False))
        if self.drift_monitor is not None:
            _, fallback = self.drift_monitor.current_status()
            return bool(fallback)
        return False

    def evaluate_order(
        self,
        *,
        requested_count: int,
        effective_cost: float,
        spread: float,
        quote_timestamp_ms: int,
        now_ms: Optional[int] = None,
        current_market_exposure_usd: float = 0.0,
        current_total_exposure_usd: float = 0.0,
        realized_daily_pnl_usd: float = 0.0,
        unrealized_daily_pnl_usd: float = 0.0,
        drift_report: Optional[Dict[str, Any]] = None,
    ) -> RiskDecision:
        """Evaluates all hard non-ML risk gates sequentially and caps contract count safely."""
        if self.is_kill_switch_active():
            return RiskDecision(False, "kill_switch_active", 0)

        if self.inference_failsafe_tripped:
            return RiskDecision(False, "inference_failsafe_active", 0)

        if self._is_drift_fallback_active(drift_report=drift_report):
            return RiskDecision(False, "drift_circuit_breaker_fallback_active", 0)

        current_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        quote_age_ms = current_ms - int(quote_timestamp_ms)
        if quote_age_ms > self.limits.max_quote_age_ms:
            return RiskDecision(
                False,
                f"stale_quote:{quote_age_ms}ms>{self.limits.max_quote_age_ms}ms",
                0,
            )

        cost = float(effective_cost)
        if cost < self.limits.min_price or cost > self.limits.max_price:
            return RiskDecision(
                False,
                f"price_out_of_bounds:{cost:.4f}_not_in_[{self.limits.min_price},{self.limits.max_price}]",
                0,
            )

        spr = float(spread)
        if spr < 0.0 or spr > self.limits.max_spread + 1e-9:
            return RiskDecision(
                False,
                f"spread_too_wide:{spr:.4f}>{self.limits.max_spread:.4f}",
                0,
            )

        daily_pnl = float(realized_daily_pnl_usd) + float(unrealized_daily_pnl_usd)
        if daily_pnl <= -float(self.limits.max_daily_loss_usd):
            return RiskDecision(
                False,
                f"daily_loss_cap_breached:{daily_pnl:.2f}<=-{self.limits.max_daily_loss_usd:.2f}",
                0,
            )

        rem_market = float(self.limits.max_market_exposure_usd) - float(current_market_exposure_usd)
        rem_total = float(self.limits.max_total_exposure_usd) - float(current_total_exposure_usd)
        allowed_notional = min(float(self.limits.max_order_notional_usd), rem_market, rem_total)
        if allowed_notional < cost - 1e-9:
            return RiskDecision(
                False,
                f"exposure_cap_exhausted:allowed={allowed_notional:.2f}<cost={cost:.4f}",
                0,
            )

        max_contracts = int(math.floor((allowed_notional + 1e-12) / max(cost, 1e-6)))
        adjusted = min(int(requested_count), max_contracts)
        if adjusted <= 0:
            return RiskDecision(False, "non_positive_adjusted_count", 0)

        return RiskDecision(True, "approved", adjusted)

    async def enforce_kill_switch_cancel_all(self, kalshi_client: KalshiClient) -> list[str]:
        if not self.is_kill_switch_active():
            return []
        return await kalshi_client.cancel_all_open_orders()

    async def handle_inference_failure(self, kalshi_client: KalshiClient) -> list[str]:
        self.inference_failsafe_tripped = True
        return await kalshi_client.cancel_all_open_orders()
