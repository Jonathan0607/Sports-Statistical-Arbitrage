"""ZINB-to-Kalshi Multi-Threshold Ladder Pricer and /v1/allocate portfolio sizer."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from scipy import stats

from src.feature_contract import FEATURE_SCHEMA_VERSION


def calculate_kalshi_fee(
    contracts: int,
    price: float,
    rate: float = 0.07,
    multiplier: float = 1.0,
) -> float:
    """Computes ceiling-rounded Kalshi contract fee in dollars.

    Formula: ceil(100 * multiplier * rate * contracts * price * (1 - price)) / 100.
    Supports both taker (rate=0.07) and passive maker (e.g. rate=0.0175 or 0.0).
    """
    c = int(contracts)
    p = float(price)
    r = float(rate)
    m = float(multiplier)
    if c <= 0 or r <= 0.0 or m <= 0.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    raw_cents = 100.0 * m * r * c * p * (1.0 - p)
    return float(round(math.ceil(max(0.0, raw_cents - 1e-12)) / 100.0, 2))


def zinb_survival_at_least(
    threshold: int,
    mu: float,
    alpha_disp: float,
    zero_inflation_prob: float = 0.0,
) -> float:
    """Computes exact ZINB discrete survival probability P(X >= threshold)."""
    k = int(threshold)
    if k <= 0:
        return 1.0
    n_nb = 1.0 / max(float(alpha_disp), 1e-6)
    p_nb = n_nb / (n_nb + max(float(mu), 1e-9))
    pi = float(np.clip(zero_inflation_prob, 0.0, 0.999))
    cutoff = k - 1
    nb_cdf = float(stats.nbinom.cdf(cutoff, n_nb, p_nb))
    cdf = pi + (1.0 - pi) * nb_cdf
    return float(np.clip(1.0 - cdf, 1e-6, 1.0 - 1e-6))


def parse_orderbook_quotes(book: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
    """Normalizes Kalshi market summary or L2 orderbook payload into YES/NO top-of-book floats."""
    now_ms = int(time.time() * 1000)
    ts_ms = int(book.get("quote_timestamp_ms") or book.get("timestamp_ms") or now_ms)

    def _f(val: Any) -> Optional[float]:
        if val is None or val == "":
            return None
        return float(val)

    yes_bid = _f(book.get("yes_bid_dollars")) or _f(book.get("best_yes_bid"))
    yes_ask = _f(book.get("yes_ask_dollars")) or _f(book.get("best_yes_ask"))
    no_bid = _f(book.get("no_bid_dollars")) or _f(book.get("best_no_bid"))
    no_ask = _f(book.get("no_ask_dollars")) or _f(book.get("best_no_ask"))

    ob = book.get("orderbook")
    if isinstance(ob, dict):
        yes_lvls = ob.get("yes") or []
        no_lvls = ob.get("no") or []
        if yes_bid is None and yes_lvls:
            raw_p = float(yes_lvls[0][0])
            yes_bid = raw_p / 100.0 if raw_p > 1.0 else raw_p
        if no_bid is None and no_lvls:
            raw_p = float(no_lvls[0][0])
            no_bid = raw_p / 100.0 if raw_p > 1.0 else raw_p

    if yes_bid is None and no_ask is not None:
        yes_bid = round(1.0 - no_ask, 4)
    if yes_ask is None and no_bid is not None:
        yes_ask = round(1.0 - no_bid, 4)
    if no_bid is None and yes_ask is not None:
        no_bid = round(1.0 - yes_ask, 4)
    if no_ask is None and yes_bid is not None:
        no_ask = round(1.0 - yes_bid, 4)

    return (
        float(yes_bid or 0.0),
        float(yes_ask or 1.0),
        float(no_bid or 0.0),
        float(no_ask or 1.0),
        ts_ms,
    )


@dataclass
class LadderRungCandidate:
    ticker: str
    player_id: str
    stat_key: str
    threshold: int
    side: str  # "bid" (buy YES) or "ask" (sell YES == buy NO at 1 - order_price)
    execution_mode: str  # "taker" or "maker"
    time_in_force: str  # "immediate_or_cancel" or "good_till_canceled"
    post_only: bool
    order_price: float  # Kalshi limit price on YES book
    effective_cost: float  # Cost per contract (order_price for YES, 1 - order_price for NO)
    unit_fee: float
    model_prob: float  # P(winning contract)
    yes_prob: float  # P(X >= threshold)
    ev_edge: float  # net EV per contract in dollars: p_model - (effective_cost + unit_fee)
    spread: float
    quote_timestamp_ms: int
    prediction_id: Optional[str] = None
    team_id: Optional[int] = None
    game_id: Optional[int] = None
    position: Optional[str] = None
    ewma_minutes: Optional[float] = None
    vacated_usage: Optional[float] = None
    opp_pace_factor: Optional[float] = None
    team_strength: Optional[float] = None
    recommended_count: int = 0
    allocated_dollars: float = 0.0


class LadderPricer:
    """Evaluates Kalshi multi-threshold prop ladders from ZINB parameters and sizes via /v1/allocate."""

    def __init__(
        self,
        *,
        min_edge_bps: float = 300.0,
        taker_fee_rate: float = 0.07,
        maker_fee_rate: float = 0.0175,
        fee_multiplier: float = 1.0,
        min_spread_for_maker: float = 0.02,
    ):
        self.min_edge_bps = float(min_edge_bps)
        self.min_edge_dollars = self.min_edge_bps / 10000.0
        self.taker_fee_rate = float(taker_fee_rate)
        self.maker_fee_rate = float(maker_fee_rate)
        self.fee_multiplier = float(fee_multiplier)
        self.min_spread_for_maker = float(min_spread_for_maker)

    def evaluate_rung(
        self,
        *,
        ticker: str,
        player_id: str,
        stat_key: str,
        threshold: int,
        mu: float,
        alpha_disp: float,
        zero_inflation_prob: float,
        orderbook: Dict[str, Any],
        prediction_id: Optional[str] = None,
        team_id: Optional[int] = None,
        game_id: Optional[int] = None,
        position: Optional[str] = None,
        ewma_minutes: Optional[float] = None,
        vacated_usage: Optional[float] = None,
        opp_pace_factor: Optional[float] = None,
        team_strength: Optional[float] = None,
    ) -> List[LadderRungCandidate]:
        """Scores YES ('bid') and NO ('ask') across taker and passive maker modes for a single rung."""
        yes_bid, yes_ask, no_bid, no_ask, ts_ms = parse_orderbook_quotes(orderbook)
        if yes_bid <= 0.0 or yes_ask >= 1.0 or yes_ask < yes_bid:
            return []

        spread = round(yes_ask - yes_bid, 4)
        p_yes = zinb_survival_at_least(threshold, mu, alpha_disp, zero_inflation_prob)
        p_no = float(np.clip(1.0 - p_yes, 1e-6, 1.0 - 1e-6))

        common = {
            "ticker": ticker,
            "player_id": str(player_id),
            "stat_key": str(stat_key),
            "threshold": int(threshold),
            "yes_prob": p_yes,
            "spread": spread,
            "quote_timestamp_ms": ts_ms,
            "prediction_id": prediction_id,
            "team_id": team_id,
            "game_id": game_id,
            "position": position,
            "ewma_minutes": ewma_minutes,
            "vacated_usage": vacated_usage,
            "opp_pace_factor": opp_pace_factor,
            "team_strength": team_strength,
        }

        candidates: List[LadderRungCandidate] = []

        # 1. YES Taker (bid @ best_yes_ask, IOC)
        fee_yes_t = calculate_kalshi_fee(1, yes_ask, rate=self.taker_fee_rate, multiplier=self.fee_multiplier)
        ev_yes_t = p_yes - (yes_ask + fee_yes_t)
        candidates.append(
            LadderRungCandidate(
                side="bid",
                execution_mode="taker",
                time_in_force="immediate_or_cancel",
                post_only=False,
                order_price=round(yes_ask, 4),
                effective_cost=round(yes_ask, 4),
                unit_fee=fee_yes_t,
                model_prob=p_yes,
                ev_edge=float(ev_yes_t),
                **common,
            )
        )

        # 2. YES Passive Maker (bid @ min(best_bid + 0.01, best_ask - 0.01), GTC post_only)
        if spread >= self.min_spread_for_maker - 1e-9:
            yes_maker = round(min(yes_bid + 0.01, yes_ask - 0.01), 4)
            if yes_maker < yes_ask - 1e-9 and yes_maker > 0.0:
                fee_yes_m = calculate_kalshi_fee(
                    1, yes_maker, rate=self.maker_fee_rate, multiplier=self.fee_multiplier
                )
                ev_yes_m = p_yes - (yes_maker + fee_yes_m)
                candidates.append(
                    LadderRungCandidate(
                        side="bid",
                        execution_mode="maker",
                        time_in_force="good_till_canceled",
                        post_only=True,
                        order_price=yes_maker,
                        effective_cost=yes_maker,
                        unit_fee=fee_yes_m,
                        model_prob=p_yes,
                        ev_edge=float(ev_yes_m),
                        **common,
                    )
                )

        # 3. NO Taker (ask @ best_yes_bid -> effective NO cost = 1 - best_yes_bid, IOC)
        no_taker_cost = round(1.0 - yes_bid, 4)
        if 0.0 < no_taker_cost < 1.0:
            fee_no_t = calculate_kalshi_fee(
                1, no_taker_cost, rate=self.taker_fee_rate, multiplier=self.fee_multiplier
            )
            ev_no_t = p_no - (no_taker_cost + fee_no_t)
            candidates.append(
                LadderRungCandidate(
                    side="ask",
                    execution_mode="taker",
                    time_in_force="immediate_or_cancel",
                    post_only=False,
                    order_price=round(yes_bid, 4),
                    effective_cost=no_taker_cost,
                    unit_fee=fee_no_t,
                    model_prob=p_no,
                    ev_edge=float(ev_no_t),
                    **common,
                )
            )

        # 4. NO Passive Maker (improved NO cost = min(no_bid + 0.01, no_ask - 0.01), order_price = 1 - no_maker)
        if spread >= self.min_spread_for_maker - 1e-9:
            no_maker_cost = round(min(no_bid + 0.01, no_ask - 0.01), 4)
            yes_ask_maker = round(1.0 - no_maker_cost, 4)
            if yes_ask_maker > yes_bid + 1e-9 and 0.0 < no_maker_cost < 1.0:
                fee_no_m = calculate_kalshi_fee(
                    1, no_maker_cost, rate=self.maker_fee_rate, multiplier=self.fee_multiplier
                )
                ev_no_m = p_no - (no_maker_cost + fee_no_m)
                candidates.append(
                    LadderRungCandidate(
                        side="ask",
                        execution_mode="maker",
                        time_in_force="good_till_canceled",
                        post_only=True,
                        order_price=yes_ask_maker,
                        effective_cost=no_maker_cost,
                        unit_fee=fee_no_m,
                        model_prob=p_no,
                        ev_edge=float(ev_no_m),
                        **common,
                    )
                )

        return candidates

    def select_optimal_rungs(
        self,
        *,
        player_id: str,
        stat_key: str,
        mu: float,
        alpha_disp: float,
        zero_inflation_prob: float,
        rungs: List[Dict[str, Any]],
        prediction_id: Optional[str] = None,
    ) -> List[LadderRungCandidate]:
        """Selects optimal non-dominated execution candidates exceeding min_edge_bps across ladder rungs."""
        best_per_rung: List[LadderRungCandidate] = []
        for rung in rungs:
            evaluated = self.evaluate_rung(
                ticker=str(rung["ticker"]),
                player_id=player_id,
                stat_key=stat_key,
                threshold=int(rung["threshold"]),
                mu=mu,
                alpha_disp=alpha_disp,
                zero_inflation_prob=zero_inflation_prob,
                orderbook=rung["orderbook"],
                prediction_id=prediction_id or rung.get("prediction_id"),
                team_id=rung.get("team_id"),
                game_id=rung.get("game_id"),
                position=rung.get("position"),
                ewma_minutes=rung.get("ewma_minutes"),
                vacated_usage=rung.get("vacated_usage"),
                opp_pace_factor=rung.get("opp_pace_factor"),
                team_strength=rung.get("team_strength"),
            )
            eligible = [c for c in evaluated if c.ev_edge >= self.min_edge_dollars]
            if not eligible:
                continue
            # Prefer highest net EV edge; tie-break toward passive maker to minimize spread crossing
            best = max(eligible, key=lambda c: (round(c.ev_edge, 6), 1 if c.execution_mode == "maker" else 0))
            best_per_rung.append(best)

        # Prune strictly dominated rungs within the same side (lower prob AND higher/equal total cost)
        non_dominated: List[LadderRungCandidate] = []
        for cand in sorted(best_per_rung, key=lambda c: c.ev_edge, reverse=True):
            c_total_cost = cand.effective_cost + cand.unit_fee
            dominated = False
            for kept in non_dominated:
                if kept.side == cand.side:
                    kept_cost = kept.effective_cost + kept.unit_fee
                    if kept.model_prob >= cand.model_prob + 1e-6 and kept_cost <= c_total_cost + 1e-6:
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(cand)
        return non_dominated

    async def size_with_allocator(
        self,
        candidates: List[LadderRungCandidate],
        *,
        inference_url: str = "http://localhost:8000",
        bankroll_usd: float = 1000.0,
        kelly_fraction: float = 0.25,
        max_bankroll_fraction: float = 0.05,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> List[LadderRungCandidate]:
        """Routes selected Kalshi ladder legs through POST /v1/allocate and sets contract counts."""
        if not candidates:
            return []

        req_payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "league": "NBA",
            "bankroll_usd": float(bankroll_usd),
            "kelly_fraction": float(kelly_fraction),
            "max_bankroll_fraction": float(max_bankroll_fraction),
            "candidates": [
                {
                    "leg_id": c.ticker,
                    "player_id": c.player_id,
                    "team_id": c.team_id,
                    "game_id": c.game_id,
                    "stat": c.stat_key,
                    "side": "Over" if c.side == "bid" else "Under",
                    "marginal_prob": float(np.clip(c.model_prob, 1e-4, 1.0 - 1e-4)),
                    "payout_odds": float(1.0 / max(c.effective_cost + c.unit_fee, 1e-4)),
                    "expected_edge": float(c.ev_edge / max(c.effective_cost + c.unit_fee, 1e-4)),
                    "line": float(c.threshold) - 0.5,
                    "position": c.position,
                    "ewma_minutes": c.ewma_minutes,
                    "vacated_usage": c.vacated_usage,
                    "opp_pace_factor": c.opp_pace_factor,
                    "team_strength": c.team_strength,
                }
                for c in candidates
            ],
        }

        if http_client is not None:
            resp = await http_client.post(f"{inference_url.rstrip('/')}/v1/allocate", json=req_payload)
            resp.raise_for_status()
            alloc_body = resp.json()
        else:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{inference_url.rstrip('/')}/v1/allocate", json=req_payload)
                resp.raise_for_status()
                alloc_body = resp.json()

        dollar_stakes = alloc_body.get("dollar_stakes", [])
        sized: List[LadderRungCandidate] = []
        for idx, cand in enumerate(candidates):
            dollars = float(dollar_stakes[idx]) if idx < len(dollar_stakes) else 0.0
            unit_outlay = cand.effective_cost + cand.unit_fee
            count = int(math.floor(max(0.0, dollars) / max(unit_outlay, 1e-4)))
            cand.allocated_dollars = dollars
            cand.recommended_count = count
            sized.append(cand)
        return sized
