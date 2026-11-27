"""Offline unit and end-to-end integration tests for Phase 4 Kalshi OMS."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from src.drift_monitor import DriftMonitor, DriftThresholds
from src.feature_contract import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.oms.kalshi_client import DEMO_BASE_URL, PROD_BASE_URL, KalshiClient
from src.oms.ladder_pricer import LadderPricer, calculate_kalshi_fee
from src.oms.reconciler import SettlementReconciler
from src.oms.risk_guardrails import RiskGuardrails, RiskLimits
from src.oms.runner import parse_args, run_cycle


def _generate_ed25519_pem() -> tuple[ed25519.Ed25519PrivateKey, bytes]:
    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return priv, pem


def _generate_rsa_pem() -> tuple[rsa.RSAPrivateKey, bytes]:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return priv, pem


def _sample_feature_payload(player_id: str = "1628983") -> Dict[str, Any]:
    feats: Dict[str, Any] = {col: 1.0 for col in FEATURE_COLUMNS}
    feats.update(
        {
            "roll_min": 34.0,
            "roll_stat": 29.0,
            "roll_usg": 0.30,
            "days_rest": 2.0,
            "is_back_to_back": 0.0,
            "opp_def_rating": 114.0,
            "opp_pace_factor": 100.0,
            "opp_pos_pts_allowed": 23.0,
            "vacated_usage": 0.05,
            "is_home_game": 1.0,
            "pace_differential": 1.5,
            "roll_ts_pct": 0.62,
            "player_id": player_id,
            "team_id": 1610612760,
            "opponent_team_id": 1610612747,
            "player_tier": 0,
        }
    )
    return feats


class _StaticPredictor:
    """Deterministic predictor stub implementing FastAPI predictor methods over real schemas."""

    def __init__(self, drift_monitor: DriftMonitor, mu: float = 29.0):
        self.drift_monitor = drift_monitor
        self.mu = float(mu)
        self.model_version = "test_champion_v1"
        self.fallback_active = False

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_version": self.model_version,
            "git_sha": "test-sha",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "loaded_targets": ["points", "rebounds", "assists"],
            "redis_connected": False,
            "redis_status": "offline_test",
            "cpp_eigen_available": True,
            "drift_status": "HEALTHY" if not self.fallback_active else "FALLBACK_ACTIVE",
            "fallback_active": self.fallback_active,
            "registry_champion_id": self.model_version,
        }

    def drift_report(self, window_days: int = 7) -> Dict[str, Any]:
        rep = self.drift_monitor.evaluate_drift(window_days=window_days)
        fallback = bool(self.fallback_active or rep.get("fallback_active", False))
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "model_version": self.model_version,
            "git_sha": "test-sha",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window_days": window_days,
            "total_predictions": int(rep.get("total_predictions", 0)),
            "settled_predictions": int(rep.get("settled_predictions", 0)),
            "action": rep.get("action", "HEALTHY"),
            "fallback_active": fallback,
            "reasons": list(rep.get("reasons", [])),
            "calibration": dict(rep.get("calibration", {})),
            "covariate_drift": dict(rep.get("covariate_drift", {})),
            "thresholds": dict(rep.get("thresholds", {})),
        }

    def predict_batch(self, request_body: Any) -> Dict[str, Any]:
        preds: List[Dict[str, Any]] = []
        for item in request_body.requests:
            pid = str(item.player_id or (item.features.player_id if item.features else "0"))
            stat_val = item.stat.value if hasattr(item.stat, "value") else str(item.stat)
            line = float(item.thresholds[0]) if item.thresholds else 24.5
            prob_over = 0.72
            ewma_val = float(item.features.roll_stat) if item.features else self.mu
            pred_id = self.drift_monitor.log_prediction(
                model_version=self.model_version,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                player_id=pid,
                stat_key=stat_val,
                predicted_mean=self.mu,
                ewma_baseline=ewma_val,
                predicted_prob=prob_over,
                line=line,
            )
            preds.append(
                {
                    "prediction_id": pred_id,
                    "player_id": pid,
                    "stat": stat_val,
                    "league": item.league.value if hasattr(item.league, "value") else str(item.league),
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "predicted_mean": self.mu,
                    "residual": self.mu - ewma_val,
                    "xgb_residual": 0.0,
                    "lgb_residual": 0.0,
                    "ewma_baseline": ewma_val,
                    "zinb": {
                        "mu": self.mu,
                        "alpha_disp": 0.10,
                        "zero_inflation_prob": 0.01,
                        "n_nb": 10.0,
                    },
                    "survival_probabilities": [
                        {"threshold": t, "prob_over": prob_over, "prob_under": 1.0 - prob_over}
                        for t in item.thresholds
                    ],
                    "cold_start": False,
                    "fallback_active": self.fallback_active,
                }
            )
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "model_version": self.model_version,
            "predictions": preds,
            "latency_ms": 0.35,
            "fallback_active": self.fallback_active,
        }

    def allocate(self, req: Any) -> Dict[str, Any]:
        n = len(req.candidates)
        stakes = [2.50 for _ in range(n)]
        weights = [s / req.bankroll_usd for s in stakes]
        edges = [float(c.expected_edge or 0.0) for c in req.candidates]
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "league": req.league.value if hasattr(req.league, "value") else str(req.league),
            "allocations": weights,
            "dollar_stakes": stakes,
            "masked_stakes": [1 for _ in range(n)],
            "expected_edges": edges,
            "joint_pairs": [],
            "condition_number_raw": 1.0,
            "condition_number_shrunk": 1.0,
            "residual_norm": 0.0,
            "solve_residual_norm": 0.0,
            "solver_backend": "test_kelly",
            "is_rank_deficient": False,
            "numerical_rank": n,
            "copula_mode": "conditional_copula",
            "latency_ms": 0.25,
        }


# 1. Auth Signatures (Ed25519 + RSA-PSS SHA-256)
def test_kalshi_auth_signatures_ed25519_and_rsa_pss():
    ed_priv, ed_pem = _generate_ed25519_pem()
    rsa_priv, rsa_pem = _generate_rsa_pem()

    ed_client = KalshiClient(env="demo", api_key_id="ed-key-1", private_key_pem=ed_pem)
    assert ed_client.key_algorithm == "ed25519"
    assert ed_client.base_url == DEMO_BASE_URL

    ts = "1732570000123"
    headers = ed_client.build_auth_headers(
        "GET",
        "/portfolio/orders?ticker=KXNBAPTS-26NOV25LAL-25&status=resting",
        timestamp_ms=ts,
    )
    assert headers["KALSHI-ACCESS-KEY"] == "ed-key-1"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == ts
    expected_msg = f"{ts}GET/trade-api/v2/portfolio/orders".encode("utf-8")
    ed_priv.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), expected_msg)

    rsa_client = KalshiClient(env="demo", api_key_id="rsa-key-1", private_key_pem=rsa_pem)
    assert rsa_client.key_algorithm == "rsa_pss_sha256"
    rsa_headers = rsa_client.build_auth_headers(
        "post",
        "/trade-api/v2/portfolio/orders?ignored=1",
        timestamp_ms=ts,
    )
    expected_rsa_msg = f"{ts}POST/trade-api/v2/portfolio/orders".encode("utf-8")
    rsa_priv.public_key().verify(
        base64.b64decode(rsa_headers["KALSHI-ACCESS-SIGNATURE"]),
        expected_rsa_msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


# 2. Prod Safety Lock & CLI Safe Defaults
def test_kalshi_prod_safety_lock_and_runner_defaults():
    _, ed_pem = _generate_ed25519_pem()
    with pytest.raises(RuntimeError, match="live_armed=True"):
        KalshiClient(env="prod", api_key_id="k1", private_key_pem=ed_pem)

    with pytest.raises(RuntimeError, match="live_armed=True"):
        KalshiClient(env="prod", live_armed=False, api_key_id="k1", private_key_pem=ed_pem)

    armed = KalshiClient(env="prod", live_armed=True, api_key_id="k1", private_key_pem=ed_pem)
    assert armed.base_url == PROD_BASE_URL

    args = parse_args(["--once"])
    assert args.env == "demo"
    assert args.live_armed is False
    assert args.dry_run is True
    assert args.no_dry_run is False


# 3. Retry / Idempotency (409 Conflict + 429 Exponential Backoff) + V2 Order Write Routing
def test_kalshi_idempotent_409_and_429_retry():
    ed_priv, ed_pem = _generate_ed25519_pem()
    attempts = {"get": 0, "post": 0, "delete": 0}
    seen_cids: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/markets"):
            attempts["get"] += 1
            if attempts["get"] == 1:
                return httpx.Response(429, json={"error": "rate_limited"})
            return httpx.Response(200, json={"markets": [{"ticker": "KX-1"}]})

        if request.method == "POST" and request.url.path.endswith("/portfolio/events/orders"):
            attempts["post"] += 1
            ts = request.headers["KALSHI-ACCESS-TIMESTAMP"]
            sig = base64.b64decode(request.headers["KALSHI-ACCESS-SIGNATURE"])
            ed_priv.public_key().verify(
                sig,
                f"{ts}POST/trade-api/v2/portfolio/events/orders".encode("utf-8"),
            )
            payload = json.loads(request.content.decode("utf-8"))
            seen_cids.append(payload["client_order_id"])
            if attempts["post"] == 1:
                return httpx.Response(
                    201,
                    json={"order": {"order_id": "kord-1", "client_order_id": payload["client_order_id"]}},
                )
            return httpx.Response(409, json={"error": "duplicate client_order_id"})

        if request.method == "DELETE" and request.url.path.endswith("/portfolio/events/orders/kord-1"):
            attempts["delete"] += 1
            assert request.url.params.get("market_ticker") == "KX-1"
            ts = request.headers["KALSHI-ACCESS-TIMESTAMP"]
            sig = base64.b64decode(request.headers["KALSHI-ACCESS-SIGNATURE"])
            # Signature must strip ?market_ticker=KX-1 query parameter
            ed_priv.public_key().verify(
                sig,
                f"{ts}DELETE/trade-api/v2/portfolio/events/orders/kord-1".encode("utf-8"),
            )
            return httpx.Response(200, json={"order": {"order_id": "kord-1", "status": "canceled"}})

        return httpx.Response(404)

    async def _run():
        async with KalshiClient(
            env="demo",
            api_key_id="k1",
            private_key_pem=ed_pem,
            transport=httpx.MockTransport(handler),
            backoff_base_sec=0.001,
            max_retries=2,
        ) as client:
            markets = await client.get_markets(limit=10)
            assert len(markets["markets"]) == 1
            assert attempts["get"] == 2

            cid = "fixed-cid-123"
            first = await client.create_order(
                ticker="KX-1",
                side="bid",
                count=2,
                price=0.45,
                client_order_id=cid,
            )
            assert first["order"]["order_id"] == "kord-1"

            dup = await client.create_order(
                ticker="KX-1",
                side="bid",
                count=2,
                price=0.45,
                client_order_id=cid,
            )
            assert dup["status"] == "duplicate_conflict"
            assert dup["idempotent_already_submitted"] is True
            assert seen_cids == [cid, cid]

            cancel_resp = await client.cancel_order("kord-1", market_ticker="KX-1")
            assert cancel_resp["order"]["status"] == "canceled"
            assert attempts["delete"] == 1

    asyncio.run(_run())


# 4. Ladder Pricing & Fee Math
def test_ladder_pricing_fees_and_allocator_sizing():
    # Ceiling-rounded Kalshi fee: ceil(100 * 0.07 * 1 * 0.5 * 0.5) / 100 = ceil(1.75)/100 = 0.02
    assert calculate_kalshi_fee(1, 0.50, rate=0.07, multiplier=1.0) == pytest.approx(0.02)
    # 10 contracts at 0.50: ceil(100 * 0.07 * 10 * 0.25) / 100 = ceil(17.5)/100 = 0.18
    assert calculate_kalshi_fee(10, 0.50, rate=0.07, multiplier=1.0) == pytest.approx(0.18)
    assert calculate_kalshi_fee(1, 0.0, rate=0.07) == 0.0

    pricer = LadderPricer(min_edge_bps=300.0)
    now_ms = int(time.time() * 1000)

    rungs = [
        {
            "ticker": "KXNBAPTS-25",
            "threshold": 25,
            "orderbook": {
                "yes_bid_dollars": "0.40",
                "yes_ask_dollars": "0.46",
                "timestamp_ms": now_ms,
            },
        },
        {
            "ticker": "KXNBAPTS-20",
            "threshold": 20,
            "orderbook": {
                "yes_bid_dollars": "0.77",
                "yes_ask_dollars": "0.79",
                "timestamp_ms": now_ms,
            },
        },
        {
            "ticker": "KXNBAPTS-35",
            "threshold": 35,
            "orderbook": {
                "yes_bid_dollars": "0.70",
                "yes_ask_dollars": "0.72",
                "timestamp_ms": now_ms,
            },
        },
    ]

    selected = pricer.select_optimal_rungs(
        player_id="1628983",
        stat_key="pts",
        mu=28.0,
        alpha_disp=0.10,
        zero_inflation_prob=0.01,
        rungs=rungs,
        prediction_id="pred-1",
    )
    tickers = {c.ticker: c for c in selected}
    assert "KXNBAPTS-20" not in tickers
    assert tickers["KXNBAPTS-25"].side == "bid"
    assert tickers["KXNBAPTS-25"].execution_mode == "maker"
    assert tickers["KXNBAPTS-25"].time_in_force == "good_till_canceled"
    assert tickers["KXNBAPTS-25"].post_only is True
    assert tickers["KXNBAPTS-25"].order_price == pytest.approx(0.41)

    # 35+ priced at 0.70 bid -> NO cost is 0.30, EV is strongly positive on side='ask'
    assert tickers["KXNBAPTS-35"].side == "ask"
    assert tickers["KXNBAPTS-35"].order_price == pytest.approx(0.71)  # 1 - no_maker_cost (0.29)
    assert tickers["KXNBAPTS-35"].effective_cost == pytest.approx(0.29)
    assert tickers["KXNBAPTS-35"].execution_mode == "maker"

    # Also verify YES taker mode executes when spread < min_spread_for_maker (tight 1c spread)
    tight_cands = pricer.evaluate_rung(
        ticker="KXNBAPTS-25T",
        player_id="1628983",
        stat_key="pts",
        threshold=25,
        mu=28.0,
        alpha_disp=0.10,
        zero_inflation_prob=0.01,
        orderbook={"yes_bid_dollars": "0.45", "yes_ask_dollars": "0.46", "timestamp_ms": now_ms},
    )
    yes_takers = [c for c in tight_cands if c.side == "bid" and c.execution_mode == "taker"]
    assert len(yes_takers) == 1
    assert yes_takers[0].time_in_force == "immediate_or_cancel"
    assert yes_takers[0].post_only is False

    from src.serving.app import create_app

    monitor = DriftMonitor(db_url="sqlite:///:memory:")
    stub = _StaticPredictor(monitor, mu=28.0)
    app = create_app(predictor=stub)

    async def _alloc():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api_client:
            return await pricer.size_with_allocator(
                selected,
                inference_url="http://test",
                bankroll_usd=1000.0,
                http_client=api_client,
            )

    sized = asyncio.run(_alloc())
    assert all(c.allocated_dollars == pytest.approx(2.50) for c in sized)
    assert tickers["KXNBAPTS-25"].recommended_count == math.floor(
        2.50 / (tickers["KXNBAPTS-25"].effective_cost + tickers["KXNBAPTS-25"].unit_fee)
    )


# 5. All 6 Deterministic Risk Guardrails + Inference Outage Cancel-All
def test_risk_guardrails_all_six_gates_and_outage_failsafe(tmp_path: Path, monkeypatch):
    _, ed_pem = _generate_ed25519_pem()
    cancelled_calls: List[str] = []
    seen_requests: List[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((request.method, request.url.path))
        if request.method == "DELETE" and request.url.path.endswith("/portfolio/events/orders"):
            cancelled_calls.extend(["open-1", "open-2"])
            return httpx.Response(200, json={"orders": [{"order_id": "open-1"}, {"order_id": "open-2"}]})
        return httpx.Response(500)

    kill_file = tmp_path / ".oms_kill_switch"
    limits = RiskLimits(
        max_quote_age_ms=5000,
        min_price=0.05,
        max_price=0.95,
        max_spread=0.08,
        max_order_notional_usd=5.00,
        max_market_exposure_usd=10.00,
        max_total_exposure_usd=50.00,
        max_daily_loss_usd=15.00,
        kill_switch_file=str(kill_file),
    )
    monitor = DriftMonitor(db_url="sqlite:///:memory:")
    guardrails = RiskGuardrails(limits=limits, drift_monitor=monitor)
    now_ms = int(time.time() * 1000)

    # Gate 1: Kill switch file + env var + cancel-all sweep
    kill_file.write_text("halt")
    d1 = guardrails.evaluate_order(
        requested_count=5, effective_cost=0.45, spread=0.02, quote_timestamp_ms=now_ms, now_ms=now_ms
    )
    assert not d1.approved and d1.reason == "kill_switch_active"
    kill_file.unlink()

    monkeypatch.setenv("OMS_KILL_SWITCH", "true")
    assert guardrails.is_kill_switch_active()

    async def _kill_sweep():
        async with KalshiClient(
            env="demo", api_key_id="k1", private_key_pem=ed_pem, transport=httpx.MockTransport(handler)
        ) as kc:
            return await guardrails.enforce_kill_switch_cancel_all(kc)

    assert asyncio.run(_kill_sweep()) == ["open-1", "open-2"]
    assert seen_requests == [("DELETE", "/trade-api/v2/portfolio/events/orders")]
    seen_requests.clear()
    monkeypatch.delenv("OMS_KILL_SWITCH")

    # Gate 2: Drift circuit-breaker fallback_active=True
    d2 = guardrails.evaluate_order(
        requested_count=5,
        effective_cost=0.45,
        spread=0.02,
        quote_timestamp_ms=now_ms,
        now_ms=now_ms,
        drift_report={"fallback_active": True},
    )
    assert not d2.approved and "drift_circuit_breaker" in d2.reason

    # Gate 3: Stale quote (>5000ms)
    d3 = guardrails.evaluate_order(
        requested_count=5,
        effective_cost=0.45,
        spread=0.02,
        quote_timestamp_ms=now_ms - 5500,
        now_ms=now_ms,
    )
    assert not d3.approved and d3.reason.startswith("stale_quote:")

    # Gate 4: Price out of [0.05, 0.95] & spread > 0.08
    d4a = guardrails.evaluate_order(
        requested_count=5, effective_cost=0.04, spread=0.02, quote_timestamp_ms=now_ms, now_ms=now_ms
    )
    assert not d4a.approved and d4a.reason.startswith("price_out_of_bounds:")

    d4b = guardrails.evaluate_order(
        requested_count=5, effective_cost=0.45, spread=0.10, quote_timestamp_ms=now_ms, now_ms=now_ms
    )
    assert not d4b.approved and d4b.reason.startswith("spread_too_wide:")

    # Gate 5: Per-order ($5), per-market ($10), total ($50), and daily loss ($15) caps
    d5_cap = guardrails.evaluate_order(
        requested_count=20, effective_cost=0.50, spread=0.02, quote_timestamp_ms=now_ms, now_ms=now_ms
    )
    assert d5_cap.approved and d5_cap.adjusted_count == 10

    d5_mkt = guardrails.evaluate_order(
        requested_count=10,
        effective_cost=0.50,
        spread=0.02,
        quote_timestamp_ms=now_ms,
        now_ms=now_ms,
        current_market_exposure_usd=9.80,
    )
    assert not d5_mkt.approved and "exposure_cap_exhausted" in d5_mkt.reason

    d5_tot = guardrails.evaluate_order(
        requested_count=10,
        effective_cost=0.50,
        spread=0.02,
        quote_timestamp_ms=now_ms,
        now_ms=now_ms,
        current_total_exposure_usd=49.80,
    )
    assert not d5_tot.approved and "exposure_cap_exhausted" in d5_tot.reason

    d5_loss = guardrails.evaluate_order(
        requested_count=5,
        effective_cost=0.50,
        spread=0.02,
        quote_timestamp_ms=now_ms,
        now_ms=now_ms,
        realized_daily_pnl_usd=-15.00,
    )
    assert not d5_loss.approved and "daily_loss_cap_breached" in d5_loss.reason

    # Gate 6: Inference outage failsafe cancels open orders & blocks subsequent submissions
    async def _outage():
        broken_api = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(503, json={"error": "down"})),
            base_url="http://broken-api",
        )
        cancelled_calls.clear()
        async with KalshiClient(
            env="demo", api_key_id="k1", private_key_pem=ed_pem, transport=httpx.MockTransport(handler)
        ) as kc:
            res = await run_cycle(
                kalshi_client=kc,
                pricer=LadderPricer(),
                guardrails=guardrails,
                reconciler=SettlementReconciler(monitor),
                inference_url="http://broken-api",
                predict_requests=[],
                market_rungs_by_player_stat={},
                dry_run=False,
                inference_http_client=broken_api,
            )
            assert res["aborted_reason"].startswith("inference_drift_outage")
            assert res["cancelled_on_failsafe"] == ["open-1", "open-2"]
            assert seen_requests == [("DELETE", "/trade-api/v2/portfolio/events/orders")]
            assert guardrails.inference_failsafe_tripped is True
        await broken_api.aclose()

    asyncio.run(_outage())


# 6. End-to-End Closed-Loop: FastAPI /v1/predict + /v1/allocate -> Kalshi -> Fills/Settlements -> DriftMonitor
def test_end_to_end_closed_loop_kalshi_settlement_into_drift_monitor(tmp_path: Path):
    from src.serving.app import create_app

    _, ed_pem = _generate_ed25519_pem()
    monitor = DriftMonitor(
        db_url="sqlite:///:memory:",
        thresholds=DriftThresholds(min_settled_samples=1),
    )
    stub = _StaticPredictor(monitor, mu=29.0)
    app = create_app(predictor=stub)

    submitted_orders: List[Dict[str, Any]] = []
    fills_store: List[Dict[str, Any]] = []
    settlements_store: List[Dict[str, Any]] = []

    def kalshi_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/portfolio/events/orders"):
            body = json.loads(request.content.decode("utf-8"))
            order_id = f"kalshi-ord-{len(submitted_orders) + 1}"
            rec = {
                "order_id": order_id,
                "client_order_id": body["client_order_id"],
                "ticker": body["ticker"],
                "side": body["side"],
                "count": int(body["count"]),
                "price": float(
                    body.get("price") or body.get("yes_price_dollars") or body.get("no_price_dollars")
                ),
            }
            submitted_orders.append(rec)
            # Simulate immediate fill with +0.01 slippage relative to limit order price
            fills_store.append(
                {
                    "fill_id": f"fill-{len(submitted_orders)}",
                    "order_id": order_id,
                    "client_order_id": body["client_order_id"],
                    "ticker": body["ticker"],
                    "side": body["side"],
                    "count": int(body["count"]),
                    "yes_price_dollars": f"{rec['price'] + 0.01:.2f}",
                    "fee_paid": "0.05",
                    "created_time": "2026-11-26T01:00:00Z",
                }
            )
            return httpx.Response(201, json={"order": rec})

        if request.method == "GET" and request.url.path.endswith("/portfolio/fills"):
            return httpx.Response(200, json={"fills": fills_store})

        if request.method == "GET" and request.url.path.endswith("/portfolio/settlements"):
            return httpx.Response(200, json={"settlements": settlements_store})

        return httpx.Response(200, json={"orders": []})

    now_ms = int(time.time() * 1000)
    predict_requests = [
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "league": "NBA",
            "stat": "pts",
            "player_id": "1628983",
            "thresholds": [24.5],
            "features": _sample_feature_payload("1628983"),
        }
    ]
    rungs_map = {
        ("1628983", "pts"): [
            {
                "ticker": "KXNBAPTS-26NOV25-25",
                "threshold": 25,
                "team_id": 1610612760,
                "game_id": 22600100,
                "position": "G",
                "ewma_minutes": 34.0,
                "orderbook": {
                    "yes_bid_dollars": "0.40",
                    "yes_ask_dollars": "0.46",
                    "timestamp_ms": now_ms,
                },
            }
        ]
    }

    reconciler = SettlementReconciler(monitor)
    guardrails = RiskGuardrails(
        limits=RiskLimits(kill_switch_file=str(tmp_path / ".oms_kill_switch")),
        drift_monitor=monitor,
    )
    pricer = LadderPricer(min_edge_bps=300.0)

    async def _e2e():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api_client:
            async with KalshiClient(
                env="demo",
                api_key_id="k1",
                private_key_pem=ed_pem,
                transport=httpx.MockTransport(kalshi_handler),
            ) as kc:
                # Dry-run default does NOT hit POST /portfolio/orders
                dry_res = await run_cycle(
                    kalshi_client=kc,
                    pricer=pricer,
                    guardrails=guardrails,
                    reconciler=reconciler,
                    inference_url="http://test",
                    predict_requests=predict_requests,
                    market_rungs_by_player_stat=rungs_map,
                    bankroll_usd=1000.0,
                    dry_run=True,
                    inference_http_client=api_client,
                )
                assert len(dry_res["orders_submitted"]) == 1
                assert dry_res["orders_submitted"][0]["dry_run"] is True
                assert len(submitted_orders) == 0

                # Live sandbox submission (--no-dry-run equivalent)
                live_res = await run_cycle(
                    kalshi_client=kc,
                    pricer=pricer,
                    guardrails=guardrails,
                    reconciler=reconciler,
                    inference_url="http://test",
                    predict_requests=predict_requests,
                    market_rungs_by_player_stat=rungs_map,
                    bankroll_usd=1000.0,
                    dry_run=False,
                    inference_http_client=api_client,
                )
                assert len(live_res["orders_submitted"]) == 1
                cid = live_res["orders_submitted"][0]["client_order_id"]
                assert len(submitted_orders) == 1
                assert live_res["fills_sync"]["fills_inserted"] == 1

                order_rec = reconciler.get_order(cid)
                assert order_rec is not None
                assert order_rec["status"] == "filled"
                assert order_rec["filled_count"] == submitted_orders[0]["count"]
                assert order_rec["slippage"] == pytest.approx(0.01)

                # Simulate market settling YES (player scored 31 pts >= 25)
                settlements_store.append(
                    {
                        "ticker": "KXNBAPTS-26NOV25-25",
                        "market_result": "yes",
                        "settlement_value": 31.0,
                        "settled_time": "2026-11-26T04:00:00Z",
                    }
                )
                settle_res = await reconciler.sync_settlements(kc)
                assert settle_res["orders_settled"] == 1
                assert settle_res["drift_telemetry_settled"] == 1

                settled_rec = reconciler.get_order(cid)
                assert settled_rec["status"] == "settled"
                cnt = int(settled_rec["filled_count"])
                expected_pnl = (1.0 - float(settled_rec["avg_fill_price"])) * cnt - 0.05
                assert settled_rec["realized_pnl"] == pytest.approx(expected_pnl)

                drift_report = monitor.evaluate_drift()
                assert drift_report["settled_predictions"] == 1
                # prob_over=0.72, hit=True -> Brier = (0.72 - 1.0)^2 = 0.0784
                assert drift_report["calibration"]["brier_score"] == pytest.approx(
                    (0.72 - 1.0) ** 2, rel=1e-4
                )
                assert drift_report["calibration"]["ece"] is not None

    asyncio.run(_e2e())
