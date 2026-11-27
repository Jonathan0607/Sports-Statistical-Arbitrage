"""Offline unit and integration tests for Phase 4B scorecard, CLV, pagination & DB migration."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
import sqlalchemy
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from src.drift_monitor import DriftMonitor, DriftThresholds
from src.oms.kalshi_client import KalshiClient
from src.oms.reconciler import SettlementReconciler, compute_clv_bps
from src.oms.report import PerformanceReporter


def _ed25519_pem() -> bytes:
    priv = ed25519.Ed25519PrivateKey.generate()
    return priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _insert_settled_order(
    reconciler: SettlementReconciler,
    *,
    cid: str,
    ticker: str,
    stat_key: str,
    side: str,
    execution_mode: str,
    limit_price: float,
    fill_price: float,
    count: int,
    fees: float,
    model_prob: float,
    market_mid_at_entry: float,
    closing_mid_price: float,
    yes_won: bool,
    settled_at: str,
    model_version: str = "champ_v1",
    env: str = "demo",
    expected_ev: float = 0.05,
) -> None:
    cost_per = limit_price if side == "bid" else (1.0 - limit_price)
    fill_cost = fill_price if side == "bid" else (1.0 - fill_price)
    payout_per = 1.0 if ((side == "bid" and yes_won) or (side == "ask" and not yes_won)) else 0.0
    pnl = (payout_per - fill_cost) * count - fees
    slip = fill_price - limit_price
    clv = compute_clv_bps(side, fill_price, closing_mid_price)

    reconciler.record_order(
        client_order_id=cid,
        ticker=ticker,
        player_id="1628983",
        stat_key=stat_key,
        threshold=25,
        side=side,
        execution_mode=execution_mode,
        limit_price=limit_price,
        effective_cost=cost_per,
        requested_count=count,
        model_prob=model_prob,
        expected_ev=expected_ev,
        env=env,
        model_version=model_version,
        best_bid_at_entry=round(market_mid_at_entry - 0.02, 4),
        best_ask_at_entry=round(market_mid_at_entry + 0.02, 4),
        market_mid_at_entry=market_mid_at_entry,
        closing_mid_price=closing_mid_price,
        clv_bps=clv,
        status="settled",
        created_at=settled_at,
    )
    with reconciler.engine.begin() as conn:
        conn.execute(
            reconciler.orders_table.update()
            .where(reconciler.orders_table.c.client_order_id == cid)
            .values(
                filled_count=count,
                avg_fill_price=fill_price,
                fees_paid=fees,
                slippage=slip,
                settled_at=settled_at,
                realized_pnl=pnl,
                clv_bps=clv,
            )
        )


# 1. Empty-DB Report Generation (no NaN / zero division, valid Markdown & JSON)
def test_empty_db_report_generation(tmp_path: Path):
    monitor = DriftMonitor(db_url="sqlite:///:memory:")
    reconciler = SettlementReconciler(monitor)
    reporter = PerformanceReporter(monitor, reconciler=reconciler)

    md_file = tmp_path / "sandbox_performance.md"
    json_file = tmp_path / "sandbox_performance.json"
    metrics = reporter.write_report(env="demo", md_path=str(md_file), json_path=str(json_file))

    assert metrics["confidence_banner"]["status"] == "WARM-UP / INSUFFICIENT SAMPLE"
    assert metrics["topline"]["settled_trades"] == 0
    assert metrics["topline"]["net_realized_pnl_usd"] == 0.0
    assert metrics["topline"]["net_roi_pct"] == 0.0
    assert metrics["topline"]["mean_clv_bps"] is None
    assert md_file.exists() and json_file.exists()
    md_text = md_file.read_text()
    assert "NaN" not in md_text
    assert "WARM-UP / INSUFFICIENT SAMPLE" in md_text
    parsed = json.loads(json_file.read_text())
    assert parsed["topline"]["settled_trades"] == 0


# 2. Exact CLV Sign Math for YES (bid) and NO (ask)
def test_clv_bps_sign_math_bid_and_ask():
    # Long YES at 0.40, market closes YES at 0.52 -> +1200 bps CLV
    assert compute_clv_bps("bid", 0.40, 0.52) == pytest.approx(1200.0)
    # Long YES at 0.55, closes at 0.50 -> -500 bps CLV
    assert compute_clv_bps("bid", 0.55, 0.50) == pytest.approx(-500.0)
    # Short YES / Long NO at YES=0.70 (NO cost 0.30), closes YES=0.60 (NO=0.40) -> +1000 bps CLV
    assert compute_clv_bps("ask", 0.70, 0.60) == pytest.approx(1000.0)
    # Short YES / Long NO at YES=0.60 (NO=0.40), closes YES=0.68 (NO=0.32) -> -800 bps CLV
    assert compute_clv_bps("ask", 0.60, 0.68) == pytest.approx(-800.0)


# 3. Model vs Market Calibration (BSS > 0, BSS < 0, Log-Loss Delta, 10-bin ECE)
def test_calibration_head_to_head_bss_positive_and_negative():
    # Sharp model beats market -> BSS > 0 and delta_log_loss < 0
    mon_good = DriftMonitor(db_url="sqlite:///:memory:")
    rec_good = SettlementReconciler(mon_good)
    for i in range(6):
        yes_win = i < 5
        _insert_settled_order(
            rec_good,
            cid=f"good-{i}",
            ticker=f"KX-G-{i}",
            stat_key="pts",
            side="bid",
            execution_mode="maker",
            limit_price=0.50,
            fill_price=0.50,
            count=2,
            fees=0.02,
            model_prob=0.85 if yes_win else 0.15,
            market_mid_at_entry=0.50,
            closing_mid_price=0.60 if yes_win else 0.40,
            yes_won=yes_win,
            settled_at=f"2026-11-26T0{i}:00:00+00:00",
        )
    cal_good = PerformanceReporter(mon_good, rec_good).build_metrics(env="demo")["calibration_head_to_head"]
    assert cal_good["model_brier"] < cal_good["market_brier"]
    assert cal_good["brier_skill_score"] > 0.5
    assert cal_good["delta_log_loss"] < 0.0
    assert cal_good["model_ece"] is not None

    # Miscalibrated model worse than market -> BSS < 0 and delta_log_loss > 0
    mon_bad = DriftMonitor(db_url="sqlite:///:memory:")
    rec_bad = SettlementReconciler(mon_bad)
    for i in range(4):
        yes_win = i == 0
        _insert_settled_order(
            rec_bad,
            cid=f"bad-{i}",
            ticker=f"KX-B-{i}",
            stat_key="reb",
            side="ask",
            execution_mode="taker",
            limit_price=0.45,
            fill_price=0.45,
            count=1,
            fees=0.02,
            model_prob=0.10 if not yes_win else 0.90,  # opposite of actual ask outcome!
            market_mid_at_entry=0.50,
            closing_mid_price=0.48,
            yes_won=yes_win,
            settled_at=f"2026-11-26T1{i}:00:00+00:00",
        )
    cal_bad = PerformanceReporter(mon_bad, rec_bad).build_metrics(env="demo")["calibration_head_to_head"]
    assert cal_bad["brier_skill_score"] < 0.0
    assert cal_bad["delta_log_loss"] > 0.0


# 4. Financial Metrics, Drawdown Sequence & Maker vs Taker Microstructure
def test_financial_metrics_max_drawdown_and_maker_vs_taker():
    mon = DriftMonitor(db_url="sqlite:///:memory:")
    rec = SettlementReconciler(mon)

    # Trade 1 (Taker WIN): ask @ YES 0.60 (cost 0.40) x 10, fill @ 0.59 -> +0.01 favorable slip?
    # Keep bid WIN first: bid @ 0.40 x 10 -> fill @ 0.41 (+1c adverse), fee 0.20 -> gross +5.90, net +5.70
    _insert_settled_order(
        rec,
        cid="seq-1",
        ticker="KX-SEQ-1",
        stat_key="pts",
        side="bid",
        execution_mode="taker",
        limit_price=0.40,
        fill_price=0.41,
        count=10,
        fees=0.20,
        model_prob=0.70,
        market_mid_at_entry=0.40,
        closing_mid_price=0.51,
        yes_won=True,
        settled_at="2026-11-26T01:00:00+00:00",
    )
    # Trade 2 (Maker LOSS): bid @ 0.50 x 6 -> fill @ 0.50, fee 0.05 -> gross -3.00, net -3.05 (cum=2.65, dd=3.05)
    _insert_settled_order(
        rec,
        cid="seq-2",
        ticker="KX-SEQ-2",
        stat_key="ast",
        side="bid",
        execution_mode="maker",
        limit_price=0.50,
        fill_price=0.50,
        count=6,
        fees=0.05,
        model_prob=0.60,
        market_mid_at_entry=0.50,
        closing_mid_price=0.47,
        yes_won=False,
        settled_at="2026-11-26T02:00:00+00:00",
    )
    # Trade 3 (Maker WIN): ask @ YES 0.70 (NO cost 0.30) x 10 -> fill @ 0.70, fee 0.10 -> net +6.90 (cum=9.55)
    _insert_settled_order(
        rec,
        cid="seq-3",
        ticker="KX-SEQ-3",
        stat_key="reb",
        side="ask",
        execution_mode="maker",
        limit_price=0.70,
        fill_price=0.70,
        count=10,
        fees=0.10,
        model_prob=0.75,
        market_mid_at_entry=0.69,
        closing_mid_price=0.60,
        yes_won=False,  # ask wins
        settled_at="2026-11-26T03:00:00+00:00",
    )
    # One unfilled resting maker order to verify fill rate = 2 / 3 (66.7%) for maker
    rec.record_order(
        client_order_id="seq-4-unfilled",
        ticker="KX-SEQ-4",
        player_id="1628983",
        stat_key="pts",
        threshold=25,
        side="bid",
        execution_mode="maker",
        limit_price=0.45,
        effective_cost=0.45,
        requested_count=4,
        model_prob=0.60,
        expected_ev=0.05,
        status="submitted",
        env="demo",
    )

    rep = PerformanceReporter(mon, rec).build_metrics(env="demo")
    top = rep["topline"]
    assert top["total_submitted_orders"] == 4
    assert top["filled_orders"] == 3
    assert top["fill_rate_pct"] == pytest.approx(75.0)
    assert top["settled_trades"] == 3
    # Capital risked: 4.10 + 3.00 + 3.00 = 10.10
    assert top["capital_risked_usd"] == pytest.approx(10.10)
    # Net PnL: 5.70 - 3.05 + 6.90 = +9.55; Fees = 0.35; Gross = +9.90
    assert top["total_fees_paid_usd"] == pytest.approx(0.35)
    assert top["net_realized_pnl_usd"] == pytest.approx(9.55)
    assert top["gross_pnl_usd"] == pytest.approx(9.90)
    assert top["net_roi_pct"] == pytest.approx(100.0 * 9.55 / 10.10)
    # Peak = 5.70 after trade 1; trough after trade 2 = 2.65 -> max DD = 3.05 (53.5088% of 5.70)
    assert top["max_drawdown_usd"] == pytest.approx(3.05)
    assert top["max_drawdown_pct"] == pytest.approx(100.0 * 3.05 / 5.70)

    modes = rep["by_execution_mode"]
    assert modes["taker"]["orders"] == 1
    assert modes["taker"]["fill_rate_pct"] == pytest.approx(100.0)
    assert modes["taker"]["avg_slippage_cents"] == pytest.approx(1.0)
    assert modes["maker"]["orders"] == 3
    assert modes["maker"]["fill_rate_pct"] == pytest.approx(200.0 / 3.0)


# 5. Strict Isolation of dry_run & rejected Orders (and env segmentation)
def test_strict_isolation_of_dry_run_rejected_and_prod_env():
    mon = DriftMonitor(db_url="sqlite:///:memory:")
    rec = SettlementReconciler(mon)

    # Guardrail-rejected order (must NOT affect live P&L/submitted counts, MUST appear in rejection breakdown)
    rec.record_order(
        client_order_id="rej-1",
        ticker="KX-REJ",
        player_id="1",
        stat_key="pts",
        threshold=20,
        side="bid",
        execution_mode="taker",
        limit_price=0.50,
        effective_cost=0.50,
        requested_count=5,
        model_prob=0.65,
        expected_ev=0.05,
        status="rejected",
        env="demo",
        rejection_reason="stale_quote:6200ms>5000ms",
    )

    # Settled dry_run shadow order (must NOT affect live P&L/ROI, MUST appear in dry_run_shadow)
    rec.record_order(
        client_order_id="dry-1",
        ticker="KX-DRY",
        player_id="1",
        stat_key="pts",
        threshold=25,
        side="bid",
        execution_mode="maker",
        limit_price=0.40,
        effective_cost=0.40,
        requested_count=5,
        model_prob=0.70,
        expected_ev=0.10,
        market_mid_at_entry=0.42,
        closing_mid_price=0.55,
        status="dry_run",
        env="demo",
    )
    with rec.engine.begin() as conn:
        conn.execute(
            rec.orders_table.update()
            .where(rec.orders_table.c.client_order_id == "dry-1")
            .values(settled_at="2026-11-26T05:00:00+00:00", realized_pnl=3.00)
        )

    # Prod order (must NOT leak into env='demo' report)
    _insert_settled_order(
        rec,
        cid="prod-1",
        ticker="KX-PROD",
        stat_key="pts",
        side="bid",
        execution_mode="taker",
        limit_price=0.40,
        fill_price=0.40,
        count=10,
        fees=0.10,
        model_prob=0.75,
        market_mid_at_entry=0.40,
        closing_mid_price=0.55,
        yes_won=True,
        settled_at="2026-11-26T06:00:00+00:00",
        env="prod",
    )

    demo_metrics = PerformanceReporter(mon, rec).build_metrics(env="demo")
    assert demo_metrics["topline"]["total_submitted_orders"] == 0
    assert demo_metrics["topline"]["settled_trades"] == 0
    assert demo_metrics["topline"]["net_realized_pnl_usd"] == 0.0
    assert demo_metrics["guardrails_and_drift"]["total_rejected_orders"] == 1
    assert demo_metrics["guardrails_and_drift"]["rejection_breakdown"]["stale_quote:6200ms>5000ms"] == 1
    assert demo_metrics["dry_run_shadow"]["total_dry_run_orders"] == 1
    assert demo_metrics["dry_run_shadow"]["settled_shadow_trades"] == 1
    assert demo_metrics["dry_run_shadow"]["shadow_net_pnl_usd"] == pytest.approx(3.00)


# 6. Cursor Pagination, capture_closing_lines() Preservation, Auto-Report Hook & SQLite Migration
def test_cursor_pagination_closing_line_capture_and_auto_report(tmp_path: Path):
    pem = _ed25519_pem()
    db_file = tmp_path / "legacy_oms.sqlite"

    # Simulate legacy oms_orders table WITHOUT Phase 4B columns to test ALTER TABLE migration
    legacy_engine = sqlalchemy.create_engine(f"sqlite:///{db_file}")
    with legacy_engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                """
                CREATE TABLE oms_orders (
                    client_order_id VARCHAR(64) PRIMARY KEY,
                    kalshi_order_id VARCHAR(64),
                    prediction_id VARCHAR(64),
                    ticker VARCHAR(128) NOT NULL,
                    player_id VARCHAR(64) NOT NULL,
                    stat_key VARCHAR(32) NOT NULL,
                    threshold INTEGER NOT NULL,
                    side VARCHAR(16) NOT NULL,
                    execution_mode VARCHAR(16) NOT NULL,
                    limit_price FLOAT NOT NULL,
                    effective_cost FLOAT NOT NULL,
                    requested_count INTEGER NOT NULL,
                    filled_count INTEGER NOT NULL DEFAULT 0,
                    avg_fill_price FLOAT,
                    fees_paid FLOAT NOT NULL DEFAULT 0.0,
                    slippage FLOAT,
                    model_prob FLOAT NOT NULL,
                    expected_ev FLOAT NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    created_at VARCHAR(64) NOT NULL,
                    settled_at VARCHAR(64),
                    realized_pnl FLOAT
                )
                """
            )
        )
    legacy_engine.dispose()

    md_out = tmp_path / "auto_scorecard.md"
    json_out = tmp_path / "auto_scorecard.json"
    monitor = DriftMonitor(
        db_url=f"sqlite:///{db_file}",
        thresholds=DriftThresholds(min_settled_samples=1),
    )
    reconciler = SettlementReconciler(
        monitor,
        auto_report_path=str(md_out),
        auto_json_report_path=str(json_out),
    )

    cols = {c["name"] for c in sqlalchemy.inspect(reconciler.engine).get_columns("oms_orders")}
    for expected_col in (
        "env",
        "model_version",
        "best_bid_at_entry",
        "best_ask_at_entry",
        "market_mid_at_entry",
        "closing_mid_price",
        "clv_bps",
        "rejection_reason",
    ):
        assert expected_col in cols

    pred_id = monitor.log_prediction(
        model_version="champ_v2",
        feature_schema_version="1.0.0",
        player_id="1628983",
        stat_key="pts",
        predicted_mean=28.0,
        ewma_baseline=24.0,
        predicted_prob=0.72,
        line=24.5,
    )
    reconciler.record_order(
        client_order_id="pag-1",
        kalshi_order_id="koid-1",
        prediction_id=pred_id,
        ticker="KX-CLOSE-1",
        player_id="1628983",
        stat_key="pts",
        threshold=25,
        side="bid",
        execution_mode="maker",
        limit_price=0.42,
        effective_cost=0.42,
        requested_count=5,
        model_prob=0.72,
        expected_ev=0.12,
        env="demo",
        model_version="champ_v2",
        best_bid_at_entry=0.40,
        best_ask_at_entry=0.46,
        market_mid_at_entry=0.43,
        status="submitted",
    )

    near_close_iso = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    market_phase = {"mid": 55, "status": "active", "result": None}
    settlement_pages = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal settlement_pages
        if request.url.path.endswith("/markets/KX-CLOSE-1"):
            return httpx.Response(
                200,
                json={
                    "market": {
                        "ticker": "KX-CLOSE-1",
                        "status": market_phase["status"],
                        "close_time": near_close_iso,
                        "yes_bid": market_phase["mid"] - 1,
                        "yes_ask": market_phase["mid"] + 1,
                        "last_price": market_phase["mid"],
                        "result": market_phase["result"],
                    }
                },
            )
        if request.url.path.endswith("/portfolio/fills"):
            cursor = request.url.params.get("cursor")
            if not cursor:
                return httpx.Response(
                    200,
                    json={
                        "fills": [
                            {
                                "fill_id": "f-page-1",
                                "order_id": "koid-1",
                                "client_order_id": "pag-1",
                                "ticker": "KX-CLOSE-1",
                                "side": "bid",
                                "count": 5,
                                "yes_price_dollars": "0.42",
                                "fee_paid": "0.05",
                            }
                        ],
                        "cursor": "cur-2",
                    },
                )
            return httpx.Response(200, json={"fills": [], "cursor": ""})

        if request.url.path.endswith("/portfolio/settlements"):
            settlement_pages += 1
            cursor = request.url.params.get("cursor")
            if not cursor:
                return httpx.Response(
                    200,
                    json={
                        "settlements": [{"ticker": "OTHER-UNRELATED", "market_result": "no"}],
                        "cursor": "set-page-2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "settlements": [
                        {
                            "ticker": "KX-CLOSE-1",
                            "market_result": "yes",
                            "settlement_value": 30.0,
                            "settled_time": "2026-11-26T04:00:00Z",
                        }
                    ],
                    "cursor": "",
                },
            )
        return httpx.Response(404)

    async def _run():
        async with KalshiClient(
            env="demo",
            api_key_id="k1",
            private_key_pem=pem,
            transport=httpx.MockTransport(handler),
        ) as kc:
            fill_res = await reconciler.sync_fills(kc)
            assert fill_res["fills_inserted"] == 1

            # First capture records pre-resolution midpoint 0.55 (+1300 bps CLV over 0.42 fill)
            clv_res1 = await reconciler.capture_closing_lines(kc)
            assert clv_res1["orders_updated"] == 1
            rec1 = reconciler.get_order("pag-1")
            assert rec1["closing_mid_price"] == pytest.approx(0.55)
            assert rec1["clv_bps"] == pytest.approx(1300.0)

            # Even if market later snaps to 0.99 after determination, closing_mid_price is NOT overwritten
            market_phase["mid"] = 99
            market_phase["status"] = "determined"
            market_phase["result"] = "yes"
            clv_res2 = await reconciler.capture_closing_lines(kc)
            assert clv_res2["orders_updated"] == 0
            rec2 = reconciler.get_order("pag-1")
            assert rec2["closing_mid_price"] == pytest.approx(0.55)

            # Paginated settlements reaches page 2, settles order, triggers auto report
            settle_res = await reconciler.sync_settlements(kc)
            assert settlement_pages == 2
            assert settle_res["orders_settled"] == 1
            assert settle_res["report_written"] == str(md_out)

    asyncio.run(_run())
    assert md_out.exists() and json_out.exists()
    report_data = json.loads(json_out.read_text())
    assert report_data["topline"]["settled_trades"] == 1
    assert report_data["topline"]["mean_clv_bps"] == pytest.approx(1300.0)
    assert "Kalshi Closed-Loop Performance Scorecard" in md_out.read_text()
