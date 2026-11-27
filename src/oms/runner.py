"""Sandbox-first CLI execution runner for Kalshi ZINB multi-threshold ladders."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

from src.drift_monitor import DriftMonitor
from src.feature_contract import FEATURE_SCHEMA_VERSION
from src.oms.kalshi_client import KalshiAPIError, KalshiClient
from src.oms.ladder_pricer import LadderPricer
from src.oms.reconciler import SettlementReconciler
from src.oms.risk_guardrails import RiskGuardrails, RiskLimits

logger = logging.getLogger("KalshiOMSRunner")


async def run_cycle(
    *,
    kalshi_client: KalshiClient,
    pricer: LadderPricer,
    guardrails: RiskGuardrails,
    reconciler: SettlementReconciler,
    inference_url: str = "http://localhost:8000",
    predict_requests: Optional[List[Dict[str, Any]]] = None,
    market_rungs_by_player_stat: Optional[Dict[tuple, List[Dict[str, Any]]]] = None,
    bankroll_usd: float = 1000.0,
    dry_run: bool = True,
    inference_http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Executes one complete inference -> ladder pricing -> Kelly allocation -> risk -> OMS cycle."""
    summary: Dict[str, Any] = {
        "dry_run": bool(dry_run),
        "orders_submitted": [],
        "orders_blocked": [],
        "cancelled_on_failsafe": [],
        "fills_sync": {},
        "settlements_sync": {},
    }

    if guardrails.is_kill_switch_active():
        if not dry_run:
            summary["cancelled_on_failsafe"] = await guardrails.enforce_kill_switch_cancel_all(
                kalshi_client
            )
        summary["aborted_reason"] = "kill_switch_active"
        return summary

    owns_client = inference_http_client is None
    client = inference_http_client or httpx.AsyncClient(timeout=5.0)
    try:
        try:
            drift_resp = await client.get(f"{inference_url.rstrip('/')}/v1/drift")
            drift_resp.raise_for_status()
            drift_report = drift_resp.json()
        except Exception as exc:
            logger.error(f"Inference /v1/drift outage ({exc}); cancelling open orders={not dry_run}.")
            if not dry_run:
                summary["cancelled_on_failsafe"] = await guardrails.handle_inference_failure(
                    kalshi_client
                )
            summary["aborted_reason"] = f"inference_drift_outage:{exc}"
            return summary

        if drift_report.get("fallback_active"):
            summary["aborted_reason"] = "drift_circuit_breaker_fallback_active"
            return summary

        if not predict_requests or not market_rungs_by_player_stat:
            summary["fills_sync"] = await reconciler.sync_fills(kalshi_client)
            summary["closing_lines_sync"] = await reconciler.capture_closing_lines(kalshi_client)
            summary["shadow_settlements_sync"] = await reconciler.sync_shadow_settlements(kalshi_client)
            summary["settlements_sync"] = await reconciler.sync_settlements(kalshi_client)
            return summary

        try:
            pred_resp = await client.post(
                f"{inference_url.rstrip('/')}/v1/predict",
                json={"schema_version": FEATURE_SCHEMA_VERSION, "requests": predict_requests},
            )
            pred_resp.raise_for_status()
            pred_body = pred_resp.json()
        except Exception as exc:
            logger.error(f"Inference /v1/predict failure ({exc}); cancelling open orders={not dry_run}.")
            if not dry_run:
                summary["cancelled_on_failsafe"] = await guardrails.handle_inference_failure(
                    kalshi_client
                )
            summary["aborted_reason"] = f"inference_predict_outage:{exc}"
            return summary

        model_version = str(pred_body.get("model_version") or "") or None
        all_candidates = []
        for pred in pred_body.get("predictions", []):
            key = (str(pred["player_id"]), str(pred["stat"]))
            rungs = market_rungs_by_player_stat.get(key, [])
            if not rungs:
                continue
            zinb = pred["zinb"]
            selected = pricer.select_optimal_rungs(
                player_id=str(pred["player_id"]),
                stat_key=str(pred["stat"]),
                mu=float(zinb["mu"]),
                alpha_disp=float(zinb["alpha_disp"]),
                zero_inflation_prob=float(zinb["zero_inflation_prob"]),
                rungs=rungs,
                prediction_id=pred.get("prediction_id"),
            )
            all_candidates.extend(selected)

        if all_candidates:
            try:
                sized_candidates = await pricer.size_with_allocator(
                    all_candidates,
                    inference_url=inference_url,
                    bankroll_usd=bankroll_usd,
                    http_client=client,
                )
            except Exception as exc:
                logger.error(f"Inference /v1/allocate failure ({exc}); cancelling open orders={not dry_run}.")
                if not dry_run:
                    summary["cancelled_on_failsafe"] = await guardrails.handle_inference_failure(
                        kalshi_client
                    )
                summary["aborted_reason"] = f"inference_allocate_outage:{exc}"
                return summary

            exposure = reconciler.get_exposure_snapshot()
            total_exp = float(exposure["total_exposure_usd"])
            realized_pnl = float(exposure["realized_daily_pnl_usd"])
            by_ticker = dict(exposure["by_ticker"])

            for cand in sized_candidates:
                req_cnt = max(1, int(cand.recommended_count))
                decision = guardrails.evaluate_order(
                    requested_count=req_cnt,
                    effective_cost=cand.effective_cost,
                    spread=cand.spread,
                    quote_timestamp_ms=cand.quote_timestamp_ms,
                    current_market_exposure_usd=by_ticker.get(cand.ticker, 0.0),
                    current_total_exposure_usd=total_exp,
                    realized_daily_pnl_usd=realized_pnl,
                    drift_report=drift_report,
                )
                if not decision.approved:
                    rej_cid = str(uuid.uuid4())
                    reconciler.record_order(
                        client_order_id=rej_cid,
                        ticker=cand.ticker,
                        player_id=cand.player_id,
                        stat_key=cand.stat_key,
                        threshold=cand.threshold,
                        side=cand.side,
                        execution_mode=cand.execution_mode,
                        limit_price=cand.order_price,
                        effective_cost=cand.effective_cost,
                        requested_count=req_cnt,
                        model_prob=cand.model_prob,
                        expected_ev=cand.ev_edge,
                        prediction_id=cand.prediction_id,
                        status="rejected",
                        env=kalshi_client.env,
                        model_version=model_version,
                        best_bid_at_entry=cand.yes_bid,
                        best_ask_at_entry=cand.yes_ask,
                        market_mid_at_entry=cand.market_mid,
                        rejection_reason=decision.reason,
                    )
                    summary["orders_blocked"].append({"ticker": cand.ticker, "reason": decision.reason})
                    continue

                cid = str(uuid.uuid4())
                if dry_run:
                    reconciler.record_order(
                        client_order_id=cid,
                        ticker=cand.ticker,
                        player_id=cand.player_id,
                        stat_key=cand.stat_key,
                        threshold=cand.threshold,
                        side=cand.side,
                        execution_mode=cand.execution_mode,
                        limit_price=cand.order_price,
                        effective_cost=cand.effective_cost,
                        requested_count=decision.adjusted_count,
                        model_prob=cand.model_prob,
                        expected_ev=cand.ev_edge,
                        prediction_id=cand.prediction_id,
                        status="dry_run",
                        env=kalshi_client.env,
                        model_version=model_version,
                        best_bid_at_entry=cand.yes_bid,
                        best_ask_at_entry=cand.yes_ask,
                        market_mid_at_entry=cand.market_mid,
                    )
                    summary["orders_submitted"].append(
                        {
                            "client_order_id": cid,
                            "ticker": cand.ticker,
                            "count": decision.adjusted_count,
                            "dry_run": True,
                        }
                    )
                else:
                    reconciler.record_order(
                        client_order_id=cid,
                        ticker=cand.ticker,
                        player_id=cand.player_id,
                        stat_key=cand.stat_key,
                        threshold=cand.threshold,
                        side=cand.side,
                        execution_mode=cand.execution_mode,
                        limit_price=cand.order_price,
                        effective_cost=cand.effective_cost,
                        requested_count=decision.adjusted_count,
                        model_prob=cand.model_prob,
                        expected_ev=cand.ev_edge,
                        prediction_id=cand.prediction_id,
                        status="pending",
                        env=kalshi_client.env,
                        model_version=model_version,
                        best_bid_at_entry=cand.yes_bid,
                        best_ask_at_entry=cand.yes_ask,
                        market_mid_at_entry=cand.market_mid,
                    )
                    try:
                        resp = await kalshi_client.create_order(
                            ticker=cand.ticker,
                            side=cand.side,
                            count=decision.adjusted_count,
                            price=cand.order_price,
                            time_in_force=cand.time_in_force,
                            post_only=cand.post_only,
                            client_order_id=cid,
                        )
                        koid = (resp.get("order") or {}).get("order_id") or resp.get("order_id")
                        reconciler.update_order_submission(
                            cid, kalshi_order_id=koid, status="submitted"
                        )
                        notional = decision.adjusted_count * cand.effective_cost
                        by_ticker[cand.ticker] = by_ticker.get(cand.ticker, 0.0) + notional
                        total_exp += notional
                        summary["orders_submitted"].append(
                            {
                                "client_order_id": cid,
                                "kalshi_order_id": koid,
                                "ticker": cand.ticker,
                                "count": decision.adjusted_count,
                                "dry_run": False,
                            }
                        )
                    except KalshiAPIError as exc:
                        reconciler.update_order_submission(
                            cid, status="rejected", rejection_reason=str(exc)
                        )
                        summary["orders_blocked"].append({"ticker": cand.ticker, "reason": str(exc)})

        summary["fills_sync"] = await reconciler.sync_fills(kalshi_client)
        summary["closing_lines_sync"] = await reconciler.capture_closing_lines(kalshi_client)
        summary["shadow_settlements_sync"] = await reconciler.sync_shadow_settlements(kalshi_client)
        summary["settlements_sync"] = await reconciler.sync_settlements(kalshi_client)
        return summary
    finally:
        if owns_client:
            await client.aclose()


async def preflight_sandbox_connectivity(kalshi_client: KalshiClient) -> Dict[str, Any]:
    """Verifies signed balance access and public market/orderbook fetch without writing orders."""
    balance = await kalshi_client.get_balance()
    markets_resp = await kalshi_client.get_markets(status="open", limit=5)
    markets = list(markets_resp.get("markets", []))
    sample_ticker = markets[0].get("ticker") if markets else None
    sample_book = await kalshi_client.get_orderbook(sample_ticker) if sample_ticker else None
    return {
        "env": kalshi_client.env,
        "key_algorithm": kalshi_client.key_algorithm,
        "balance": balance,
        "open_markets_sampled": len(markets),
        "sample_ticker": sample_ticker,
        "orderbook_loaded": bool(sample_book),
    }


async def async_main(args: argparse.Namespace) -> None:
    dry_run = not args.no_dry_run
    default_demo_key = os.path.expanduser("~/.kalshi/demo.pem")
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not key_path and not os.getenv("KALSHI_PRIVATE_KEY_PEM") and args.env == "demo":
        if os.path.exists(default_demo_key):
            key_path = default_demo_key

    drift_monitor = DriftMonitor()
    pricer = LadderPricer(min_edge_bps=args.min_edge_bps)
    guardrails = RiskGuardrails(limits=RiskLimits(), drift_monitor=drift_monitor)
    reconciler = SettlementReconciler(
        drift_monitor=drift_monitor,
        auto_report_path="docs/sandbox_performance.md",
        auto_json_report_path="benchmarks/sandbox_performance.json",
    )

    async with KalshiClient(
        env=args.env,
        live_armed=args.live_armed,
        private_key_path=key_path,
    ) as kalshi_client:
        preflight = await preflight_sandbox_connectivity(kalshi_client)
        logger.info(f"Kalshi preflight connectivity verified: {preflight}")
        while True:
            res = await run_cycle(
                kalshi_client=kalshi_client,
                pricer=pricer,
                guardrails=guardrails,
                reconciler=reconciler,
                inference_url=args.inference_url,
                dry_run=dry_run,
            )
            logger.info(f"OMS cycle completed: {res}")
            if args.once:
                break
            await asyncio.sleep(args.poll_interval)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sandbox-first Kalshi OMS Runner")
    parser.add_argument("--env", choices=["demo", "prod"], default="demo")
    parser.add_argument("--live-armed", action="store_true", default=False)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", action="store_true", default=False)
    parser.add_argument("--once", action="store_true", default=False)
    parser.add_argument("--inference-url", default="http://localhost:8000")
    parser.add_argument("--min-edge-bps", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(async_main(parse_args()))
