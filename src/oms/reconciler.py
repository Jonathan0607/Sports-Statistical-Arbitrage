"""SQLAlchemy OMS order/fill ledger and closed-loop settlement reconciliation into DriftMonitor."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import sqlalchemy

from src.drift_monitor import DriftMonitor
from src.oms.kalshi_client import KalshiClient

logger = logging.getLogger("SettlementReconciler")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SettlementReconciler:
    """Persists OMS orders/fills and reconciles settled outcomes directly into DriftMonitor."""

    def __init__(self, drift_monitor: DriftMonitor):
        self.drift_monitor = drift_monitor
        self.engine = drift_monitor.engine
        self.metadata = drift_monitor.metadata

        if "oms_orders" not in self.metadata.tables:
            self.orders_table = sqlalchemy.Table(
                "oms_orders",
                self.metadata,
                sqlalchemy.Column("client_order_id", sqlalchemy.String(64), primary_key=True),
                sqlalchemy.Column("kalshi_order_id", sqlalchemy.String(64), nullable=True, index=True),
                sqlalchemy.Column(
                    "prediction_id",
                    sqlalchemy.String(64),
                    sqlalchemy.ForeignKey("prediction_telemetry.prediction_id"),
                    nullable=True,
                    index=True,
                ),
                sqlalchemy.Column("ticker", sqlalchemy.String(128), nullable=False, index=True),
                sqlalchemy.Column("player_id", sqlalchemy.String(64), nullable=False),
                sqlalchemy.Column("stat_key", sqlalchemy.String(32), nullable=False),
                sqlalchemy.Column("threshold", sqlalchemy.Integer, nullable=False),
                sqlalchemy.Column("side", sqlalchemy.String(16), nullable=False),
                sqlalchemy.Column("execution_mode", sqlalchemy.String(16), nullable=False),
                sqlalchemy.Column("limit_price", sqlalchemy.Float, nullable=False),
                sqlalchemy.Column("effective_cost", sqlalchemy.Float, nullable=False),
                sqlalchemy.Column("requested_count", sqlalchemy.Integer, nullable=False),
                sqlalchemy.Column("filled_count", sqlalchemy.Integer, nullable=False, default=0),
                sqlalchemy.Column("avg_fill_price", sqlalchemy.Float, nullable=True),
                sqlalchemy.Column("fees_paid", sqlalchemy.Float, nullable=False, default=0.0),
                sqlalchemy.Column("slippage", sqlalchemy.Float, nullable=True),
                sqlalchemy.Column("model_prob", sqlalchemy.Float, nullable=False),
                sqlalchemy.Column("expected_ev", sqlalchemy.Float, nullable=False),
                sqlalchemy.Column("status", sqlalchemy.String(32), nullable=False, index=True),
                sqlalchemy.Column("created_at", sqlalchemy.String(64), nullable=False),
                sqlalchemy.Column("settled_at", sqlalchemy.String(64), nullable=True),
                sqlalchemy.Column("realized_pnl", sqlalchemy.Float, nullable=True),
            )
        else:
            self.orders_table = self.metadata.tables["oms_orders"]

        if "oms_fills" not in self.metadata.tables:
            self.fills_table = sqlalchemy.Table(
                "oms_fills",
                self.metadata,
                sqlalchemy.Column("fill_id", sqlalchemy.String(64), primary_key=True),
                sqlalchemy.Column("client_order_id", sqlalchemy.String(64), nullable=True, index=True),
                sqlalchemy.Column("kalshi_order_id", sqlalchemy.String(64), nullable=True, index=True),
                sqlalchemy.Column("ticker", sqlalchemy.String(128), nullable=False, index=True),
                sqlalchemy.Column("side", sqlalchemy.String(16), nullable=False),
                sqlalchemy.Column("fill_count", sqlalchemy.Integer, nullable=False),
                sqlalchemy.Column("fill_price", sqlalchemy.Float, nullable=False),
                sqlalchemy.Column("fee_paid", sqlalchemy.Float, nullable=False, default=0.0),
                sqlalchemy.Column("slippage", sqlalchemy.Float, nullable=False, default=0.0),
                sqlalchemy.Column("filled_at", sqlalchemy.String(64), nullable=False),
            )
        else:
            self.fills_table = self.metadata.tables["oms_fills"]

        self.metadata.create_all(self.engine)

    def record_order(
        self,
        *,
        client_order_id: str,
        ticker: str,
        player_id: str,
        stat_key: str,
        threshold: int,
        side: str,
        execution_mode: str,
        limit_price: float,
        effective_cost: float,
        requested_count: int,
        model_prob: float,
        expected_ev: float,
        prediction_id: Optional[str] = None,
        kalshi_order_id: Optional[str] = None,
        status: str = "submitted",
        created_at: Optional[str] = None,
    ) -> str:
        row = {
            "client_order_id": str(client_order_id),
            "kalshi_order_id": str(kalshi_order_id) if kalshi_order_id else None,
            "prediction_id": str(prediction_id) if prediction_id else None,
            "ticker": str(ticker),
            "player_id": str(player_id),
            "stat_key": str(stat_key),
            "threshold": int(threshold),
            "side": str(side),
            "execution_mode": str(execution_mode),
            "limit_price": float(limit_price),
            "effective_cost": float(effective_cost),
            "requested_count": int(requested_count),
            "filled_count": 0,
            "avg_fill_price": None,
            "fees_paid": 0.0,
            "slippage": None,
            "model_prob": float(model_prob),
            "expected_ev": float(expected_ev),
            "status": str(status),
            "created_at": created_at or _utc_now_iso(),
            "settled_at": None,
            "realized_pnl": None,
        }
        with self.engine.begin() as conn:
            conn.execute(self.orders_table.insert(), row)
        return str(client_order_id)

    def update_order_submission(
        self,
        client_order_id: str,
        *,
        kalshi_order_id: Optional[str] = None,
        status: str = "submitted",
    ) -> None:
        vals: Dict[str, Any] = {"status": str(status)}
        if kalshi_order_id is not None:
            vals["kalshi_order_id"] = str(kalshi_order_id)
        with self.engine.begin() as conn:
            conn.execute(
                self.orders_table.update()
                .where(self.orders_table.c.client_order_id == client_order_id)
                .values(**vals)
            )

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(
                sqlalchemy.select(self.orders_table).where(
                    self.orders_table.c.client_order_id == client_order_id
                )
            ).mappings().first()
        return dict(row) if row else None

    def get_exposure_snapshot(self) -> Dict[str, Any]:
        """Aggregates open market/total notional exposure and settled daily realized PnL."""
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.engine.connect() as conn:
            rows = conn.execute(sqlalchemy.select(self.orders_table)).mappings().all()

        by_ticker: Dict[str, float] = {}
        total_exposure = 0.0
        realized_daily_pnl = 0.0

        for r in rows:
            status = r["status"]
            if status in ("submitted", "partially_filled", "filled"):
                cnt = int(r["filled_count"] or r["requested_count"])
                notional = cnt * float(r["effective_cost"])
                by_ticker[r["ticker"]] = by_ticker.get(r["ticker"], 0.0) + notional
                total_exposure += notional
            elif status == "settled" and r["realized_pnl"] is not None:
                settled_ts = str(r["settled_at"] or "")
                if settled_ts.startswith(today_prefix):
                    realized_daily_pnl += float(r["realized_pnl"])

        return {
            "by_ticker": by_ticker,
            "total_exposure_usd": float(total_exposure),
            "realized_daily_pnl_usd": float(realized_daily_pnl),
        }

    async def sync_fills(
        self, kalshi_client: KalshiClient, min_ts: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ingests /portfolio/fills, updates order fill stats and slippage deterministically."""
        fills = await kalshi_client.get_fills(min_ts=min_ts)
        inserted = 0
        updated_orders = 0

        with self.engine.begin() as conn:
            existing_fill_ids = {
                r[0] for r in conn.execute(sqlalchemy.select(self.fills_table.c.fill_id)).fetchall()
            }

            touched_cids = set()
            for f in fills:
                fid = str(f.get("fill_id") or f.get("trade_id") or f.get("id") or uuid.uuid4().hex)
                if fid in existing_fill_ids:
                    continue

                koid = f.get("order_id")
                cid = f.get("client_order_id")
                order_row = None
                if cid:
                    order_row = conn.execute(
                        sqlalchemy.select(self.orders_table).where(
                            self.orders_table.c.client_order_id == str(cid)
                        )
                    ).mappings().first()
                if order_row is None and koid:
                    order_row = conn.execute(
                        sqlalchemy.select(self.orders_table).where(
                            self.orders_table.c.kalshi_order_id == str(koid)
                        )
                    ).mappings().first()

                resolved_cid = str(order_row["client_order_id"]) if order_row else (str(cid) if cid else None)
                resolved_koid = str(koid) if koid else (str(order_row["kalshi_order_id"]) if order_row else None)

                raw_price = f.get("yes_price_dollars") or f.get("price") or (
                    float(f["yes_price"]) / 100.0 if f.get("yes_price") is not None else 0.0
                )
                fill_price = float(raw_price)
                fill_count = int(float(f.get("count") or f.get("fill_count") or 0))
                fee_paid = float(f.get("fee_paid") or f.get("fee_dollars") or 0.0)
                limit_p = float(order_row["limit_price"]) if order_row else fill_price
                slip = float(fill_price - limit_p)

                conn.execute(
                    self.fills_table.insert(),
                    {
                        "fill_id": fid,
                        "client_order_id": resolved_cid,
                        "kalshi_order_id": resolved_koid,
                        "ticker": str(f.get("ticker") or (order_row["ticker"] if order_row else "")),
                        "side": str(f.get("side") or (order_row["side"] if order_row else "bid")),
                        "fill_count": fill_count,
                        "fill_price": fill_price,
                        "fee_paid": fee_paid,
                        "slippage": slip,
                        "filled_at": str(f.get("created_time") or _utc_now_iso()),
                    },
                )
                existing_fill_ids.add(fid)
                inserted += 1
                if resolved_cid:
                    touched_cids.add(resolved_cid)

            for cid in touched_cids:
                order_row = conn.execute(
                    sqlalchemy.select(self.orders_table).where(
                        self.orders_table.c.client_order_id == cid
                    )
                ).mappings().first()
                if not order_row:
                    continue

                order_fills = conn.execute(
                    sqlalchemy.select(self.fills_table).where(
                        self.fills_table.c.client_order_id == cid
                    )
                ).mappings().all()

                total_cnt = sum(int(x["fill_count"]) for x in order_fills)
                if total_cnt <= 0:
                    continue
                weighted_px = sum(int(x["fill_count"]) * float(x["fill_price"]) for x in order_fills) / total_cnt
                total_fees = sum(float(x["fee_paid"]) for x in order_fills)
                slippage = float(weighted_px - float(order_row["limit_price"]))
                new_status = (
                    "settled"
                    if order_row["status"] == "settled"
                    else ("filled" if total_cnt >= int(order_row["requested_count"]) else "partially_filled")
                )

                conn.execute(
                    self.orders_table.update()
                    .where(self.orders_table.c.client_order_id == cid)
                    .values(
                        filled_count=total_cnt,
                        avg_fill_price=weighted_px,
                        fees_paid=total_fees,
                        slippage=slippage,
                        status=new_status,
                    )
                )
                updated_orders += 1

        return {"fills_inserted": inserted, "orders_updated": updated_orders}

    async def sync_settlements(
        self, kalshi_client: KalshiClient, min_ts: Optional[int] = None
    ) -> Dict[str, Any]:
        """Reconciles /portfolio/settlements into oms_orders PnL and DriftMonitor telemetry."""
        settlements = await kalshi_client.get_settlements(min_ts=min_ts)
        settled_orders = 0
        drift_updates = 0
        now_iso = _utc_now_iso()

        pending_drift: List[Dict[str, Any]] = []
        with self.engine.begin() as conn:
            for s in settlements:
                ticker = str(s.get("ticker", ""))
                if not ticker:
                    continue
                market_result = str(s.get("market_result") or s.get("result") or "").strip().lower()
                if market_result not in ("yes", "no"):
                    continue

                yes_won = market_result == "yes"
                orders = conn.execute(
                    sqlalchemy.select(self.orders_table).where(
                        sqlalchemy.and_(
                            self.orders_table.c.ticker == ticker,
                            self.orders_table.c.status.in_(
                                ["pending", "submitted", "partially_filled", "filled"]
                            ),
                        )
                    )
                ).mappings().all()

                for o in orders:
                    filled = int(o["filled_count"] or 0)
                    avg_px = float(o["avg_fill_price"]) if o["avg_fill_price"] is not None else float(o["limit_price"])
                    fees = float(o["fees_paid"] or 0.0)
                    if filled > 0:
                        if str(o["side"]).lower() == "bid":
                            payout_per = 1.0 if yes_won else 0.0
                            cost_per = avg_px
                        else:
                            payout_per = 1.0 if (not yes_won) else 0.0
                            cost_per = 1.0 - avg_px
                        pnl = float((payout_per - cost_per) * filled - fees)
                    else:
                        pnl = 0.0

                    settled_ts = str(s.get("settled_time") or now_iso)
                    conn.execute(
                        self.orders_table.update()
                        .where(self.orders_table.c.client_order_id == o["client_order_id"])
                        .values(
                            status="settled",
                            settled_at=settled_ts,
                            realized_pnl=pnl,
                        )
                    )
                    settled_orders += 1

                    if o["prediction_id"]:
                        thresh = int(o["threshold"])
                        realized_val = s.get("settlement_value")
                        if realized_val is None:
                            realized_val = float(thresh) if yes_won else float(max(0, thresh - 1))
                        pending_drift.append(
                            {
                                "prediction_id": str(o["prediction_id"]),
                                "realized_value": float(realized_val),
                                "realized_hit": bool(yes_won),
                            }
                        )

        for item in pending_drift:
            ok = self.drift_monitor.record_settlement(
                prediction_id=item["prediction_id"],
                realized_value=item["realized_value"],
                realized_hit=item["realized_hit"],
            )
            if ok:
                drift_updates += 1

        return {"orders_settled": settled_orders, "drift_telemetry_settled": drift_updates}
