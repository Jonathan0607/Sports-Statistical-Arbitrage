"""Authenticated async Kalshi Trade API v2 client supporting Ed25519 & RSA-PSS keys.

Defaults strictly to Sandbox (`demo`). Production (`prod`) raises RuntimeError unless BOTH
`env="prod"` and `live_armed=True` are explicitly supplied.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

logger = logging.getLogger("KalshiClient")

DEMO_BASE_URL = "https://external-api.demo.kalshi.co/trade-api/v2"
PROD_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
API_PREFIX = "/trade-api/v2"


class KalshiAPIError(RuntimeError):
    """Raised on non-recoverable Kalshi Trade API v2 HTTP errors."""

    def __init__(self, status_code: int, message: str, body: Any = None):
        super().__init__(f"Kalshi HTTP {status_code}: {message}")
        self.status_code = status_code
        self.body = body


class KalshiClient:
    """Async REST client for Kalshi Trade API v2 with sandbox-first enforcement."""

    def __init__(
        self,
        *,
        env: str = "demo",
        live_armed: bool = False,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[Union[str, bytes]] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.25,
        seed: int = 42,
    ):
        env_key = (env or "demo").strip().lower()
        if env_key not in ("demo", "prod"):
            raise ValueError(f"Unsupported Kalshi env '{env}'; expected 'demo' or 'prod'.")
        if env_key == "prod" and not live_armed:
            raise RuntimeError(
                "Production Kalshi API access blocked: both env='prod' and live_armed=True are required."
            )

        self.env = env_key
        self.live_armed = bool(live_armed)
        self.base_url = (base_url or (PROD_BASE_URL if env_key == "prod" else DEMO_BASE_URL)).rstrip("/")
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY", "")
        self._private_key = self._load_private_key(
            private_key_pem=private_key_pem or os.getenv("KALSHI_PRIVATE_KEY_PEM"),
            private_key_path=private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH"),
        )
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_sec = float(backoff_base_sec)
        self._rng = random.Random(seed)
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _load_private_key(
        private_key_pem: Optional[Union[str, bytes]] = None,
        private_key_path: Optional[str] = None,
    ) -> Any:
        raw_bytes: Optional[bytes] = None
        if private_key_pem:
            raw_bytes = (
                private_key_pem.encode("utf-8")
                if isinstance(private_key_pem, str)
                else private_key_pem
            )
        elif private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                raw_bytes = f.read()

        if not raw_bytes:
            return None

        key = serialization.load_pem_private_key(raw_bytes, password=None)
        if not isinstance(key, (ed25519.Ed25519PrivateKey, rsa.RSAPrivateKey)):
            raise TypeError(
                f"Unsupported private key type '{type(key).__name__}'; expected Ed25519 or RSA."
            )
        return key

    @property
    def key_algorithm(self) -> Optional[str]:
        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            return "ed25519"
        if isinstance(self._private_key, rsa.RSAPrivateKey):
            return "rsa_pss_sha256"
        return None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "KalshiClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @staticmethod
    def _normalize_signed_path(path: str) -> str:
        clean_path = urlsplit(path).path or "/"
        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path
        if clean_path.startswith(API_PREFIX):
            return clean_path
        return f"{API_PREFIX}{clean_path}"

    def build_auth_headers(
        self,
        method: str,
        path: str,
        timestamp_ms: Optional[str] = None,
    ) -> Dict[str, str]:
        """Builds Kalshi v2 HMAC-less asymmetric auth headers (Ed25519 or RSA-PSS SHA-256)."""
        if self._private_key is None or not self.api_key_id:
            raise RuntimeError("Kalshi API key ID and PEM private key are required for authenticated calls.")

        ts = str(timestamp_ms) if timestamp_ms is not None else str(int(time.time() * 1000))
        signed_path = self._normalize_signed_path(path)
        message = f"{ts}{method.upper()}{signed_path}".encode("utf-8")

        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            sig_bytes = self._private_key.sign(message)
        elif isinstance(self._private_key, rsa.RSAPrivateKey):
            sig_bytes = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        else:
            raise TypeError(f"Unsupported key object: {type(self._private_key).__name__}")

        sig_b64 = base64.b64encode(sig_bytes).decode("ascii")
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
        handle_conflict_idempotently: bool = False,
    ) -> Dict[str, Any]:
        client = await self._ensure_client()
        rel_path = path if path.startswith("/") else f"/{path}"
        if rel_path.startswith(API_PREFIX):
            rel_path = rel_path[len(API_PREFIX) :] or "/"

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            headers: Dict[str, str] = {"Accept": "application/json"}
            if json_body is not None:
                headers["Content-Type"] = "application/json"
            if authenticated:
                headers.update(self.build_auth_headers(method, rel_path))

            try:
                resp = await client.request(
                    method=method.upper(),
                    url=rel_path,
                    params=params,
                    json=json_body,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_base_sec * (2 ** attempt) + self._rng.uniform(0.0, 0.1)
                    await asyncio.sleep(delay)
                    continue
                raise KalshiAPIError(0, f"Network error calling {rel_path}: {exc}") from exc

            if resp.status_code == 409 and handle_conflict_idempotently:
                try:
                    parsed = resp.json()
                except Exception:
                    parsed = {"raw": resp.text}
                return {
                    "status": "duplicate_conflict",
                    "idempotent_already_submitted": True,
                    "client_order_id": (json_body or {}).get("client_order_id"),
                    "response": parsed,
                }

            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                if attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = max(0.0, float(retry_after))
                        except ValueError:
                            delay = self.backoff_base_sec * (2 ** attempt)
                    else:
                        delay = self.backoff_base_sec * (2 ** attempt) + self._rng.uniform(0.0, 0.1)
                    await asyncio.sleep(delay)
                    continue

            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise KalshiAPIError(resp.status_code, str(body), body=body)

            if not resp.content:
                return {}
            return resp.json()

        raise KalshiAPIError(0, f"Exhausted retries calling {rel_path}: {last_exc}")

    # ------------------------------------------------------------------
    # Public Market Data
    # ------------------------------------------------------------------
    async def get_markets(
        self,
        *,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"status": status, "limit": int(limit)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/markets", params=params, authenticated=False)

    async def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict[str, Any]:
        params = {"depth": int(depth)} if depth is not None else None
        return await self._request(
            "GET", f"/markets/{ticker}/orderbook", params=params, authenticated=False
        )

    # ------------------------------------------------------------------
    # Authenticated Order Lifecycle
    # ------------------------------------------------------------------
    async def create_order(
        self,
        *,
        ticker: str,
        side: str,
        count: int,
        price: Union[float, str],
        time_in_force: str = "immediate_or_cancel",
        self_trade_prevention_type: str = "taker_at_cross",
        post_only: bool = False,
        cancel_order_on_pause: bool = True,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        side_norm = side.strip().lower()
        if side_norm not in ("bid", "ask"):
            raise ValueError(f"Invalid Kalshi order side '{side}'; expected 'bid' or 'ask'.")
        if time_in_force not in ("immediate_or_cancel", "good_till_canceled", "fill_or_kill"):
            raise ValueError(f"Invalid time_in_force '{time_in_force}'.")
        if int(count) <= 0:
            raise ValueError(f"Order count must be positive, got {count}.")

        price_val = float(price)
        if not (0.0001 <= price_val <= 0.9999):
            raise ValueError(f"Order price {price_val} outside [0.0001, 0.9999].")

        cid = client_order_id or str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "ticker": ticker,
            "side": side_norm,
            "count": str(int(count)),
            "price": f"{price_val:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "post_only": bool(post_only),
            "cancel_order_on_pause": bool(cancel_order_on_pause),
            "client_order_id": cid,
        }
        return await self._request(
            "POST",
            "/portfolio/events/orders",
            json_body=payload,
            authenticated=True,
            handle_conflict_idempotently=True,
        )

    async def cancel_order(
        self, order_id: str, market_ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {"market_ticker": market_ticker} if market_ticker else None
        return await self._request(
            "DELETE",
            f"/portfolio/events/orders/{order_id}",
            params=params,
            authenticated=True,
        )

    async def get_open_orders(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"status": "resting"}
        if ticker:
            params["ticker"] = ticker
        data = await self._request("GET", "/portfolio/orders", params=params, authenticated=True)
        return list(data.get("orders", []))

    async def cancel_all_open_orders(self, ticker: Optional[str] = None) -> List[str]:
        params = {"market_ticker": ticker} if ticker else None
        data = await self._request(
            "DELETE",
            "/portfolio/events/orders",
            params=params,
            authenticated=True,
        )
        raw_orders = (
            data.get("orders")
            or data.get("canceled_orders")
            or data.get("cancelled_orders")
            or data.get("order_ids")
            or []
        )
        cancelled: List[str] = []
        for item in raw_orders:
            if isinstance(item, dict):
                oid = item.get("order_id") or item.get("id")
                if oid:
                    cancelled.append(str(oid))
            elif item is not None:
                cancelled.append(str(item))
        return cancelled

    # ------------------------------------------------------------------
    # Portfolio & Reconciliation
    # ------------------------------------------------------------------
    async def get_balance(self) -> Dict[str, Any]:
        return await self._request("GET", "/portfolio/balance", authenticated=True)

    async def get_positions(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"ticker": ticker} if ticker else None
        data = await self._request("GET", "/portfolio/positions", params=params, authenticated=True)
        return list(data.get("market_positions", data.get("positions", [])))

    async def get_fills(self, min_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": int(limit)}
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        data = await self._request("GET", "/portfolio/fills", params=params, authenticated=True)
        return list(data.get("fills", []))

    async def get_settlements(self, min_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": int(limit)}
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        data = await self._request("GET", "/portfolio/settlements", params=params, authenticated=True)
        return list(data.get("settlements", []))
