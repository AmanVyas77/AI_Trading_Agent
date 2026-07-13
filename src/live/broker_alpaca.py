"""
Thin Alpaca paper-trading adapter.

DESIGN CONSTRAINT: this module has NO code path to the live-money
endpoint. The paper URL is a module-level constant, the client is
instantiated with paper=True, and __init__ asserts that the SDK's
resolved base URL equals that constant. Any drift raises immediately.

Public surface:
    get_account()              -> {equity, cash, buying_power}
    get_positions()            -> [{ticker, qty, market_value}]
    submit_orders(orders)      -> [order_id, …]  (records live_orders row)
    poll_fills(order_ids, ...) -> {order_id: {status, filled_qty, ...}}

Orders are MARKET/DAY only. Fractional notional is preferred when the
target notional isn't a whole share of the last price; the fallback is
whole-share qty computed from the last-known close.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import yaml
from dotenv import load_dotenv

from alpaca.common.enums import BaseURL
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# ── Paper endpoint hard-lock ─────────────────────────────────────────────────
PAPER_BASE_URL = "https://paper-api.alpaca.markets"

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)

DB_PATH = ROOT / _CFG["data"]["paths"]["db"]

TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
    OrderStatus.DONE_FOR_DAY,
}


# ── SQLite live_orders ───────────────────────────────────────────────────────

def _ensure_live_orders_table() -> None:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS live_orders ("
            "  order_id     TEXT PRIMARY KEY, "
            "  month        TEXT NOT NULL, "
            "  ticker       TEXT NOT NULL, "
            "  side         TEXT NOT NULL, "
            "  notional     REAL, "
            "  qty          REAL, "
            "  status       TEXT NOT NULL, "
            "  submitted_at TEXT NOT NULL"
            ")"
        )


def _log_order(row: dict) -> None:
    _ensure_live_orders_table()
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO live_orders "
            "(order_id, month, ticker, side, notional, qty, status, submitted_at) "
            "VALUES (:order_id, :month, :ticker, :side, :notional, :qty, "
            ":status, :submitted_at)",
            row,
        )


# ── AlpacaPaperBroker ────────────────────────────────────────────────────────

class AlpacaPaperBroker:
    """Wraps alpaca-py TradingClient in paper mode. Refuses to live-trade."""

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY missing from environment. "
                "Add them to .env — never paste secrets into chat."
            )

        # paper=True forces the SDK to pick TRADING_PAPER. We double-check
        # below so a future SDK bug can't silently switch us to live.
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )

        resolved = self._client._base_url  # BaseURL enum
        if resolved != BaseURL.TRADING_PAPER:
            raise RuntimeError(
                f"Kill-switch: TradingClient resolved to {resolved!r}, "
                f"expected {BaseURL.TRADING_PAPER!r} (paper). Refusing to run."
            )
        if str(resolved.value) != PAPER_BASE_URL:
            raise RuntimeError(
                f"Kill-switch: SDK paper URL {resolved.value!r} does not match "
                f"the pinned constant {PAPER_BASE_URL!r}. Refusing to run."
            )
        logger.info("AlpacaPaperBroker locked to %s", PAPER_BASE_URL)

    # ── Account ─────────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return PAPER_BASE_URL

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

    def get_positions(self) -> list[dict]:
        positions = self._client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
            }
            for p in positions
        ]

    # ── Orders ──────────────────────────────────────────────────────────

    def _submit_one(self, order: dict, month: str) -> dict:
        """Submit a single MARKET DAY order. Prefer notional; fall back
        to whole-share qty if notional isn't valid for the symbol."""
        side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL
        req_kwargs = dict(
            symbol=order["ticker"],
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        notional = float(order.get("notional", 0.0))
        submitted = None
        submit_notional: Optional[float] = None
        submit_qty: Optional[float] = None

        # Try notional (fractional) first. If the broker rejects (e.g.
        # a non-fractionable symbol), retry with whole-share qty using
        # the current market price so the operator always gets a fill.
        try:
            req = MarketOrderRequest(notional=round(notional, 2), **req_kwargs)
            submitted = self._client.submit_order(req)
            submit_notional = round(notional, 2)
        except Exception as exc:
            logger.warning(
                "Notional order for %s rejected (%s) — retrying whole-share qty",
                order["ticker"], exc,
            )
            price = self._latest_price(order["ticker"])
            if price is None or price <= 0:
                raise RuntimeError(
                    f"cannot size whole-share fallback for {order['ticker']}: "
                    "no price available"
                )
            qty = max(1, int(notional / price))
            req = MarketOrderRequest(qty=qty, **req_kwargs)
            submitted = self._client.submit_order(req)
            submit_qty = float(qty)

        row = {
            "order_id": str(submitted.id),
            "month": month,
            "ticker": order["ticker"],
            "side": order["side"],
            "notional": submit_notional,
            "qty": submit_qty,
            "status": str(submitted.status.value if hasattr(submitted.status, "value") else submitted.status),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
        }
        _log_order(row)
        return row

    def _latest_price(self, ticker: str) -> Optional[float]:
        """Fallback: use the DB's most recent adj_close for whole-share sizing."""
        try:
            with sqlite3.connect(str(DB_PATH)) as conn:
                row = conn.execute(
                    "SELECT adj_close FROM prices WHERE ticker=? "
                    "ORDER BY date DESC LIMIT 1",
                    (ticker,),
                ).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def submit_orders(self, orders: list[dict], month: str) -> list[str]:
        """Submit sequentially so cash frees up before consuming it."""
        ids: list[str] = []
        for order in orders:
            record = self._submit_one(order, month=month)
            ids.append(record["order_id"])
        return ids

    def poll_fills(
        self,
        order_ids: Iterable[str],
        timeout_s: float = 60.0,
        interval_s: float = 2.0,
    ) -> dict[str, dict]:
        """Poll each order until terminal status or timeout. Returns
        {order_id: {status, filled_qty, filled_avg_price, symbol}}."""
        pending = list(order_ids)
        final: dict[str, dict] = {}
        deadline = time.monotonic() + timeout_s

        while pending and time.monotonic() < deadline:
            still_pending: list[str] = []
            for oid in pending:
                order = self._client.get_order_by_id(oid)
                status = order.status
                if status in TERMINAL_STATUSES:
                    final[oid] = {
                        "status": str(status.value),
                        "symbol": order.symbol,
                        "filled_qty": float(order.filled_qty or 0),
                        "filled_avg_price": float(order.filled_avg_price or 0)
                            if order.filled_avg_price else None,
                    }
                else:
                    still_pending.append(oid)
            pending = still_pending
            if pending:
                time.sleep(interval_s)

        # anything still pending after timeout: record last-seen status
        for oid in pending:
            order = self._client.get_order_by_id(oid)
            final[oid] = {
                "status": str(order.status.value),
                "symbol": order.symbol,
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0)
                    if order.filled_avg_price else None,
            }

        # persist final statuses back to live_orders
        with sqlite3.connect(str(DB_PATH)) as conn:
            for oid, cell in final.items():
                conn.execute(
                    "UPDATE live_orders SET status = ? WHERE order_id = ?",
                    (cell["status"], oid),
                )
        return final

    def get_open_orders(self) -> list[dict]:
        """Return currently-open orders as {ticker, side, id, qty, notional}."""
        req = GetOrdersRequest(status="open")
        open_orders = self._client.get_orders(req)
        rows: list[dict] = []
        for o in open_orders:
            side = o.side.value if hasattr(o.side, "value") else str(o.side)
            rows.append({
                "id": str(o.id),
                "ticker": o.symbol,
                "side": side,
                "qty": float(o.qty) if o.qty is not None else None,
                "notional": float(o.notional) if o.notional is not None else None,
            })
        return rows
