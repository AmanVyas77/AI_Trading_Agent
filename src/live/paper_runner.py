"""
One-command monthly paper-trading loop.

Flow:
    1. run_refresh()                       — data current + freshness assert
    2. build_targets()                     — score + regime-gate + persist
    3. broker.get_positions() / get_account()
    4. diff_orders(targets, positions, equity)
    5. --dry-run (DEFAULT): print orders and exit
       --execute: cancel any open orders, submit sequentially, poll fills
    6. write logs/live_runs/<YYYY-MM>_run.json (freshness, targets, orders,
       fills, positions-vs-targets deltas) + append live_runs SQLite row

The default is --dry-run so a bare invocation never touches the broker.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml

from src.live.rebalance import build_targets, diff_orders
from src.live.refresh import run_refresh

ROOT = Path(__file__).resolve().parents[2]
LIVE_RUNS_DIR = ROOT / "logs" / "live_runs"

with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)

DB_PATH = ROOT / _CFG["data"]["paths"]["db"]

logger = logging.getLogger(__name__)


# ── SQLite live_runs ─────────────────────────────────────────────────────────

def _ensure_live_runs_table() -> None:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS live_runs ("
            "  run_id      INTEGER PRIMARY KEY AUTOINCREMENT, "
            "  month       TEXT NOT NULL, "
            "  ran_at      TEXT NOT NULL, "
            "  mode        TEXT NOT NULL, "         # dry_run | execute
            "  n_orders    INTEGER NOT NULL, "
            "  verdict     TEXT, "
            "  payload     TEXT NOT NULL"
            ")"
        )


def _persist_run(record: dict) -> None:
    _ensure_live_runs_table()
    LIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    month_key = record["month"]
    path = LIVE_RUNS_DIR / f"{month_key}_run.json"

    # If this is a re-run for the same month, keep prior payloads by
    # bumping a suffix on the file; SQLite gets a new row either way.
    if path.exists() and record["mode"] == "execute":
        stamp = record["ran_at"].replace(":", "").replace("-", "")
        path = LIVE_RUNS_DIR / f"{month_key}_run_{stamp}.json"

    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO live_runs (month, ran_at, mode, n_orders, verdict, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record["month"],
                record["ran_at"],
                record["mode"],
                record["n_orders"],
                record.get("verdict"),
                json.dumps(record, default=str),
            ),
        )
    logger.info("Run record → %s", path)


# ── Positions-vs-targets ─────────────────────────────────────────────────────

def _position_vs_target_deltas(
    targets: dict, positions: list[dict], equity: float,
) -> list[dict]:
    target_w = {t["ticker"]: t["weight"] for t in targets.get("targets", [])}
    pos_w = {
        p["ticker"]: float(p["market_value"]) / equity for p in positions
    } if equity > 0 else {}

    rows: list[dict] = []
    for tkr in sorted(set(target_w) | set(pos_w)):
        rows.append({
            "ticker": tkr,
            "target_weight": round(target_w.get(tkr, 0.0), 6),
            "actual_weight": round(pos_w.get(tkr, 0.0), 6),
            "delta": round(pos_w.get(tkr, 0.0) - target_w.get(tkr, 0.0), 6),
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def run(
    as_of: Optional[str] = None,
    execute: bool = False,
    skip_refresh: bool = False,
    order_timeout_s: float = 60.0,
) -> dict:
    """One paper-trading cycle. `execute=False` is the safe default."""
    # Deferred so --help works without a live SDK import cycle.
    from src.live.broker_alpaca import AlpacaPaperBroker

    mode = "execute" if execute else "dry_run"
    logger.info("paper_runner: mode=%s  as_of=%s", mode, as_of or "today")

    # 1. refresh
    if skip_refresh:
        logger.info("--skip-refresh: reading last freshness report from JSONL")
        try:
            with open(ROOT / "logs" / "live_refresh_log.jsonl") as f:
                fresh_report = json.loads(list(f)[-1])
        except Exception as exc:
            raise RuntimeError(f"cannot load prior freshness report: {exc}")
    else:
        fresh_report = run_refresh(as_of=as_of)

    # 2. targets
    targets = build_targets()
    month = datetime.fromisoformat(targets["as_of"]).strftime("%Y-%m")

    # 3. broker
    broker = AlpacaPaperBroker()
    account = broker.get_account()
    positions = broker.get_positions()
    equity = account["equity"]

    # 4. diff
    orders = diff_orders(targets, current_positions=positions, equity=equity)
    gross_exposure = targets["gross_exposure"] * equity
    logger.info(
        "Account: equity=$%.2f  cash=$%.2f  buying_power=$%.2f  "
        "target gross=$%.2f (mult=%.2f)",
        equity, account["cash"], account["buying_power"],
        gross_exposure, targets["regime"]["multiplier"],
    )

    # Margin sanity: refuse to submit if gross > buying_power
    if execute and gross_exposure > account["buying_power"] + 1.0:
        raise RuntimeError(
            f"Refusing to submit: target gross ${gross_exposure:.2f} exceeds "
            f"buying_power ${account['buying_power']:.2f}"
        )

    # 5. submit / dry-run
    submitted_ids: list[str] = []
    skipped_open: list[dict] = []
    fills: dict[str, dict] = {}

    if execute:
        # Idempotency (sprint rule 5): if the broker already has an open
        # order for a (ticker, side) we were about to submit, skip that
        # order. A queued Sunday order stays queued until Monday's open;
        # re-running before then must not double-submit.
        open_orders = broker.get_open_orders()
        open_keys = {(o["ticker"], o["side"]) for o in open_orders}
        submit_orders_list: list[dict] = []
        for o in orders:
            if (o["ticker"], o["side"]) in open_keys:
                skipped_open.append(o)
                logger.info(
                    "SKIP submit: %s %s already has open order at broker",
                    o["side"], o["ticker"],
                )
            else:
                submit_orders_list.append(o)

        submitted_ids = broker.submit_orders(submit_orders_list, month=month)
        if submitted_ids:
            logger.info("Submitted %d orders — polling fills (≤%.0fs)…",
                        len(submitted_ids), order_timeout_s)
            fills = broker.poll_fills(submitted_ids, timeout_s=order_timeout_s)

        # refresh positions after fills for the deltas table
        positions = broker.get_positions()
        account = broker.get_account()
        equity = account["equity"]

    deltas = _position_vs_target_deltas(targets, positions, equity)

    # 6. record
    record = {
        "month": month,
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "as_of": as_of,
        "broker": {
            "endpoint": broker.base_url,
            "skipped_already_open": skipped_open,
        },
        "account_before" if execute else "account": account,
        "freshness": fresh_report,
        "targets": targets,
        "orders": orders,
        "n_orders": len(orders),
        "n_submitted": len(submitted_ids),
        "submitted_ids": submitted_ids,
        "fills": fills,
        "positions_vs_targets": deltas,
    }
    _persist_run(record)
    return record


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Live paper-trading monthly loop")
    ap.add_argument("--as-of", default=None, help="Anchor date (YYYY-MM-DD)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually submit orders (default: dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help=argparse.SUPPRESS)  # backwards-compat no-op
    ap.add_argument("--skip-refresh", action="store_true",
                    help="Reuse last freshness report from JSONL")
    ap.add_argument("--order-timeout", type=float, default=60.0)
    args = ap.parse_args()

    record = run(
        as_of=args.as_of,
        execute=args.execute,
        skip_refresh=args.skip_refresh,
        order_timeout_s=args.order_timeout,
    )

    print("\n" + "═" * 72)
    print(f"  paper_runner — month={record['month']}  mode={record['mode']}")
    print("═" * 72)
    r = record["targets"]["regime"]
    print(f"  Regime:      mult={r['multiplier']}  rule={r['rule_signal']}  "
          f"stale={r['stale']}")
    print(f"  n_selected:  {record['targets']['n_selected']}")
    print(f"  n_orders:    {record['n_orders']}")
    print(f"  n_submitted: {record.get('n_submitted', 0)} "
          f"(skipped_already_open: "
          f"{len(record['broker'].get('skipped_already_open', []))})")
    if record["orders"]:
        print("  Orders:")
        for o in record["orders"]:
            print(f"    {o['side']:<4s}  {o['ticker']:<6s}  "
                  f"notional=${o['notional']:.2f}")
    if record["fills"]:
        n_filled = sum(1 for f in record["fills"].values() if f["status"] == "filled")
        print(f"  Fills:       {n_filled}/{len(record['fills'])} filled")
    if record["positions_vs_targets"]:
        print("  Positions vs targets (top 10 by |delta|):")
        rows = sorted(record["positions_vs_targets"],
                      key=lambda r: -abs(r["delta"]))[:10]
        for r_ in rows:
            print(f"    {r_['ticker']:<6s}  target={r_['target_weight']:+.4f}  "
                  f"actual={r_['actual_weight']:+.4f}  "
                  f"delta={r_['delta']:+.4f}")
    print("═" * 72)


if __name__ == "__main__":
    main()
