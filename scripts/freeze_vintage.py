#!/usr/bin/env python
"""Physically freeze every data input the exit-layer diagnostic path reads.

Why
---
`src/utils/data_vintage.py` fingerprints the data — it detects that the inputs
changed, but does not stop them changing. The Alpha Vantage backfill runs nightly
(see `logs/av_backfill_state.jsonl`) and retroactively revises historical rows, so
without a physical snapshot each of Prompts 2/3/4 would read a different dataset
and none of their numbers would be comparable to each other or to the -0.3049
measured on 2026-08-03.

This copies the inputs byte-for-byte into a snapshot directory and writes a
MANIFEST.json recording sha256/size/mtime for each, the data_vintage digest, the
git HEAD sha, and the interpreter/version block.

The frozen model is deliberately NOT copied — it is referenced in place and its
md5 asserted, since it is already immutable by convention.

Usage
-----
    .venv/bin/python scripts/freeze_vintage.py --out backtests/vintage_2026-08-04
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data_vintage import price_vintage  # noqa: E402

EXPECTED_MODEL_MD5 = "296e589f4da205eb1d171c2121d90f82"
EXPECTED_VERSIONS = {"numpy": "2.4.4", "sklearn": "1.8.0", "pandas": "2.3.3"}

# Copied byte-for-byte into the snapshot.
FROZEN_INPUTS = [
    "data/quant_research.db",
    "data/processed/ensemble_feature_matrix.parquet",
    "data/processed/holdout_scores.parquet",
]

# Recorded (hash asserted) but NOT copied — immutable by convention / git-tracked.
REFERENCED_INPUTS = [
    "models/ensemble_models.pkl",
    "config/settings.yaml",
]

WINDOW = ("2025-01-01", "2026-06-30")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _stat(path: Path) -> dict:
    st = path.stat()
    return {
        "source_abspath": str(path.resolve()),
        "size_bytes": st.st_size,
        "sha256": _sha256(path),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


def _interpreter_block() -> dict:
    import numpy, pandas, sklearn  # noqa: E401
    return {
        "sys_executable": sys.executable,
        "python_version": sys.version,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
    }


def _assert_interpreter() -> dict:
    block = _interpreter_block()
    if ".venv" not in block["sys_executable"]:
        raise SystemExit(f"ABORT: not the repo .venv → {block['sys_executable']}")
    if "anaconda" in block["sys_executable"].lower():
        raise SystemExit("ABORT: anaconda interpreter")
    bad = {k: block[k] for k, v in EXPECTED_VERSIONS.items() if block[k] != v}
    if bad:
        raise SystemExit(f"ABORT: version mismatch {bad}, expected {EXPECTED_VERSIONS}")
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="snapshot directory")
    args = ap.parse_args()

    interp = _assert_interpreter()
    print(f"interpreter OK: {interp['sys_executable']}")

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── model: reference in place, assert md5 ────────────────────────────
    model = ROOT / "models" / "ensemble_models.pkl"
    model_md5 = _md5(model)
    if model_md5 != EXPECTED_MODEL_MD5:
        raise SystemExit(
            f"ABORT: model md5 {model_md5} != expected {EXPECTED_MODEL_MD5}")
    print(f"model md5 OK (referenced in place, not copied): {model_md5}")

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": ("Frozen data vintage for the Markov exit-layer diagnostic "
                    "(Prompts 2-4). The AV backfill retroactively revises "
                    "historical rows nightly; this snapshot makes those prompts "
                    "mutually comparable."),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
        "interpreter": interp,
        "window": list(WINDOW),
        "frozen": {},
        "referenced": {},
    }

    # ── copy frozen inputs ───────────────────────────────────────────────
    for rel in FROZEN_INPUTS:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"ABORT: missing input {src}")
        dst = out / Path(rel).name
        print(f"copying {rel} ({src.stat().st_size/1e6:.1f} MB) → {dst.name}")
        shutil.copy2(src, dst)

        src_meta = _stat(src)
        dst_sha = _sha256(dst)
        if dst_sha != src_meta["sha256"]:
            # A concurrent backfill write during the copy would land here.
            raise SystemExit(
                f"ABORT: copy of {rel} does not match source sha256 "
                f"(src={src_meta['sha256']} dst={dst_sha}) — source changed "
                "mid-copy; re-run when the backfill is idle")
        manifest["frozen"][Path(rel).name] = {
            **src_meta,
            "snapshot_relpath": Path(rel).name,
            "source_relpath": rel,
            "verified_after_copy": True,
        }

    # ── referenced (not copied) ──────────────────────────────────────────
    for rel in REFERENCED_INPUTS:
        src = ROOT / rel
        meta = _stat(src)
        meta["source_relpath"] = rel
        meta["copied"] = False
        if rel.endswith("ensemble_models.pkl"):
            meta["md5"] = model_md5
            meta["md5_expected"] = EXPECTED_MODEL_MD5
        manifest["referenced"][Path(rel).name] = meta

    # ── data_vintage digest over the SNAPSHOT db ─────────────────────────
    snap_db = out / "quant_research.db"
    manifest["price_vintage"] = price_vintage(snap_db, *WINDOW)
    src_vintage = price_vintage(ROOT / "data" / "quant_research.db", *WINDOW)
    if manifest["price_vintage"]["sha256"] != src_vintage["sha256"]:
        raise SystemExit("ABORT: snapshot price_vintage != source price_vintage")
    manifest["price_vintage_matches_source_at_freeze_time"] = True

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nMANIFEST → {out / 'MANIFEST.json'}")
    pv = manifest["price_vintage"]
    print(f"  price_vintage sha256 = {pv['sha256']}")
    print(f"  rows={pv['n_rows']} tickers={pv['n_tickers']} "
          f"span={pv['date_min']}→{pv['date_max']}")
    print(f"  git HEAD = {manifest['git_head']} (dirty={manifest['git_dirty']})")
    total = sum(v["size_bytes"] for v in manifest["frozen"].values())
    print(f"  frozen {len(manifest['frozen'])} files, {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
