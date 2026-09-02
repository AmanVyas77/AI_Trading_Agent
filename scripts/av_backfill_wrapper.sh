#!/bin/bash
# launchd wrapper for src.data.news_pipeline av-backfill
# rationale: invoking anaconda python directly from launchd hits
# 'InterruptedError: [Errno 4]' during fs-cache init when the working
# directory contains a space. cd first, then exec.
#
# PROJECT_DIR is self-detected from this script's own location
# (scripts/ -> project root). This is deliberate: the project root has
# moved three times, and a hardcoded absolute path here has silently
# broken the nightly job on every move. Deriving it from $0 means a
# future move no longer requires editing this file.
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/opt/anaconda3/bin/python"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
export PATH="/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"

echo "===== $(date '+%Y-%m-%d %H:%M:%S %z') launchd fire (PROJECT_DIR=$PROJECT_DIR) ====="
exec "$PYTHON" -m src.data.news_pipeline av-backfill "$@"
