#!/usr/bin/env bash
# Container entrypoint: guarantee the core model weights exist in the persisted
# HF cache on first start, then hand off to whatever command was passed (the
# server by default). snapshot_download is idempotent, so later starts only do a
# cheap etag check before the server boots.
set -uo pipefail

echo "[entrypoint] prefetching model weights (first run downloads tens of GB; later runs are instant) ..."
python -m webapp.prefetch_models \
  || echo "[entrypoint] prefetch warning: continuing; the server will lazy-load any missing weights."

exec "$@"
