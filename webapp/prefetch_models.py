"""Idempotent prefetch of the core Hunyuan3D weights so the FIRST container start
populates the persisted HF cache (the hf-cache volume). snapshot_download skips
files already present, so subsequent starts cost only a cheap etag check.

Both the shape model (hunyuan3d-dit-v2-1) and the paint model (hunyuanpaintpbr)
live in the same repo, so one snapshot covers everything the server loads.
Run via the container entrypoint before the server starts.
"""

import os
import sys


def main() -> int:
    repo = os.environ.get("HY3D_MODEL_REPO", "tencent/Hunyuan3D-2.1")
    token = os.environ.get("HF_TOKEN") or None
    cache = os.environ.get("HF_HOME", "~/.cache/huggingface")
    # Optional scoping: HY3D_PREFETCH_PATTERNS="hunyuan3d-dit-v2-1/*,hunyuan3d-paintpbr-v2-1/*"
    raw_patterns = os.environ.get("HY3D_PREFETCH_PATTERNS", "").strip()
    allow_patterns = [p.strip() for p in raw_patterns.split(",") if p.strip()] or None

    from huggingface_hub import snapshot_download

    # Presence check first: local_files_only does NO network and returns the cached
    # snapshot path only if every file is already there. If so, skip the download
    # entirely — we only hit the network when weights are actually missing.
    try:
        path = snapshot_download(repo, allow_patterns=allow_patterns, local_files_only=True)
        print(f"[prefetch] {repo} already present at {path}; skipping download.", flush=True)
        return 0
    except Exception:  # noqa: BLE001  (any miss -> fall through to download)
        pass

    print(f"[prefetch] {repo} weights missing; downloading into {cache} "
          f"(first run is tens of GB) ...", flush=True)
    try:
        path = snapshot_download(repo, token=token, allow_patterns=allow_patterns)
        print(f"[prefetch] {repo} ready at {path}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        # Non-fatal: the server's --preload / lazy load retries the download. Do not
        # block startup just because the prefetch hit a transient network error.
        print(f"[prefetch] WARNING: could not prefetch {repo}: {e}", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
