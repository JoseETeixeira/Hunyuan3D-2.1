#!/usr/bin/env bash
# Detect the host NVIDIA GPU's CUDA compute capability and build the image with the
# native CUDA extensions (custom_rasterizer + mesh painter) compiled for exactly
# that architecture, then start the stack. Docker build has no GPU access, so the
# arch must be detected here on the host and passed in as a build arg.
#
# Usage:
#   ./build.sh                 # -> docker compose up --build
#   ./build.sh build           # -> docker compose build
#   ./build.sh up -d --build   # forward any compose args
set -euo pipefail

map_name_to_arch() {
  local n; n=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "$n" in
    *"rtx 50"*|*5090*|*5080*|*5070*|*5060*|*b200*|*blackwell*)                 echo "12.0" ;;  # Blackwell
    *"rtx 40"*|*4090*|*4080*|*4070*|*4060*|*ada*|*l40*|*l4*)                   echo "8.9"  ;;  # Ada Lovelace
    *a100*|*a800*)                                                             echo "8.0"  ;;  # Ampere (datacenter)
    *"rtx 30"*|*3090*|*3080*|*3070*|*3060*|*a6000*|*a40*|*a10*|*ampere*)       echo "8.6"  ;;  # Ampere (consumer)
    *"rtx 20"*|*2080*|*2070*|*2060*|*"titan rtx"*|*t4*|*"gtx 16"*|*1660*|*1650*|*turing*) echo "7.5" ;;  # Turing
    *v100*)                                                                    echo "7.0"  ;;  # Volta
    *"gtx 10"*|*1080*|*1070*|*1060*|*pascal*)                                  echo "6.1"  ;;  # Pascal
    *) echo "" ;;
  esac
}

detect_archs() {
  local caps names n a; local -a list=()
  # Primary: nvidia-smi reports compute capability directly (driver 510+), e.g. "8.6", "12.0".
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
  while IFS= read -r c; do
    c=$(echo "$c" | xargs)
    [[ "$c" =~ ^[0-9]+\.[0-9]+$ ]] && list+=("$c")
  done <<< "$caps"
  if [ ${#list[@]} -eq 0 ]; then
    # Fallback for older drivers without compute_cap: map GPU model name -> arch.
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    while IFS= read -r n; do
      [ -z "$n" ] && continue
      a=$(map_name_to_arch "$n")
      [ -n "$a" ] && list+=("$a")
    done <<< "$names"
  fi
  [ ${#list[@]} -eq 0 ] && return 0
  printf '%s\n' "${list[@]}" | awk 'NF' | sort -u | paste -sd';' -
}

ARCH=$(detect_archs)
if [ -z "$ARCH" ]; then
  echo "[build] WARNING: could not detect GPU compute capability; defaulting to 12.0 (RTX 50-series)." >&2
  ARCH="12.0"
fi
export TORCH_CUDA_ARCH_LIST="$ARCH"
echo "[build] Detected GPU arch(s): $TORCH_CUDA_ARCH_LIST"

if [ "$#" -eq 0 ]; then
  set -- up --build
fi
echo "[build] docker compose $* (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
exec docker compose "$@"
