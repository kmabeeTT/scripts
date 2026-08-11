#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# End-to-end SPMD validation for all three forge embedding models on QB2:
# for each model -- launch the server, wait for ready, run the release flow
# (downsampled evals + benchmarks, no spec tests), tear down, record result.
#
# Sequential by design: only one tt-xla process can own the chips at a time, and
# each model takes all four under SPMD DP.
#
# Must run from a tt-xla venv:
#   cd ~/tt-xla && source venv/activate && \
#     ~/scripts/model_servers/run_all_embedding_releases.sh
#
# Options:
#   --models "A B C"   default all three
#   --mode MODE        ci-nightly (default, 10% evals) | smoke-test (5%)
#   --port P           default 8021
#   --warmup-timeout S wait for /tt-liveness 200 (default 2700 -- first run
#                      compiles the whole padding ladder, which is slow)
set -eo pipefail

MODELS="Qwen3-Embedding-0.6B bge-m3 Qwen3-Embedding-4B"
MODE=ci-nightly
PORT=8021
WARMUP_TIMEOUT=2700
OUT_DIR=${OUT_DIR:-$HOME/tt-inference-server/embedding_spmd_$(date +%Y%m%d_%H%M%S)}

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift 2;;  --models=*) MODELS="${1#*=}"; shift;;
    --mode) MODE="$2"; shift 2;;      --mode=*) MODE="${1#*=}"; shift;;
    --port) PORT="$2"; shift 2;;      --port=*) PORT="${1#*=}"; shift;;
    --warmup-timeout) WARMUP_TIMEOUT="$2"; shift 2;; --warmup-timeout=*) WARMUP_TIMEOUT="${1#*=}"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

HERE=$(dirname "$0")
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"
: > "$SUMMARY"

teardown() {
  pkill -9 -f "uvicorn main:app" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  fuser -k "${PORT}"/tcp 2>/dev/null || true
  sleep 8
  # A killed engine can leave a chip held; without this the next launch cannot
  # acquire the device and fails confusingly.
  for d in /dev/tenstorrent/*; do
    # `|| true` is load-bearing: fuser exits non-zero when nothing holds the
    # device, and with `set -e` + pipefail a bare assignment from a failing
    # command substitution takes that status and kills the script.
    holder=$(fuser "$d" 2>/dev/null | tr -d ' ') || true
    [ -n "$holder" ] && kill -9 $holder 2>/dev/null || true
  done
  sleep 4
}

trap teardown EXIT

for MODEL in $MODELS; do
  SLUG=$(echo "$MODEL" | tr 'A-Z.' 'a-z_')
  SRV_LOG="$OUT_DIR/server_${SLUG}.log"
  REL_LOG="$OUT_DIR/release_${SLUG}.log"

  echo "==================================================================="
  echo "[$MODEL] tearing down any prior server"
  teardown

  echo "[$MODEL] launching SPMD DP server -> $SRV_LOG"
  LOG_DIR="$OUT_DIR" TT_INFERENCE_SERVER_ROOT=${TT_INFERENCE_SERVER_ROOT:-$HOME/tt-inference-server-emb} \
    nohup "$HERE/launch_embedding_dp_uvicorn.sh" --model "$MODEL" --port "$PORT" \
    > "$SRV_LOG" 2>&1 &

  echo "[$MODEL] waiting up to ${WARMUP_TIMEOUT}s for /tt-liveness 200 ..."
  ready=0
  waited=0
  while [ "$waited" -lt "$WARMUP_TIMEOUT" ]; do
    code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/tt-liveness" || true)
    if [ "$code" = "200" ]; then ready=1; break; fi
    # Bail out early if the engine died rather than burning the whole timeout.
    if ! pgrep -f "uvicorn main:app" >/dev/null 2>&1; then
      echo "[$MODEL] ⛔ server process gone after ${waited}s"
      break
    fi
    sleep 15; waited=$((waited+15))
  done

  if [ "$ready" != "1" ]; then
    echo "FAIL  $MODEL  server never became ready (${waited}s); see $SRV_LOG" | tee -a "$SUMMARY"
    continue
  fi
  echo "[$MODEL] ready after ${waited}s"

  echo "[$MODEL] running release ($MODE) -> $REL_LOG"
  if LOG_DIR="$OUT_DIR" "$HERE/run_release_embedding_forge.sh" \
       --model "$MODEL" --port "$PORT" --mode "$MODE" > "$REL_LOG" 2>&1; then
    echo "PASS  $MODEL  ready=${waited}s" | tee -a "$SUMMARY"
  else
    echo "FAIL  $MODEL  release exited non-zero; see $REL_LOG" | tee -a "$SUMMARY"
  fi
done

echo "==================================================================="
echo "SUMMARY ($OUT_DIR)"
cat "$SUMMARY"
