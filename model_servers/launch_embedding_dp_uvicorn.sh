#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Generic forge EMBEDDING model launcher, SPMD data-parallel over every visible
# chip (bare-metal uvicorn, no docker). Built for QB2 (2x P300 = 4 Blackhole
# chips) as the scale-down proxy for the 32-chip BH galaxy: same code path,
# dp_size 4 instead of 32.
#
# ONE worker takes all chips as a group and the forge plugin replicates the model
# across them (ENABLE_DATA_PARALLEL=true). Deliberately NOT the per-chip layout:
# pinning a single chip aborts in device init on multi-chip Blackhole
# (tt-xla#5521 -- the tt-mlir op-model mock device re-applies TT_VISIBLE_DEVICES
# after discovery already renumbered the chips from 0).
#
# Must run from a tt-xla venv (TT_METAL_HOME and venv/activate resolve from $(pwd)):
#   cd ~/tt-xla && source venv/activate && \
#     ~/scripts/model_servers/launch_embedding_dp_uvicorn.sh --model Qwen3-Embedding-4B
#
# Usage:
#   launch_embedding_dp_uvicorn.sh [--model NAME] [--port P] [--batch N] [--seq N]
#
#   --model  Qwen3-Embedding-0.6B (default) | Qwen3-Embedding-4B | bge-m3
#   --port   default 8021
#   --batch  GLOBAL max_num_seqs across all replicas (default 32 = 8/chip on QB2)
#   --seq    max_model_length (default 128)
set -eo pipefail  # NOT -u: venv/activate references vars unset until sourced

MODEL_ARG=Qwen3-Embedding-0.6B
PORT=${PORT:-8021}
BATCH=${MAX_NUM_SEQS:-32}
SEQ=${VLLM__MAX_MODEL_LENGTH:-128}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL_ARG="$2"; shift 2;; --model=*) MODEL_ARG="${1#*=}"; shift;;
    --port) PORT="$2"; shift 2;;       --port=*) PORT="${1#*=}"; shift;;
    --batch) BATCH="$2"; shift 2;;     --batch=*) BATCH="${1#*=}"; shift;;
    --seq) SEQ="$2"; shift 2;;         --seq=*) SEQ="${1#*=}"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# MODEL_RUNNER must be explicit. Without it, settings._set_config_overrides()
# picks the FIRST runner in INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP whose model
# set contains this name -- and "bge-m3" belongs to two specs (a forge one and an
# older non-forge one), so it silently resolves to the wrong runner, finds no
# (runner, p300x2) config, and falls back to tt-sdxl-trace:
#   ImportError: Failed to load model runner tt-sdxl-trace: No module named 'models'
case "$MODEL_ARG" in
  Qwen3-Embedding-0.6B) DEFAULT_RUNNER=vllmforge_qwen_embedding_0_6b ;;
  Qwen3-Embedding-4B)   DEFAULT_RUNNER=vllmforge_qwen_embedding ;;
  bge-m3)               DEFAULT_RUNNER=vllmforge_bge_m3 ;;
  *) echo "⛔ unsupported --model '$MODEL_ARG' (expect Qwen3-Embedding-0.6B | Qwen3-Embedding-4B | bge-m3)" >&2; exit 2;;
esac
export MODEL_RUNNER=${MODEL_RUNNER:-$DEFAULT_RUNNER}

TT_INFERENCE_SERVER_ROOT=${TT_INFERENCE_SERVER_ROOT:-$HOME/tt-inference-server-emb}

export MODEL="$MODEL_ARG"
export DEVICE=${DEVICE:-p300x2}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
# Real checkout, not $(pwd)/tt-metal -- that path does not exist under ~/tt-xla,
# and TT_METAL_CACHE is composed from TT_METAL_HOME, so a bogus value fails late.
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal}
export IS_GALAXY=${IS_GALAXY:-False}

# No TT_MESH_GRAPH_DESC_PATH and no per-chip pin: SPMD DP wants every visible
# chip and the plugin derives the mesh itself. Setting either is what breaks it.
export DEVICE_IDS=${DEVICE_IDS:-'(0,1,2,3)'}
export ENABLE_DATA_PARALLEL=${ENABLE_DATA_PARALLEL:-true}

# REQUIRED whenever max_num_seqs > 1: settings.py auto-enables the dynamic
# batcher, which dispatches via device_runner._run_async() -- not implemented by
# the embedding runner (it uses the classic is_request_batchable + run() path).
# Without this every request 500s with:
#   'VLLMForgeEmbeddingQwenRunner' object has no attribute '_run_async'
export USE_DYNAMIC_BATCHER=${USE_DYNAMIC_BATCHER:-false}

# Global batch, split across the replicas. Must be >1 (or the plugin silently
# disables DP) and a multiple of the chip count (or batches pad with zero rows).
export MAX_NUM_SEQS="$BATCH"
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-$BATCH}
export VLLM__MAX_NUM_SEQS="$BATCH"
export VLLM__MAX_MODEL_LENGTH="$SEQ"
export VLLM__MAX_NUM_BATCHED_TOKENS=${VLLM__MAX_NUM_BATCHED_TOKENS:-$((BATCH * SEQ))}
export VLLM__MIN_CONTEXT_LENGTH=${VLLM__MIN_CONTEXT_LENGTH:-32}

export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
# OFF: trace fails to compile on the pooling path even with DP enabled --
#   'ttnn.capture_or_execute_trace' op All output tensors of trace function
#   must be on device ... "ttnn.from_device"(%arg2)
# Perf knob, not correctness. Chase separately.
export ENABLE_TRACE=${ENABLE_TRACE:-false}
# ON by default in the runner: with const-eval off, weight prep re-runs on every
# call and the served path is ~153x slower.
export ENABLE_CONST_EVAL=${ENABLE_CONST_EVAL:-true}
export DEFAULT_THROTTLE_LEVEL=${DEFAULT_THROTTLE_LEVEL:-0}
export TT_METAL_INSPECTOR_RPC=${TT_METAL_INSPECTOR_RPC:-0}

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}

SLUG=$(echo "$MODEL_ARG" | tr 'A-Z.' 'a-z_')
LOG_DIR=${LOG_DIR:-$HOME/scripts}
LOG="$LOG_DIR/launch_${SLUG}_dp_p${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting $MODEL_ARG SPMD DP server (uvicorn): DEVICE=$DEVICE DEVICE_IDS=$DEVICE_IDS PORT=$PORT"
echo "  dp=all visible chips, global batch=$VLLM__MAX_NUM_SEQS, seq=$VLLM__MAX_MODEL_LENGTH,"
echo "  gmu=$GPU_MEMORY_UTILIZATION, trace=$ENABLE_TRACE, const_eval=$ENABLE_CONST_EVAL"
echo "tt-inference-server root=$TT_INFERENCE_SERVER_ROOT"
echo "log=$LOG"

cd "$TT_INFERENCE_SERVER_ROOT/tt-media-server"
uvicorn main:app --lifespan on --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL" |& tee "$LOG"
