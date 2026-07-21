#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Falcon3-7B-Instruct single-chip forge LLM launcher (bare-metal, direct
# uvicorn -- no docker), mirroring the CI/production config in
# workflows/model_specs/dev/cnn.yaml (forge P150 spec, verified against it
# directly): b32, 32768 ctx, chunk 1024, gmu 0.35, bfp8 weights+KV, opt=1,
# device sampling, trace, b1-prefill. Pinned to chip 0.
#
# Must run from a tt-xla venv (TT_METAL_HOME/mesh descriptor path resolve
# from $(pwd), venv/activate needs it too):
#   cd ~/tt-xla && source venv/activate && \
#     ~/scripts/model_servers/launch_falcon3_7b_instruct_uvicorn.sh
#
# TT_INFERENCE_SERVER_ROOT selects which tt-inference-server checkout's
# tt-media-server to launch (default: tt-inference-server-2).
set -eo pipefail  # NOT -u: venv/activate references vars unset until sourced

TT_INFERENCE_SERVER_ROOT=${TT_INFERENCE_SERVER_ROOT:-$HOME/tt-inference-server-2}

export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$(pwd)/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}"
export MODEL=${MODEL:-Falcon3-7B-Instruct}
export DEVICE=${DEVICE:-p150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
export IS_GALAXY=${IS_GALAXY:-False}

# Falcon3-7B FORGE spec (cnn.yaml). Overridable so we can sweep chunk/b1 knobs.
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-32768}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-1024}
export MIN_NUM_SEQS=${MIN_NUM_SEQS:-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD:-16}
unset FP32_DEST_ACC_EN

export DEVICE_IDS=${DEVICE_IDS:-'(0)'}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}
PORT=${PORT:-8019}

DEVICE_IDS_SAFE=$(echo "$DEVICE_IDS" | tr -dc '0-9,')
LOG_DIR="$HOME/scripts"
LOG="$LOG_DIR/launch_falcon3_7b_instruct_uvicorn_dev${DEVICE_IDS_SAFE}_p${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Falcon3-7B-Instruct server (uvicorn): DEVICE_IDS=$DEVICE_IDS PORT=$PORT ctx=$MAX_MODEL_LENGTH b=$MAX_NUM_SEQS gmu=$GPU_MEMORY_UTILIZATION chunk=$PREFILL_CHUNK_SIZE min_num_seqs=$MIN_NUM_SEQS threshold=$PREFILL_BATCH_THRESHOLD"
echo "tt-inference-server root=$TT_INFERENCE_SERVER_ROOT"
echo "log=$LOG"

cd "$TT_INFERENCE_SERVER_ROOT/tt-media-server"
uvicorn main:app --lifespan on --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL" |& tee "$LOG"
