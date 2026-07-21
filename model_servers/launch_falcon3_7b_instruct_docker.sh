#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Falcon3-7B-Instruct: launch via run.py --docker-server, matching the
# CI/production config (workflows/model_specs/dev/cnn.yaml, forge P150 spec):
# b32, 32768 ctx, chunk 1024, gmu 0.35, bfp8 weights+KV, opt=1, device sampling,
# trace, b1-prefill. No --vllm-override-args -- letting run.py resolve
# cnn.yaml's default spec unmodified is what reproduces CI exactly (confirmed
# against https://github.com/tenstorrent/tt-shield/actions/runs/29803259960).
#
# Needs TT_VISIBLE_DEVICES + TT_MESH_GRAPH_DESC_PATH set in the
# TT_INFERENCE_SERVER_ROOT/.env file (--env-file, gitignored, host-local) --
# the container's own auto-mesh-descriptor logic (tt-media-server/utils/
# runner_utils.py) only fires for Whisper/SpeechT5-TTS runners, not
# vllm_forge/LLM, so it must be supplied externally. Example .env lines
# (paths are INSIDE the container, not the host -- confirmed via
# `docker inspect <image>` + `find` inside the image):
#   TT_VISIBLE_DEVICES=0
#   TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/app/server/venv-worker/lib/python3.12/site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
#
# Run from anywhere; no tt-xla venv needed (docker-server mode runs outside
# any PJRT venv on the host).
set -euo pipefail

TT_INFERENCE_SERVER_ROOT=${TT_INFERENCE_SERVER_ROOT:-$HOME/tt-inference-server-2}
IMAGE=${IMAGE:-ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:35a716ccced50430fe24c22c6e5258ee48b9a89e_e89c4ce_88235779891}
MODEL=${MODEL:-Falcon3-7B-Instruct}
DEVICE_ID=${DEVICE_ID:-0}
SERVICE_PORT=${SERVICE_PORT:-8010}

LOG_DIR="$HOME/scripts"
LOG="$LOG_DIR/launch_falcon3_7b_instruct_docker_dev${DEVICE_ID}_p${SERVICE_PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] starting: $MODEL (CI-matched docker config) device=$DEVICE_ID port=$SERVICE_PORT"
echo "[$(date)] tt-inference-server root=$TT_INFERENCE_SERVER_ROOT"
echo "[$(date)] log=$LOG"

cd "$TT_INFERENCE_SERVER_ROOT"
python3 run.py \
  --model "$MODEL" \
  --tt-device p150 \
  --engine forge \
  --impl forge-vllm-plugin \
  --workflow server \
  --docker-server \
  --no-auth \
  --device-id "$DEVICE_ID" \
  --service-port "$SERVICE_PORT" \
  --override-docker-image "$IMAGE" \
  |& tee "$LOG"

echo "[$(date)] server kicked off"
