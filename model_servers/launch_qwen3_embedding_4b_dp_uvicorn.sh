#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qwen3-Embedding-4B SPMD data-parallel launcher for QB2 (4 Blackhole chips).
# Thin wrapper so per-model tuning cannot drift from the shared launcher.
#
#   cd ~/tt-xla && source venv/activate && \
#     ~/scripts/model_servers/launch_qwen3_embedding_4b_dp_uvicorn.sh
exec "$(dirname "$0")/launch_embedding_dp_uvicorn.sh" --model "Qwen3-Embedding-4B" "$@"
