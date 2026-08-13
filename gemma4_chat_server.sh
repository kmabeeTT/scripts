#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Launches the interactive Gemma4 12B chat server on this QB2 box (4x p150, 1x4
# mesh -- the default): tt-metal's models/demos/gemma4/demo/serve_interactive.py,
# on branch kmabee/gemma4-interactive-server. Not vLLM -- single request at a
# time, no incremental KV reuse across turns; fine at chat latency.
#
# For a single-chip (1x1) run instead, see gemma4_chat_server_1chip.sh.
#
# All defaults (model, HF/TT cache paths, port, mesh) are baked into the server
# file itself -- override any of them by exporting the same env var before
# calling this script, e.g. HF_MODEL, TT_CACHE_PATH, PORT, MESH_DEVICE,
# GEMMA4_MAX_SEQ_LEN.
#
# Usage:
#   ~/scripts/gemma4_chat_server.sh
# Then, in another shell:
#   ~/scripts/client_demo.sh 128

set -e

TT_METAL_DIR=${TT_METAL_DIR:-/home/kmabee/tt-metal}
cd "$TT_METAL_DIR"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "kmabee/gemma4-interactive-server" ]; then
    echo "warning: $TT_METAL_DIR is on branch '$BRANCH', not kmabee/gemma4-interactive-server -- serve_interactive.py may not exist here." >&2
fi

export PYTHONPATH="$TT_METAL_DIR"
export TT_METAL_HOME="$TT_METAL_DIR"
source "$TT_METAL_DIR/python_env/bin/activate"

exec python3 -m pytest -sq models/demos/gemma4/demo/serve_interactive.py::test_serve
