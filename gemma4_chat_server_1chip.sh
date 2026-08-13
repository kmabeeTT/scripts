#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Same server as gemma4_chat_server.sh, but on a single p150 chip (1x1 mesh)
# instead of this QB2's default 4-chip (1x4) mesh -- same model, same code,
# just MESH_DEVICE=P150 plus a personal TT_CACHE_PATH.
#
# Two things differ from the 4-chip default, both required:
#   - MESH_DEVICE=P150 selects the 1x1 mesh (serve_interactive.py already
#     supports this out of the box).
#   - TT_CACHE_PATH points at a personal writable directory instead of the
#     shared /mnt/models/huggingface/tt_cache/ -- that directory is owned
#     root:ubuntu and kmabee has read-only access there. It already has a
#     TP=4-sharded weight cache (from 4-chip runs) but no TP=1 cache, and
#     writing a fresh one fails with EACCES. ~/gemma4_tt_cache holds a TP=1
#     cache of the full 48-layer model already converted from the (still
#     shared, read-only) HF safetensors -- first run elsewhere will convert
#     and populate this on its own if it's ever missing/cleared.
#
# Perf on this box: TTFT ~86-90ms, ~16-17 tok/s/user (roughly half of the
# 4-chip mesh's ~35 tok/s -- TP=4 gives ~2x from parallelism, not 4x, due to
# the usual CCL/communication overhead).
#
# Usage:
#   ~/scripts/gemma4_chat_server_1chip.sh
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
export MESH_DEVICE=${MESH_DEVICE:-P150}
export TT_CACHE_PATH=${TT_CACHE_PATH:-/home/kmabee/gemma4_tt_cache/google--gemma-4-12B-it}
source "$TT_METAL_DIR/python_env/bin/activate"

exec python3 -m pytest -sq models/demos/gemma4/demo/serve_interactive.py::test_serve
