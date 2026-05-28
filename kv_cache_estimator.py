#!/usr/bin/env python3
"""
Estimate on-device DRAM usage for a forge LLM serving config on P150 (32 GB DRAM).

Breaks the budget into three contributors:
  - Weights    = param_count × bytes_per_weight   (bfloat16 = 2 B; bfp8 = 1 B)
  - KV pool    = max(batch_size, max_concurrency) × max_seq_len × per_token_KV
                 (per_token_KV = layers × kv_heads × head_dim × 2 × 2 B, GQA-aware)
  - Activations = hidden_size × batch_size × max_seq_len × 2 × layers_concurrent
                 (heuristic: ~layers_concurrent intermediate buffers live at once)

max_concurrency is an upper bound on how many simultaneous full-length seqs
fit in the KV pool — it's the value vLLM logs as "Maximum concurrency for N
tokens per request: Mx". The effective pool sizing is
`min(max_concurrency, batch_size)`: capping at batch_size avoids reserving
slots vLLM can't use. Setting max_concurrency < batch_size makes vLLM preempt
when too many requests run at full length, but the pool stays small (frees
DRAM for activations).

Suggested gpu_memory_utilization (gmu) is what to pass to vLLM so the KV pool
ends up sized to fit the chosen (batch_size, max_seq_len, max_concurrency):
    gmu = kv_pool_gb / dram_gb (rounded up to 2 decimal places).
This assumes vLLM is given the real device DRAM as its total_memory_size — i.e.
the post-tt-xla-fix world, or our `TT_KV_POOL_GB=<dram_gb>` workaround.

Examples (your 5 forge LLMs on P150):
    python scripts/kv_cache_estimator.py --model Falcon3-7B-Instruct
    python scripts/kv_cache_estimator.py --model Llama-3.1-8B-Instruct
    python scripts/kv_cache_estimator.py --model Llama-3.2-3B-Instruct
    python scripts/kv_cache_estimator.py --model Qwen3-4B
    python scripts/kv_cache_estimator.py --model Qwen3-8B

Single-config detail mode:
    python scripts/kv_cache_estimator.py --model Llama-3.1-8B-Instruct \
        --batch-size 4 --seq-len 16384 --max-concurrency 4

    (--max-concurrency is capped at batch_size internally — setting it higher
     wastes pool slots vLLM can't use.)

Custom sweep:
    python scripts/kv_cache_estimator.py --model Qwen3-8B \
        --batch-sizes 1,2,4,8,16 --seq-lens 2048,4096,8192,16384,32768

Tune the activation heuristic to match empirical OOM data (default 10 ≈ matches
the v5/v6 b×seq=128K OOM observed in this session):
    --activation-layers 10
"""

import argparse
import math
import sys

# ---------------------------------------------------------------------------
# Model architecture registry. Cross-check with HF config.json when adding new
# entries; param_count_b is the marketed billion-parameter count.
# ---------------------------------------------------------------------------
MODELS = {
    "Falcon3-7B-Instruct": dict(
        num_layers=28,
        num_kv_heads=8,
        head_dim=256,        # Falcon3 uses head_dim=256 (different from Llama)
        hidden_size=3072,
        param_count_b=7.46,
    ),
    "Llama-3.1-8B-Instruct": dict(
        num_layers=32,
        num_kv_heads=8,      # GQA: 32 Q heads / 8 KV heads
        head_dim=128,
        hidden_size=4096,
        param_count_b=8.03,
    ),
    "Llama-3.2-3B-Instruct": dict(
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=3072,
        param_count_b=3.21,
    ),
    "Qwen3-4B": dict(
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=2560,
        param_count_b=4.02,
    ),
    "Qwen3-8B": dict(
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        param_count_b=8.19,
    ),
}

P150_DRAM_GB = 32.0
BYTES_BF16 = 2
BYTES_BFP8 = 1


def suggest_gmu(kv_pool_gb, dram_gb):
    """Round kv_pool_gb / dram_gb UP to nearest 0.01 (conservative)."""
    raw = kv_pool_gb / dram_gb
    return math.ceil(raw * 100) / 100


def estimate(cfg, batch_size, seq_len, max_concurrency, weight_dtype, activation_layers, dram_gb):
    """Return a dict of GB usage by component + fit + suggested gmu."""

    # 1. Weights
    bytes_per_weight = BYTES_BFP8 if weight_dtype == "bfp8" else BYTES_BF16
    weights_gb = cfg["param_count_b"] * 1e9 * bytes_per_weight / (1024**3)

    # 2. Per-token KV (KV dtype follows model dtype = bfloat16 in vLLM default)
    bytes_per_kv_token = cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"] * 2 * BYTES_BF16

    # 3. KV pool: sized for min(max_concurrency, batch_size) full-length seqs.
    #    Capping at batch_size avoids reserving pool slots vLLM can't use
    #    (e.g. b=1 with max_concurrency=4 = 3 wasted slots). When
    #    max_concurrency < batch_size, vLLM preempts at runtime rather than
    #    OOMing at startup — trade-off lives at runtime.
    effective_concurrency = min(max_concurrency, batch_size)
    kv_pool_gb = (effective_concurrency * seq_len * bytes_per_kv_token) / (1024**3)

    # 4. Activations heuristic
    activations_gb = (cfg["hidden_size"] * batch_size * seq_len * BYTES_BF16 * activation_layers) / (1024**3)

    total_gb = weights_gb + kv_pool_gb + activations_gb

    # 5. vLLM-reported "Maximum concurrency for {seq_len} tokens per request"
    vllm_concurrency = (kv_pool_gb * (1024**3)) / (seq_len * bytes_per_kv_token)

    # 6. Suggested gmu = kv_pool_gb / dram_gb (assumes vLLM sees dram_gb as total)
    suggested_gpu_mem_util = suggest_gmu(kv_pool_gb, dram_gb)

    return dict(
        weights_gb=weights_gb,
        kv_pool_gb=kv_pool_gb,
        activations_gb=activations_gb,
        total_gb=total_gb,
        vllm_concurrency=vllm_concurrency,
        bytes_per_kv_token=bytes_per_kv_token,
        suggested_gpu_mem_util=suggested_gpu_mem_util,
    )


def print_detail(model_name, batch_size, seq_len, max_concurrency, weight_dtype, activation_layers, dram_gb):
    cfg = MODELS[model_name]
    e = estimate(cfg, batch_size, seq_len, max_concurrency, weight_dtype, activation_layers, dram_gb)

    print(f"\n=== {model_name}  b={batch_size}  seq={seq_len}  max_concurrency={max_concurrency} ===")
    print(f"  Per-token KV:       {e['bytes_per_kv_token']/1024:.1f} KB"
          f"   (layers={cfg['num_layers']} × kv_heads={cfg['num_kv_heads']} × head_dim={cfg['head_dim']} × 2 × 2B)")
    print()
    print(f"  Weights ({weight_dtype}):    {e['weights_gb']:6.2f} GB")
    effective_conc = min(max_concurrency, batch_size)
    print(f"  KV pool:            {e['kv_pool_gb']:6.2f} GB"
          f"   ({effective_conc} × {seq_len} × {e['bytes_per_kv_token']/1024:.1f} KB"
          f"; effective_concurrency = min(max_concurrency={max_concurrency}, batch_size={batch_size}))")
    print(f"  Activations:        {e['activations_gb']:6.2f} GB"
          f"   (hidden={cfg['hidden_size']} × b × seq × 2B × {activation_layers} layers; heuristic)")
    print(f"  -------------------- ------")
    print(f"  Total:              {e['total_gb']:6.2f} GB")
    print(f"  DRAM budget:        {dram_gb:6.2f} GB")
    headroom = dram_gb - e['total_gb']
    fits = "✅ FITS" if headroom >= 0 else "❌ OVER BUDGET"
    print(f"  Headroom:           {headroom:+6.2f} GB   {fits}")
    print()
    print(f"  vLLM 'Maximum concurrency for {seq_len} tokens per request': {e['vllm_concurrency']:.2f}x")
    if e['vllm_concurrency'] < batch_size:
        print(f"  ⚠  vllm_concurrency ({e['vllm_concurrency']:.1f}x) < batch_size ({batch_size}) — expect preemption "
              f"if all requests are at full length.")
    print()
    if headroom >= 0:
        print(f"  Suggested GPU_MEMORY_UTILIZATION: {e['suggested_gpu_mem_util']:.2f}"
              f"   (= {e['kv_pool_gb']:.2f} GB KV pool / {dram_gb:.0f} GB DRAM, rounded up)")
        print(f"    → vLLM total_memory_size must = {dram_gb:.0f} GB; in today's stack pass "
              f"TT_KV_POOL_GB={int(dram_gb)} + GPU_MEMORY_UTILIZATION={e['suggested_gpu_mem_util']:.2f}")
    else:
        print(f"  Suggested GPU_MEMORY_UTILIZATION: N/A (config doesn't fit)")


def print_matrix(model_name, batch_sizes, seq_lens, max_concurrency, weight_dtype, activation_layers, dram_gb):
    cfg = MODELS[model_name]
    print(f"\n=== {model_name}  (max_concurrency={max_concurrency}, weight_dtype={weight_dtype}, "
          f"activation_layers={activation_layers}, DRAM={dram_gb:.0f}GB) ===\n")
    print(f"  Per-token KV: {(cfg['num_layers'] * cfg['num_kv_heads'] * cfg['head_dim'] * 2 * BYTES_BF16) / 1024:.1f} KB"
          f"   ({cfg['num_layers']} layers × {cfg['num_kv_heads']} KV heads × {cfg['head_dim']} head_dim × 2 K/V × 2B bf16)")
    weights = cfg['param_count_b'] * 1e9 * (BYTES_BFP8 if weight_dtype == "bfp8" else BYTES_BF16) / (1024**3)
    print(f"  Weights:      {weights:.2f} GB ({cfg['param_count_b']:.2f}B params × {1 if weight_dtype=='bfp8' else 2}B)\n")

    header = f"  {'batch':>5}  " + "  ".join(f"{s:>8}" for s in seq_lens)

    # Table 1: Total DRAM (GB)
    print(f"  Total DRAM (GB) — ✅ fits {dram_gb:.0f}GB, ❌ exceeds")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in batch_sizes:
        cells = []
        for s in seq_lens:
            e = estimate(cfg, b, s, max_concurrency, weight_dtype, activation_layers, dram_gb)
            mark = "✅" if e['total_gb'] <= dram_gb else "❌"
            cells.append(f"{e['total_gb']:6.1f}{mark}")
        print(f"  b={b:>3}  " + "  ".join(cells))

    # Table 2: vLLM concurrency
    print()
    print(f"  vLLM 'Maximum concurrency Nx' (pool / (seq × KV/tok))")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in batch_sizes:
        cells = []
        for s in seq_lens:
            e = estimate(cfg, b, s, max_concurrency, weight_dtype, activation_layers, dram_gb)
            warn = "⚠" if e['vllm_concurrency'] < b else " "
            cells.append(f"{e['vllm_concurrency']:6.1f}x{warn}")
        print(f"  b={b:>3}  " + "  ".join(cells))

    # Table 3: Suggested GPU_MEMORY_UTILIZATION
    print()
    print(f"  Suggested GPU_MEMORY_UTILIZATION (= KV pool / {dram_gb:.0f} GB DRAM, rounded up; N/A if doesn't fit)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in batch_sizes:
        cells = []
        for s in seq_lens:
            e = estimate(cfg, b, s, max_concurrency, weight_dtype, activation_layers, dram_gb)
            if e['total_gb'] <= dram_gb:
                cells.append(f"  {e['suggested_gpu_mem_util']:6.2f}  ")
            else:
                cells.append(f"  {'N/A':>6}  ")
        print(f"  b={b:>3}  " + "  ".join(cells))

    print()
    print("  ⚠ = vllm_concurrency < batch_size; expect preemption at full-length workloads.")
    print(f"  Pool sizing uses min(max_concurrency={max_concurrency}, batch_size) — values above "
          f"batch_size are capped, so the same max_concurrency might produce different vllm_concurrency "
          f"across rows.")
    print("  Tip: max_concurrency=1 keeps the KV pool minimal at the cost of preempting any "
          "concurrent full-length req.")
    print(f"  GPU_MEMORY_UTILIZATION values assume vLLM sees full {dram_gb:.0f} GB as total_memory_size "
          "(i.e. post-tt-xla fix OR TT_KV_POOL_GB workaround).")


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", required=True, choices=sorted(MODELS.keys()),
                   help="Model name (one of the 5 forge LLMs)")
    p.add_argument("--dram-gb", type=float, default=P150_DRAM_GB,
                   help=f"Device DRAM in GB (default {P150_DRAM_GB} for P150)")
    p.add_argument("--weight-dtype", choices=["bf16", "bfp8"], default="bfp8",
                   help="On-device weight dtype (default bfp8 — forge appears to pack to ~half size).")
    p.add_argument("--max-concurrency", type=int, default=1,
                   help="Upper bound on simultaneous full-length seqs in the KV pool (default 1 = "
                        "minimal pool, preempt to fit). Effective value is min(max_concurrency, "
                        "batch_size) — setting above batch_size wastes pool slots vLLM can't use. "
                        "Setting < batch_size makes vLLM preempt at runtime.")
    p.add_argument("--activation-layers", type=int, default=10,
                   help="Heuristic: # of intermediate-buffer layers live at once (default 10).")

    # Single-config mode
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)

    # Sweep mode (default if no single-config)
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes for sweep (default 1,2,4,8,16,32)")
    p.add_argument("--seq-lens", default="2048,4096,8192,16384,32768,65536",
                   help="Comma-separated seq lens for sweep (default 2k..64k)")

    args = p.parse_args()

    if args.batch_size is not None and args.seq_len is not None:
        print_detail(args.model, args.batch_size, args.seq_len, args.max_concurrency,
                     args.weight_dtype, args.activation_layers, args.dram_gb)
    else:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        seq_lens = [int(x) for x in args.seq_lens.split(",")]
        print_matrix(args.model, batch_sizes, seq_lens, args.max_concurrency,
                     args.weight_dtype, args.activation_layers, args.dram_gb)


if __name__ == "__main__":
    main()
