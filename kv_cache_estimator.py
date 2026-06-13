#!/usr/bin/env python3
"""
Estimate on-device DRAM feasibility for an LLM serving config on Tenstorrent
(P150 = 31.88 GiB usable DRAM), and tell you what (batch, seq_len, gmu) will FIT.

============================================================================
WHAT CHANGED (v2 — calibrated to June-2026 DRAM profiling + chunked prefill)
============================================================================
The old activation model was `hidden × batch × seq × 2B × 10`, which gave
~171 GiB for Llama-8B b32×64K and bore no relation to reality. We've since done
Operation-level DRAM profiling and learned the real contributors. v2 models them
explicitly:

  PERSISTENT (resident the whole run):
    • weights        = params × bytes/param            (bfp8 ≈ 1 B, bf16 = 2 B)
    • kv_pool        = concurrency × seq × per_token_KV (KV dtype configurable;
                       bfp8 KV = 64 KB/tok for Llama-8B, bf16 = 128 KB/tok)
    • trace_residency= decode-graph buffers held for trace replay when
                       enable_trace=True (~3.3 GiB for Llama-8B b32; batch-scaled)

  TRANSIENT (peak during the prefill warmup — this is what actually OOMs):
    • buffer_embed   = max_num_batched_tokens × hidden × 1 B   ("Buffer A")
                       mnbt defaults to batch×max_model_len (= 2.1M @ b32×64K →
                       ~8 GiB!). Chunked prefill lets you cap mnbt and shrink this.
    • buffer_ffn     = prefill_width × ffn_intermediate × 2 B  ("Buffer B")
                       the MLP SiLU activation (ttnn.silu). prefill_width =
                       max_model_len (the FFN is NOT chunked even with chunked
                       prefill), so this is mnbt-INDEPENDENT and ~max_model_len-bound.
                       Llama-8B 64K: 65536 × 14336 × 2 = 1.75 GiB.
    • other_act      = prefill_width × hidden × 2 B × OTHER_MULT
                       calibrated catch-all for the remaining unfused prefill
                       activations (qkv/o_proj/gate/up/down/residual live buffers).
    These buffers peak in DIFFERENT phases, so the transient peak is
    max(embed_phase=buffer_embed, mlp_phase=buffer_ffn+other_act) — NOT their sum.
    (buffer_embed is intra-op: it only ever showed in the OOM, never in the
    op-boundary peak, confirming it's not co-resident with the MLP activations.)

KEY FINDINGS baked in:
  • The gmu ceiling is set by the prefill TRANSIENT competing with the KV pool
    for *contiguous* DRAM — not by total bytes. We derate usable DRAM by
    --usable-frac (default 0.95) to approximate the contiguity/banking limit.
  • `max_num_batched_tokens`, NOT `prefill_chunk_size`, sizes buffer_embed.
    mnbt defaults to batch×max_model_len; cap it (recommend batch×chunk) to
    decouple prefill DRAM from max_model_len.
  • Anchor (validated): Llama-3.1-8B b32×64K, bfp8 wts+KV, mnbt=65536, trace=on
    → peak ≈ 29.6 GiB, FITS at gmu ≤ 0.35, OOMs at 0.40 (Buffer B can't find a
    contiguous slab). v2 reproduces this.

============================================================================
PUSH-BUTTON USAGE
============================================================================
    python scripts/kv_cache_estimator.py --model Llama-3.1-8B-Instruct
        → fit matrix (batch × seq) with max gmu per cell, good defaults
          (bfp8, chunked prefill on, mnbt=batch×chunk, trace on).

    python scripts/kv_cache_estimator.py --model Llama-3.1-8B-Instruct \
        --batch-size 32 --seq-len 65536            # single-config detail

    # model the OLD non-chunked behavior (mnbt forced to batch×seq):
    python scripts/kv_cache_estimator.py --model Llama-3.1-8B-Instruct \
        --batch-size 32 --seq-len 65536 --no-chunked-prefill

    # try a specific mnbt:
    python ... --batch-size 32 --seq-len 65536 --mnbt 2048

NOTE: feasibility is judged against the prefill WARMUP peak (prefill_width =
max_model_len = the seq you pass), which is the real OOM point. Actual serving
with short prompts uses less. Calibration anchors on b32×64K Llama-8B; re-tune
--other-mult / --trace-gb / --usable-frac if you gather more data points.
"""

import argparse
import math

# ---------------------------------------------------------------------------
# Model architecture registry. Cross-check with HF config.json when adding new
# entries. ffn_intermediate is intermediate_size (drives Buffer B / the MLP
# SiLU activation) — VERIFY these against config.json; only Llama-3.1-8B
# (14336) is confirmed from IR.
# ---------------------------------------------------------------------------
MODELS = {
    "Falcon3-7B-Instruct": dict(
        num_layers=28, num_kv_heads=8, head_dim=256, hidden_size=3072,
        ffn_intermediate=23040, param_count_b=7.46,
    ),
    "Llama-3.1-8B-Instruct": dict(
        num_layers=32, num_kv_heads=8, head_dim=128, hidden_size=4096,
        ffn_intermediate=14336, param_count_b=8.03,   # ffn confirmed from IR
    ),
    "Llama-3.2-3B-Instruct": dict(
        num_layers=28, num_kv_heads=8, head_dim=128, hidden_size=3072,
        ffn_intermediate=8192, param_count_b=3.21,
    ),
    "Qwen3-4B": dict(
        num_layers=36, num_kv_heads=8, head_dim=128, hidden_size=2560,
        ffn_intermediate=9728, param_count_b=4.02,
    ),
    "Qwen3-8B": dict(
        num_layers=36, num_kv_heads=8, head_dim=128, hidden_size=4096,
        ffn_intermediate=12288, param_count_b=8.19,
    ),
}

# P150: 8 DRAM banks × 4272341376 B ≈ 31.88 GiB usable (matches plugin log).
P150_DRAM_GIB = 31.88
GIB = 1024**3

# Calibration constants (anchored on Llama-3.1-8B b32×64K, bfp8, mnbt=65536,
# trace on → measured peak ~29.6-30.0 GiB; OOM at gmu 0.40).
DEFAULT_OTHER_MULT = 11.0   # other prefill activations as multiple of (prefill_width×hidden×2B)
DEFAULT_TRACE_GIB = 3.3     # trace residency for the b32 Llama-8B anchor (batch/arch-scaled below)
DEFAULT_USABLE_FRAC = 0.95  # contiguity/banking derate: peak must be <= frac×DRAM
TRACE_REF = dict(batch=32, hidden=4096, layers=32)  # anchor for trace scaling

WBYTES = {"bf16": 2.0, "bfp8": 1.0}
KVBYTES = {"bf16": 2.0, "bfp8": 1.0}
ACT_EMBED_BYTES = 1.0  # embedding/hidden buffer (Buffer A) observed as 1 B (bfp8) in IR


def trace_residency_gib(cfg, batch, enable_trace, trace_gib):
    """Trace-replay resident buffers, scaled from the b32 Llama-8B anchor by
    batch × hidden × layers (the decode-graph activation footprint)."""
    if not enable_trace:
        return 0.0
    ref = TRACE_REF["batch"] * TRACE_REF["hidden"] * TRACE_REF["layers"]
    cur = batch * cfg["hidden_size"] * cfg["num_layers"]
    return trace_gib * (cur / ref)


def estimate(cfg, batch, seq_len, *, max_concurrency, weight_dtype, kv_dtype,
             chunked_prefill, mnbt, prefill_chunk_size, chunked_ffn,
             enable_trace, other_mult, trace_gib, usable_frac, dram_gib):
    """Return component GiB + peak + fit + max gmu/concurrency for (batch, seq)."""
    H = cfg["hidden_size"]
    FFN = cfg["ffn_intermediate"]

    # --- PERSISTENT ---
    weights = cfg["param_count_b"] * 1e9 * WBYTES[weight_dtype] / GIB

    per_token_kv = cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"] * 2 * KVBYTES[kv_dtype]
    eff_conc = min(max_concurrency, batch)
    # KV DEMAND if every concurrent seq were simultaneously at full length. This is
    # NOT what gets reserved — vLLM reserves gmu×DRAM and preempts when demand
    # exceeds the pool. Informational only; b×seq full demand routinely exceeds DRAM.
    kv_demand_full = eff_conc * seq_len * per_token_kv / GIB

    trace = trace_residency_gib(cfg, batch, enable_trace, trace_gib)

    # --- TRANSIENT (prefill warmup peak) ---
    # mnbt: with chunked prefill you may cap it; without, vLLM uses batch×max_model_len.
    if not chunked_prefill:
        eff_mnbt = batch * seq_len
    else:
        eff_mnbt = mnbt if mnbt is not None else batch * prefill_chunk_size
    buffer_embed = eff_mnbt * H * ACT_EMBED_BYTES / GIB           # Buffer A

    # FFN (Buffer B) + other activations scale with the prefill bucket width.
    # The FFN is NOT chunked today -> width = max_model_len (= seq_len here).
    # --chunked-ffn models the hypothetical where it would be chunk-bounded.
    ffn_width = prefill_chunk_size if (chunked_ffn and chunked_prefill) else seq_len
    buffer_ffn = ffn_width * FFN * 2.0 / GIB                      # Buffer B
    other_act = ffn_width * H * 2.0 * other_mult / GIB

    # The transient buffers peak in DIFFERENT prefill phases, not simultaneously:
    #   embed phase: the mnbt-bound buffer_embed (Buffer A) — an intra-op buffer
    #     (that's why it only ever showed in the OOM, never in the op-boundary peak).
    #   mlp/attn phase: buffer_ffn (Buffer B) + the other unfused activations.
    # The warmup peak is the LARGER phase, not the sum (summing wrongly penalizes
    # the non-chunked 8 GiB-embed case, which empirically still ran at gmu ~0.30).
    embed_phase = buffer_embed
    mlp_phase = buffer_ffn + other_act
    transient = max(embed_phase, mlp_phase)
    # Non-KV footprint must fit alongside SOME KV pool. This is the real warmup
    # gate: if weights+trace+transient already exceeds usable DRAM, no gmu helps.
    non_kv = weights + trace + transient

    usable = usable_frac * dram_gib
    fits = non_kv <= usable

    # KV pool you can still afford after the non-KV footprint -> max gmu and the
    # max #full-length seqs that pool holds (vLLM's "maximum concurrency").
    kv_headroom = max(0.0, usable - non_kv)
    max_gmu = math.floor((kv_headroom / dram_gib) * 100) / 100  # round DOWN (conservative)
    max_concurrency_fit = kv_headroom * GIB / (seq_len * per_token_kv)
    # Peak DRAM when the KV pool is sized to the max that fits.
    peak = non_kv + kv_headroom

    return dict(
        weights=weights, kv_demand_full=kv_demand_full, trace=trace,
        buffer_embed=buffer_embed, buffer_ffn=buffer_ffn, other_act=other_act,
        embed_phase=embed_phase, mlp_phase=mlp_phase,
        transient=transient, non_kv=non_kv, peak=peak, fits=fits,
        per_token_kv=per_token_kv, eff_conc=eff_conc, eff_mnbt=eff_mnbt,
        ffn_width=ffn_width, usable=usable, kv_headroom=kv_headroom,
        max_gmu=max_gmu, max_concurrency_fit=max_concurrency_fit,
    )


def print_detail(model, batch, seq_len, args):
    cfg = MODELS[model]
    e = estimate(cfg, batch, seq_len, **_kw(args))
    cp = "on" if args.chunked_prefill else "OFF"
    print(f"\n=== {model}  b={batch}  seq={seq_len}  (chunked_prefill={cp}, "
          f"mnbt={e['eff_mnbt']:,}, wts={args.weight_dtype}, kv={args.kv_dtype}, "
          f"trace={'on' if args.enable_trace else 'off'}) ===")
    print(f"  per-token KV: {e['per_token_kv']/1024:.1f} KB"
          f"  ({cfg['num_layers']}L × {cfg['num_kv_heads']}kv × {cfg['head_dim']}hd × 2 × {KVBYTES[args.kv_dtype]:.0f}B)")
    print()
    print(f"  NON-KV FOOTPRINT (must fit, leaving room for the KV pool)")
    print(f"    weights            {e['weights']:7.2f} GiB")
    print(f"    trace_residency    {e['trace']:7.2f} GiB   ({'enabled' if args.enable_trace else 'disabled'})")
    dom = "embed" if e['embed_phase'] >= e['mlp_phase'] else "mlp/attn"
    print(f"    -- prefill transient = max(embed_phase, mlp_phase); width={e['ffn_width']:,} --")
    print(f"    embed_phase        {e['embed_phase']:7.2f} GiB   buffer_embed (A) = mnbt {e['eff_mnbt']:,} × {cfg['hidden_size']} × 1B")
    print(f"    mlp_phase          {e['mlp_phase']:7.2f} GiB   buffer_ffn (B) {e['buffer_ffn']:.2f} ({e['ffn_width']:,}×{cfg['ffn_intermediate']}×2B, SiLU) + other_act {e['other_act']:.2f}")
    print(f"    transient (max)    {e['transient']:7.2f} GiB   [{dom} phase dominates]")
    print(f"  ---------------------------------")
    print(f"    non-KV total       {e['non_kv']:7.2f} GiB")
    print(f"    usable ({args.usable_frac:g}×{args.dram_gib:.2f}) {e['usable']:7.2f} GiB")
    head = e['usable'] - e['non_kv']
    print(f"    KV headroom        {head:+7.2f} GiB   {'✅ FITS' if e['fits'] else '❌ model+transient too big — no gmu helps'}")
    print()
    if e['fits']:
        print(f"  → max gmu that fits:        {e['max_gmu']:.2f}      (KV pool up to {e['kv_headroom']:.2f} GiB)")
        print(f"  → max full-length seqs held: {e['max_concurrency_fit']:.1f}x at seq={seq_len}  "
              f"(vLLM 'maximum concurrency')")
        if e['max_concurrency_fit'] < batch:
            print(f"     ⚠ < batch={batch}: can't hold all {batch} seqs at full {seq_len} at once → "
                  f"vLLM preempts (fine if prompts are short; only the KV CEILING is {seq_len}).")
        print(f"  KV demand if all {e['eff_conc']} seqs were at full {seq_len}: {e['kv_demand_full']:.1f} GiB "
              f"(informational — not reserved)")
    else:
        over = e['non_kv'] - e['usable']
        print(f"  ❌ non-KV footprint over by {over:.2f} GiB. Levers: cap mnbt (now {e['eff_mnbt']:,}; "
              f"embed_phase={e['embed_phase']:.2f} GiB), lower max_model_len (mlp_phase={e['mlp_phase']:.2f} "
              f"scales with it), drop trace (−{e['trace']:.2f}), or smaller batch.")


def print_matrix(model, batch_sizes, seq_lens, args):
    cfg = MODELS[model]
    cp = "on" if args.chunked_prefill else "OFF"
    print(f"\n=== {model}  (chunked_prefill={cp}, mnbt={'batch×chunk' if args.mnbt is None and args.chunked_prefill else (args.mnbt or 'batch×seq')}, "
          f"wts={args.weight_dtype}, kv={args.kv_dtype}, trace={'on' if args.enable_trace else 'off'}, "
          f"usable={args.usable_frac:g}×{args.dram_gib:.2f}={args.usable_frac*args.dram_gib:.1f} GiB) ===\n")
    wt = cfg['param_count_b']*1e9*WBYTES[args.weight_dtype]/GIB
    print(f"  weights {wt:.2f} GiB | per-token KV {cfg['num_layers']*cfg['num_kv_heads']*cfg['head_dim']*2*KVBYTES[args.kv_dtype]/1024:.0f} KB | ffn_intermediate {cfg['ffn_intermediate']}\n")

    header = f"  {'batch':>5}  " + "  ".join(f"{s:>9}" for s in seq_lens)

    print("  NON-KV FOOTPRINT (GiB) = weights+trace+prefill transient — ✅ fits (room for KV), ❌ too big")
    print(header); print("  " + "-" * (len(header) - 2))
    for b in batch_sizes:
        cells = []
        for s in seq_lens:
            e = estimate(cfg, b, s, **_kw(args))
            cells.append(f"{e['non_kv']:6.1f}{'✅' if e['fits'] else '❌'}")
        print(f"  b={b:>3}  " + "  ".join(f"{c:>9}" for c in cells))

    print()
    print("  MAX gmu that fits (— = model+transient already too big)")
    print(header); print("  " + "-" * (len(header) - 2))
    for b in batch_sizes:
        cells = []
        for s in seq_lens:
            e = estimate(cfg, b, s, **_kw(args))
            cells.append(f"{e['max_gmu']:.2f}" if e['fits'] else "—")
        print(f"  b={b:>3}  " + "  ".join(f"{c:>9}" for c in cells))

    print()
    print("  MAX full-length concurrency at that gmu")
    print(header); print("  " + "-" * (len(header) - 2))
    for b in batch_sizes:
        cells = []
        for s in seq_lens:
            e = estimate(cfg, b, s, **_kw(args))
            cells.append(f"{e['max_concurrency_fit']:.1f}x" if e['fits'] else "—")
        print(f"  b={b:>3}  " + "  ".join(f"{c:>9}" for c in cells))

    print()
    print("  Notes: non-KV = weights + trace + prefill transient, where transient =")
    print("  max(embed_phase, mlp_phase) (the buffers peak in different phases, not summed).")
    print("  Feasibility judged at the prefill WARMUP peak (prefill width = seq = max_model_len).")
    print("  'max gmu' = largest KV pool that fits alongside the transient, contiguity-derated")
    print("  by --usable-frac. Concurrency < batch just means vLLM preempts — not an OOM.")
    if not args.chunked_prefill:
        print("  chunked_prefill OFF → mnbt forced to batch×seq → buffer_embed is huge (this")
        print("  is the pre-chunked-prefill world; explains the b32×64K OOMs at any gmu).")


def _kw(args):
    return dict(
        max_concurrency=args.max_concurrency, weight_dtype=args.weight_dtype,
        kv_dtype=args.kv_dtype, chunked_prefill=args.chunked_prefill,
        mnbt=args.mnbt, prefill_chunk_size=args.prefill_chunk_size,
        chunked_ffn=args.chunked_ffn, enable_trace=args.enable_trace,
        other_mult=args.other_mult, trace_gib=args.trace_gib,
        usable_frac=args.usable_frac, dram_gib=args.dram_gib,
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    p.add_argument("--dram-gib", type=float, default=P150_DRAM_GIB,
                   help=f"Usable device DRAM in GiB (default {P150_DRAM_GIB} for P150)")

    # dtypes
    p.add_argument("--weight-dtype", choices=["bf16", "bfp8"], default="bfp8")
    p.add_argument("--kv-dtype", choices=["bf16", "bfp8"], default="bfp8",
                   help="KV cache dtype (default bfp8 = half the bf16 footprint)")

    # chunked prefill + mnbt (the key new knobs)
    cp = p.add_mutually_exclusive_group()
    cp.add_argument("--chunked-prefill", dest="chunked_prefill", action="store_true", default=True)
    cp.add_argument("--no-chunked-prefill", dest="chunked_prefill", action="store_false")
    p.add_argument("--mnbt", "--max-num-batched-tokens", type=int, default=None,
                   help="max_num_batched_tokens. Default with chunked prefill = batch×chunk; "
                        "without = batch×seq. Sizes buffer_embed (Buffer A).")
    p.add_argument("--prefill-chunk-size", type=int, default=2048,
                   help="Chunk size for chunked prefill (default 2048; also the default-mnbt factor).")
    p.add_argument("--chunked-ffn", action="store_true",
                   help="HYPOTHETICAL: model the FFN/activations as chunk-bounded too "
                        "(not implemented in the stack yet; off by default).")

    # trace
    tr = p.add_mutually_exclusive_group()
    tr.add_argument("--enable-trace", dest="enable_trace", action="store_true", default=True)
    tr.add_argument("--no-trace", dest="enable_trace", action="store_false")

    p.add_argument("--max-concurrency", type=int, default=None,
                   help="Cap on simultaneous full-length seqs in the KV pool "
                        "(default = batch_size). Effective = min(this, batch).")

    # calibration knobs
    p.add_argument("--other-mult", type=float, default=DEFAULT_OTHER_MULT,
                   help=f"Other-activation working-set multiple (default {DEFAULT_OTHER_MULT}, calibrated).")
    p.add_argument("--trace-gib", type=float, default=DEFAULT_TRACE_GIB,
                   help=f"Trace residency for b32 Llama-8B anchor (default {DEFAULT_TRACE_GIB}, batch-scaled).")
    p.add_argument("--usable-frac", type=float, default=DEFAULT_USABLE_FRAC,
                   help=f"Contiguity/banking derate; peak must be <= frac×DRAM (default {DEFAULT_USABLE_FRAC}).")

    # single vs sweep
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p.add_argument("--seq-lens", default="2048,4096,8192,16384,32768,65536")

    args = p.parse_args()
    if args.max_concurrency is None:
        # default concurrency = batch (single-config) or large (sweep, capped per-row)
        args.max_concurrency = args.batch_size if args.batch_size else 10**9

    if args.batch_size is not None and args.seq_len is not None:
        print_detail(args.model, args.batch_size, args.seq_len, args)
    else:
        bs = [int(x) for x in args.batch_sizes.split(",")]
        sl = [int(x) for x in args.seq_lens.split(",")]
        print_matrix(args.model, bs, sl, args)


if __name__ == "__main__":
    main()
