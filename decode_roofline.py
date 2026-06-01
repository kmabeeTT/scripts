#!/usr/bin/env python3
"""Estimate roofline decode throughput (tok/sec) for LLMs on single-chip
Wormhole (n150) and Blackhole (p150).

Decode (batch=1, autoregressive) is DRAM-bandwidth bound: every weight is
streamed from DRAM once per generated token. So:

    tok/sec = DRAM_bandwidth(GB/s) / model_size(GB) * efficiency

Weights default to bfp8 (block floating point, blocks of 32 sharing an 8-bit
exponent) = 8.25 bits/param = 1.03125 bytes/param.

    n150 (Wormhole)  : 288 GB/s   (Galaxy WH chip is ~336)
    p150 (Blackhole) : 512 GB/s
    efficiency       : 0.7  (achievable fraction of peak bandwidth)

Per Ognjen Djuricic's compiler-pass formula.

Usage:
    decode_roofline.py                 # table for 3B, 7B, 8B, 11B
    decode_roofline.py 8               # single 8B model
    decode_roofline.py 13.5            # fractional billions ok
    decode_roofline.py 70 --dtype bfp4 # large/MoE models in bfp4
    decode_roofline.py 8 --wh-bw 336   # Galaxy WH bandwidth
    decode_roofline.py 8 --efficiency 0.65
"""

import argparse

# bytes/param for the storage formats we care about. bfp* are block floating
# point with blocks of 32 sharing one 8-bit exponent:
#   bfp8: (32*8 + 8)/32 bits = 8.25  -> 1.03125 bytes
#   bfp4: (32*4 + 8)/32 bits = 4.25  -> 0.53125 bytes
BYTES_PER_PARAM = {
    "bfp4": 0.53125,
    "bfp8": 1.03125,
    "fp8": 1.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp32": 4.0,
}

DEFAULT_DTYPE = "bfp8"
DEFAULT_EFFICIENCY = 0.7
DEFAULT_WH_BW_GBPS = 288   # n150  Wormhole single chip
DEFAULT_BH_BW_GBPS = 512   # p150  Blackhole single chip

DEFAULT_SIZES_B = [3, 7, 8, 11]


def decode_toks(params_b, bw_gbps, bytes_per_param, efficiency):
    """tok/sec for `params_b` billion params at `bw_gbps` GB/s DRAM bandwidth."""
    model_size_gb = params_b * bytes_per_param  # billions of params * bytes/param = GB
    return bw_gbps / model_size_gb * efficiency


def print_table(sizes_b, wh_bw, bh_bw, bytes_per_param, efficiency):
    print(f"{'params':>8} {'wh tok/s':>10} {'bh tok/s':>10}")
    print(f"{'-'*8:>8} {'-'*10:>10} {'-'*10:>10}")
    for p in sizes_b:
        wh = decode_toks(p, wh_bw, bytes_per_param, efficiency)
        bh = decode_toks(p, bh_bw, bytes_per_param, efficiency)
        print(f"{f'{p:g}B':>8} {wh:>10.1f} {bh:>10.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Roofline decode tok/sec for single-chip Wormhole (n150) and Blackhole (p150).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "params_b", nargs="?", type=float, default=None,
        help="model size in billions of params (default: table for 3B,7B,8B,11B)",
    )
    parser.add_argument(
        "--dtype", choices=sorted(BYTES_PER_PARAM), default=DEFAULT_DTYPE,
        help="on-device weight storage format",
    )
    parser.add_argument(
        "--bytes-per-param", type=float, default=None,
        help="override bytes/param directly (takes precedence over --dtype)",
    )
    parser.add_argument(
        "--efficiency", type=float, default=DEFAULT_EFFICIENCY,
        help="achievable fraction of peak DRAM bandwidth",
    )
    parser.add_argument(
        "--wh-bw", type=float, default=DEFAULT_WH_BW_GBPS,
        help="Wormhole DRAM bandwidth GB/s (Galaxy WH chip ~336)",
    )
    parser.add_argument(
        "--bh-bw", type=float, default=DEFAULT_BH_BW_GBPS,
        help="Blackhole DRAM bandwidth GB/s",
    )
    args = parser.parse_args()

    bytes_per_param = (
        args.bytes_per_param if args.bytes_per_param is not None
        else BYTES_PER_PARAM[args.dtype]
    )

    sizes = DEFAULT_SIZES_B if args.params_b is None else [args.params_b]
    print_table(sizes, args.wh_bw, args.bh_bw, bytes_per_param, args.efficiency)


if __name__ == "__main__":
    main()
