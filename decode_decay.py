#!/usr/bin/env python3
"""Measure INSTANTANEOUS decode rate vs generated-token index for one stream.

Unlike test_all_llm_servers.sh's per-stream tok/s (which is the *average* over
the whole generation), this records the arrival time of every streamed token
and reports decode tok/s in buckets along the generation — so you see the
*shape* of any slowdown, not just the length-averaged number.

It exists because of a surprising finding on the TT vLLM stack: decode slows as
tokens are GENERATED, but NOT as context is PREFILLED. i.e. generating to depth
8k runs ~2x slower than prefilling to depth 8k and decoding there. Context size
alone doesn't explain it, which points at per-decode-step host overhead growing
with output length. This harness makes that curve explicit for a bug report.

Usage:
  ./decode_decay.py                              # ISL 128, OSL 4096, port 8004
  ./decode_decay.py --isl 128 --osl 8192 --bucket 512
  ./decode_decay.py --port 8101 --model Qwen/Qwen3-8B
  ./decode_decay.py --baseline                   # also probe decode at a PREFILLED
                                                 # depth (ISL=OSL) for comparison

Notes:
- Assumes one token per streamed chunk (true for the TT vLLM /v1/completions
  build here); the per-token timestamps are the ground truth either way.
- Uses a cold, non-cacheable token-ID prompt (varied + per-run salt) so prefill
  isn't served from the server's prefix cache.
"""
import argparse, json, os, random, sys, time
import requests

TOK_LO, TOK_BAND = 1000, 20000          # ids valid/non-special on every vocab (>=32k)
RUN_SALT = random.Random().randrange(1 << 30)


def token_prompt(n, tag=0):
    """n distinct, non-cacheable input token IDs (cold prefill each run)."""
    rng = random.Random(RUN_SALT * 1_000_003 + n * 31 + tag)
    return [TOK_LO + rng.randrange(TOK_BAND) for _ in range(n)]


def discover_model(host, port, key):
    r = requests.get(f"http://{host}:{port}/v1/models",
                     headers={"Authorization": f"Bearer {key}"}, timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def stream_times(host, port, key, model, prompt, max_tokens):
    """Return (ttft_s, [perf_counter timestamp of each generated token])."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "ignore_eos": True, "stream": True}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    times = []
    r = requests.post(f"http://{host}:{port}/v1/completions",
                      headers=headers, json=body, stream=True, timeout=1800)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        ch = (json.loads(data).get("choices") or [{}])[0]
        if ch.get("text"):
            times.append(time.perf_counter())
    if not times:
        raise RuntimeError("no tokens generated")
    return times[0] - t0, times


def bar(frac, width=30):
    return "#" * int(round(frac * width))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("HOST", "localhost"))
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "your-secret-key"))
    ap.add_argument("--model", default=None, help="default: auto-discover via /v1/models")
    ap.add_argument("--isl", type=int, default=128, help="input tokens (prefill depth)")
    ap.add_argument("--osl", type=int, default=4096, help="output tokens to generate")
    ap.add_argument("--bucket", type=int, default=256, help="tokens per measurement bucket")
    ap.add_argument("--baseline", action="store_true",
                    help="also measure decode at a PREFILLED depth (ISL=OSL, short decode) "
                         "to contrast prefilled-depth vs generated-depth")
    args = ap.parse_args()

    model = args.model or discover_model(args.host, args.port, args.api_key)
    print(f"model={model}  host={args.host}:{args.port}  ISL={args.isl}  OSL={args.osl}  "
          f"bucket={args.bucket}")

    ttft, times = stream_times(args.host, args.port, args.api_key, model,
                               token_prompt(args.isl), args.osl)
    n = len(times)
    overall = (n - 1) / (times[-1] - times[0]) if n > 1 else float("nan")
    print(f"TTFT={ttft*1000:.0f}ms   generated={n} tokens   "
          f"avg decode={overall:.1f} tok/s\n")

    # Bucket tokens 1..n-1 (token 0's arrival is TTFT, not a decode step).
    rows = []
    a = 1
    while a < n:
        b = min(a + args.bucket - 1, n - 1)
        dt = times[b] - times[a - 1]
        rate = (b - a + 1) / dt if dt > 0 else float("nan")
        depth = args.isl + (a + b) // 2          # absolute context at bucket midpoint
        rows.append((a, b, depth, rate))
        a = b + 1

    peak = max(r[3] for r in rows)
    print(f"{'gen tok':>13}   {'ctx depth':>9}   {'tok/s':>7}   {'ms/tok':>7}   rate (vs peak)")
    print(f"{'-'*13}   {'-'*9}   {'-'*7}   {'-'*7}   {'-'*30}")
    for a, b, depth, rate in rows:
        print(f"{a:>5}-{b:<7}   {depth:>9}   {rate:>7.1f}   {1000/rate:>7.1f}   "
              f"{bar(rate/peak)}")

    if n > args.bucket:
        first = rows[0][3]
        last = rows[-1][3]
        print(f"\ndecode rate: {first:.1f} tok/s (start) -> {last:.1f} tok/s (end)  "
              f"= {first/last:.2f}x slowdown over {n} generated tokens")

    if args.baseline:
        # Prefill to ~the same final depth, then decode a short burst: this is
        # decode at that depth WITHOUT having generated its way there.
        depth = args.isl + args.osl
        burst = min(64, args.bucket)
        bt, btimes = stream_times(args.host, args.port, args.api_key, model,
                                  token_prompt(depth, tag=1), burst)
        if len(btimes) > 1:
            brate = (len(btimes) - 1) / (btimes[-1] - btimes[0])
            gen_end_rate = rows[-1][3]
            print(f"\nBASELINE (decode at depth {depth} reached two ways):")
            print(f"  prefilled  -> {brate:.1f} tok/s  (TTFT={bt*1000:.0f}ms, decode {len(btimes)} tok)")
            print(f"  generated  -> {gen_end_rate:.1f} tok/s  (end-of-generation bucket above)")
            if gen_end_rate > 0:
                print(f"  => generating to this depth is {brate/gen_end_rate:.2f}x slower "
                      f"than prefilling to it")


if __name__ == "__main__":
    main()
