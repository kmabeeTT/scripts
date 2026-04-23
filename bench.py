#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark all running inference servers on ports 8000-8003.
Measures tok/s and TTFT across prompt/output length configurations.

Usage:
    python3 bench.py                  # fast mode, localhost
    python3 bench.py --full           # exhaustive mode, localhost
    python3 bench.py --remote         # fast mode, 10.32.48.16
    python3 bench.py --full --remote  # exhaustive mode, 10.32.48.16
    python3 bench.py --host 1.2.3.4   # custom host
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

PORTS = [8000, 8001, 8002, 8003]
DEFAULT_HOST = "localhost"
REMOTE_HOST = "10.32.48.16"

FAST_CONFIGS = [
    ("short",  "Tell me a quick story",                            32),
    ("long",   "Tell me a quick story",                           128),
]
FAST_REPS = 2

FULL_CONFIGS = [
    ("32tok",  "Tell me a quick story",                            32),
    ("64tok",  "Tell me a quick story",                            64),
    ("128tok", "Explain the theory of relativity simply",         128),
    ("256tok", "Write a detailed essay on the history of AI",     256),
]
FULL_REPS = 5


def check_server(host, port, headers):
    try:
        r = requests.get(f"http://{host}:{port}/tt-liveness", headers=headers, timeout=2)
        if r.status_code == 200 and r.json().get("model_ready"):
            mr = requests.get(f"http://{host}:{port}/v1/models", headers=headers, timeout=2).json()
            model = mr["data"][0]["id"] if mr.get("data") else "unknown"
            return model
    except Exception:
        pass
    return None


def run_request(host, port, model, prompt, max_tokens, headers):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.6,
    }
    start = time.perf_counter()
    ttft = None
    token_count = 0
    try:
        r = requests.post(
            f"http://{host}:{port}/v1/chat/completions",
            headers=headers, json=payload, stream=True, timeout=120,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            text = chunk["choices"][0].get("delta", {}).get("content", "")
            if text and ttft is None:
                ttft = time.perf_counter() - start
            if text:
                token_count += 1
    except Exception:
        return None, None, None
    elapsed = time.perf_counter() - start
    tps = token_count / elapsed if elapsed > 0 and token_count > 0 else None
    return tps, ttft, token_count


def main():
    parser = argparse.ArgumentParser(description="Benchmark TT inference servers")
    parser.add_argument("--full", action="store_true", help="Exhaustive mode (more configs, more reps)")
    parser.add_argument("--remote", action="store_true", help=f"Use remote host {REMOTE_HOST}")
    parser.add_argument("--host", default=None, help="Custom host IP (overrides --remote)")
    args = parser.parse_args()

    host = args.host or (REMOTE_HOST if args.remote else DEFAULT_HOST)
    api_key = os.environ.get("API_KEY", "your-secret-key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    configs = FULL_CONFIGS if args.full else FAST_CONFIGS
    n_reps = FULL_REPS if args.full else FAST_REPS
    mode_str = f"FULL ({n_reps} reps)" if args.full else f"FAST ({n_reps} reps)"

    # Discover active servers
    print(f"Scanning {host} ports {PORTS[0]}-{PORTS[-1]}...")
    servers = {}
    for p in PORTS:
        m = check_server(host, p, headers)
        status = f"{m} ✓" if m else "not ready"
        print(f"  port {p}: {status}")
        if m:
            servers[p] = m

    if not servers:
        print("\nNo servers found. Exiting.")
        sys.exit(1)

    print(f"\nMode: {mode_str}")
    print(f"Configs: {[(c[0], c[2]) for c in configs]}")
    print()

    results = {}

    for port, model in servers.items():
        model_short = model.split("/")[-1]
        for label, prompt, max_tokens in configs:
            key = (port, model_short, label, max_tokens)
            tps_list, ttft_list = [], []
            print(f"  [{model_short} | port {port} | {label}] ", end="", flush=True)
            for _ in range(n_reps):
                tps, ttft, _ = run_request(host, port, model, prompt, max_tokens, headers)
                if tps is not None:
                    tps_list.append(tps)
                    ttft_list.append(ttft * 1000 if ttft else 0)
                    print(f"{tps:.1f}", end=" ", flush=True)
                else:
                    print("ERR", end=" ", flush=True)
            print()
            results[key] = (tps_list, ttft_list)

    # Summary table
    col = 105
    print()
    print("=" * col)
    print(f"{'Model':<42} {'Port':>4} {'Config':<8} {'MaxTok':>6}  {'Avg tok/s':>9} {'Min':>6} {'Max':>6}  {'Avg TTFT':>9}")
    print("=" * col)
    for (port, model_short, label, max_tokens), (tps_list, ttft_list) in sorted(results.items()):
        if tps_list:
            print(
                f"{model_short:<42} {port:>4} {label:<8} {max_tokens:>6}"
                f"  {statistics.mean(tps_list):>9.1f}"
                f" {min(tps_list):>6.1f} {max(tps_list):>6.1f}"
                f"  {statistics.mean(ttft_list):>8.0f}ms"
            )
        else:
            print(f"{model_short:<42} {port:>4} {label:<8} {max_tokens:>6}  {'ERROR':>9}")
    print("=" * col)


if __name__ == "__main__":
    main()
