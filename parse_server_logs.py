#!/usr/bin/env python3
"""Parse tt-media-server journald logs and report usage statistics."""

import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta

YEAR = 2026  # journald logs don't include year


def fetch_logs(since="2026-03-22"):
    result = subprocess.run(
        ["sudo", "journalctl", "-u", "tt-media-server", "--since", since, "--no-pager"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error fetching logs: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.splitlines()


def parse_timestamp(line):
    """Extract datetime from journald log line like 'Mar 24 06:19:43 ...'"""
    m = re.match(r"^([A-Z][a-z]+ \d+ \d+:\d+:\d+)", line)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%b %d %H:%M:%S")
        return dt.replace(year=YEAR)
    except ValueError:
        return None


def parse_date(line):
    dt = parse_timestamp(line)
    return dt.date() if dt else None


def main():
    print("Fetching logs from journald...")
    lines = fetch_logs()
    print(f"Fetched {len(lines)} log lines.\n")

    # -- Accumulators --
    requests_per_day = defaultdict(int)           # day -> count
    streaming_times = defaultdict(list)           # day -> [seconds]
    streaming_tokens = defaultdict(list)          # day -> [item_count]
    worker_requests = defaultdict(int)            # worker_id -> count
    task_ids = defaultdict(set)                   # day -> {task_uuid}
    errors_per_day = defaultdict(list)            # day -> [error_msg]
    service_starts = []                           # [(datetime, success_bool)]
    service_crashes = defaultdict(int)            # day -> count
    weight_load_times = []                        # [seconds]
    worker_stop_times = []                        # [seconds]
    hourly_requests = defaultdict(int)            # hour (0-23) -> count
    tokens_per_request = []                       # all (seconds, tokens) pairs
    device_generation_complete = defaultdict(int) # device_id -> count

    # Regex patterns
    re_streaming_complete = re.compile(
        r"\[process_streaming_request\] async generator completed in ([\d.]+) seconds\. Yielded (\d+) items"
    )
    re_worker_request = re.compile(
        r"Worker (\d+) processing streaming request for task ([0-9a-f-]{36})"
    )
    re_device_gen_complete = re.compile(
        r"Device (\d+): Streaming generation completed"
    )
    re_service_start = re.compile(r"Started tt-media-server\.service")
    re_service_fail = re.compile(r"tt-media-server\.service: Failed with result '([^']+)'")
    re_restart_counter = re.compile(r"restart counter is at (\d+)")
    re_weight_load = re.compile(r"Loading weights took ([\d.]+) seconds")
    re_stop_workers = re.compile(r"\[stop_workers\] executed in ([\d.]+) seconds\. Stopping workers")
    re_error = re.compile(r"(ERROR|TT_FATAL|Exception|Traceback|CRITICAL)", re.IGNORECASE)
    re_port_in_use = re.compile(r"error while attempting to bind on address.*address already in use")

    for line in lines:
        day = parse_date(line)
        ts = parse_timestamp(line)

        # Streaming request completion (timing + token count)
        m = re_streaming_complete.search(line)
        if m and day:
            secs = float(m.group(1))
            tokens = int(m.group(2))
            streaming_times[day].append(secs)
            streaming_tokens[day].append(tokens)
            tokens_per_request.append((secs, tokens))
            requests_per_day[day] += 1
            if ts:
                hourly_requests[ts.hour] += 1
            continue

        # Worker dispatching a request
        m = re_worker_request.search(line)
        if m and day:
            worker_id = int(m.group(1))
            task_uuid = m.group(2)
            worker_requests[worker_id] += 1
            task_ids[day].add(task_uuid)
            continue

        # Device generation completed
        m = re_device_gen_complete.search(line)
        if m:
            device_generation_complete[int(m.group(1))] += 1
            continue

        # Service started
        if re_service_start.search(line) and ts:
            service_starts.append(ts)
            continue

        # Service failed
        m = re_service_fail.search(line)
        if m and day:
            service_crashes[day] += 1
            continue

        # Weight loading time
        m = re_weight_load.search(line)
        if m:
            weight_load_times.append(float(m.group(1)))
            continue

        # Worker stop time
        m = re_stop_workers.search(line)
        if m:
            worker_stop_times.append(float(m.group(1)))
            continue

        # Errors
        if re_error.search(line) and day:
            # Extract a short description
            short = line.split(": ", 2)[-1][:120] if ": " in line else line[:120]
            errors_per_day[day].append(short)
            continue

        if re_port_in_use.search(line) and day:
            errors_per_day[day].append("Port 8000 already in use")

    # ===== REPORT =====
    all_days = sorted(set(
        list(requests_per_day.keys()) +
        list(errors_per_day.keys()) +
        list(service_crashes.keys()) +
        list(streaming_times.keys())
    ))

    print("=" * 72)
    print("  TT-MEDIA-SERVER  USAGE REPORT")
    print(f"  Log range: {all_days[0]} to {all_days[-1]}" if all_days else "  No data")
    print(f"  Model: Llama-3.1-8B-Instruct on p300x2 (4 devices)")
    print("=" * 72)

    # -- Overall summary --
    total_requests = sum(requests_per_day.values())
    total_tokens = sum(t for day_tokens in streaming_tokens.values() for t in day_tokens)
    all_times = [t for day_times in streaming_times.values() for t in day_times]
    all_tok = [t for day_tokens in streaming_tokens.values() for t in day_tokens]
    total_errors = sum(len(v) for v in errors_per_day.values())
    total_crashes = sum(service_crashes.values())

    print("\n--- OVERALL SUMMARY ---")
    print(f"  Total requests completed:   {total_requests}")
    print(f"  Total tokens generated:     {total_tokens:,}")
    print(f"  Total service failures:     {total_crashes}")
    print(f"  Total error log lines:      {total_errors}")
    print(f"  Service start count:        {len(service_starts)}")
    if all_times:
        print(f"  Avg response time:          {sum(all_times)/len(all_times):.2f}s")
        print(f"  Median response time:       {sorted(all_times)[len(all_times)//2]:.2f}s")
        print(f"  Max response time:          {max(all_times):.2f}s")
    if all_tok:
        print(f"  Avg tokens per request:     {sum(all_tok)/len(all_tok):.1f}")
        print(f"  Median tokens per request:  {sorted(all_tok)[len(all_tok)//2]}")
        print(f"  Max tokens per request:     {max(all_tok)}")
    if all_times and all_tok:
        avg_tps = sum(t / s for s, t in tokens_per_request if s > 0) / len(tokens_per_request)
        print(f"  Avg throughput:             {avg_tps:.1f} tokens/sec")

    # -- Per-day breakdown --
    print("\n--- DAILY BREAKDOWN ---")
    print(f"  {'Date':<12} {'Requests':>9} {'Tokens':>9} {'Avg Tok':>8} {'Avg Time':>9} {'Med Time':>9} {'Errors':>7} {'Crashes':>8}")
    print(f"  {'-'*10:<12} {'-'*9:>9} {'-'*9:>9} {'-'*8:>8} {'-'*9:>9} {'-'*9:>9} {'-'*7:>7} {'-'*8:>8}")
    for day in all_days:
        reqs = requests_per_day.get(day, 0)
        toks = streaming_tokens.get(day, [])
        times = streaming_times.get(day, [])
        errs = len(errors_per_day.get(day, []))
        crashes = service_crashes.get(day, 0)
        avg_tok = f"{sum(toks)/len(toks):.0f}" if toks else "-"
        avg_time = f"{sum(times)/len(times):.2f}s" if times else "-"
        med_time = f"{sorted(times)[len(times)//2]:.2f}s" if times else "-"
        total_day_tok = sum(toks)
        print(f"  {str(day):<12} {reqs:>9} {total_day_tok:>9,} {avg_tok:>8} {avg_time:>9} {med_time:>9} {errs:>7} {crashes:>8}")

    # -- Hourly distribution --
    if hourly_requests:
        print("\n--- HOURLY REQUEST DISTRIBUTION ---")
        peak_hour = max(hourly_requests, key=hourly_requests.get)
        max_count = max(hourly_requests.values())
        for h in range(24):
            count = hourly_requests.get(h, 0)
            bar = "█" * int(40 * count / max_count) if max_count > 0 and count > 0 else ""
            marker = " ◄ peak" if h == peak_hour and count > 0 else ""
            print(f"  {h:02d}:00  {count:>5}  {bar}{marker}")

    # -- Worker load distribution --
    if worker_requests:
        print("\n--- WORKER LOAD DISTRIBUTION ---")
        total_worker = sum(worker_requests.values())
        for wid in sorted(worker_requests):
            count = worker_requests[wid]
            pct = 100 * count / total_worker if total_worker else 0
            bar = "█" * int(40 * count / max(worker_requests.values()))
            print(f"  Worker {wid}:  {count:>5} requests ({pct:5.1f}%)  {bar}")

    # -- Device generation distribution --
    if device_generation_complete:
        print("\n--- DEVICE GENERATION COMPLETIONS ---")
        for did in sorted(device_generation_complete):
            print(f"  Device {did}:  {device_generation_complete[did]:>5} completions")

    # -- Response time distribution --
    if all_times:
        print("\n--- RESPONSE TIME DISTRIBUTION ---")
        buckets = [
            ("< 1s", 0, 1),
            ("1-5s", 1, 5),
            ("5-15s", 5, 15),
            ("15-30s", 15, 30),
            ("30-60s", 30, 60),
            ("60-120s", 60, 120),
            ("> 120s", 120, float("inf")),
        ]
        max_bucket = 0
        bucket_counts = []
        for label, lo, hi in buckets:
            c = sum(1 for t in all_times if lo <= t < hi)
            bucket_counts.append((label, c))
            max_bucket = max(max_bucket, c)
        for label, c in bucket_counts:
            bar = "█" * int(40 * c / max_bucket) if max_bucket > 0 and c > 0 else ""
            print(f"  {label:>8}  {c:>5}  {bar}")

    # -- Token count distribution --
    if all_tok:
        print("\n--- TOKENS PER REQUEST DISTRIBUTION ---")
        tok_buckets = [
            ("1-10", 1, 11),
            ("11-50", 11, 51),
            ("51-100", 51, 101),
            ("101-250", 101, 251),
            ("251-500", 251, 501),
            ("501-1000", 501, 1001),
            ("> 1000", 1001, float("inf")),
        ]
        max_tb = 0
        tb_counts = []
        for label, lo, hi in tok_buckets:
            c = sum(1 for t in all_tok if lo <= t < hi)
            tb_counts.append((label, c))
            max_tb = max(max_tb, c)
        for label, c in tb_counts:
            bar = "█" * int(40 * c / max_tb) if max_tb > 0 and c > 0 else ""
            print(f"  {label:>10}  {c:>5}  {bar}")

    # -- Operational stats --
    print("\n--- OPERATIONAL STATS ---")
    if weight_load_times:
        print(f"  Model weight load times:    avg {sum(weight_load_times)/len(weight_load_times):.2f}s "
              f"(min {min(weight_load_times):.2f}s, max {max(weight_load_times):.2f}s, n={len(weight_load_times)})")
    if worker_stop_times:
        print(f"  Worker shutdown times:      avg {sum(worker_stop_times)/len(worker_stop_times):.2f}s "
              f"(min {min(worker_stop_times):.2f}s, max {max(worker_stop_times):.2f}s, n={len(worker_stop_times)})")
    print(f"  Service start events:       {len(service_starts)}")
    if service_starts and len(service_starts) > 1:
        uptimes = []
        for i in range(len(service_starts) - 1):
            delta = service_starts[i + 1] - service_starts[i]
            uptimes.append(delta)
        longest = max(uptimes)
        shortest = min(uptimes)
        print(f"  Longest uptime window:      {longest}")
        print(f"  Shortest uptime window:     {shortest}")

    # -- Top errors --
    if errors_per_day:
        print("\n--- TOP ERRORS ---")
        error_counts = defaultdict(int)
        for day_errors in errors_per_day.values():
            for e in day_errors:
                # Normalize UUIDs, PIDs, timestamps for grouping
                normalized = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", e)
                normalized = re.sub(r"pid=\d+", "pid=<PID>", normalized)
                normalized = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+", "<TS>", normalized)
                error_counts[normalized] += 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  [{count:>3}x]  {err[:100]}")

    print("\n" + "=" * 72)
    print("  END OF REPORT")
    print("=" * 72)


if __name__ == "__main__":
    main()
