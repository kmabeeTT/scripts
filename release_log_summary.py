#!/usr/bin/env python3
"""Summarize a tt-inference-server release/eval workflow log.

Extracts per-step / per-substep wall times, eval timeouts + scores, benchmark
sweep timing, and the acceptance result — so you can quickly compare runs.

Usage:
    release_log_summary.py LOG [LOG2 ...]
    release_log_summary.py local_release_qwen3_8b_p8009_*.log

Works on logs from run.py --workflow {evals,release,...} (run_local_release_*.sh
output or workflow_logs). Robust to the two timestamp formats in these logs:
  "2026-06-20 17:31:04,109"   (run.py / python logging)
  "2026-06-20:17:41:05"       (lm-eval)
"""
import re
import sys
from datetime import datetime

TS = re.compile(r"(\d{4}-\d{2}-\d{2})[ :](\d{2}:\d{2}:\d{2})")


def ts(line):
    m = TS.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def fmt_dur(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def first_ts_at_or_after(lines, idx):
    for j in range(idx, len(lines)):
        t = ts(lines[j])
        if t:
            return t
    return None


def summarize(path):
    with open(path, errors="replace") as f:
        lines = f.readlines()

    out = [f"\n{'='*78}", f"LOG: {path}", "=" * 78]

    # --- overall: RELEASE banners (own date format + elapsed + exit) ---
    start_banner = end_banner = None
    elapsed = exitcode = None
    for ln in lines:
        if "RELEASE RUN START" in ln:
            start_banner = ln.strip()
        if "RELEASE RUN END" in ln:
            end_banner = ln.strip()
            me = re.search(r"elapsed (\d+)s", ln)
            ex = re.search(r"exit=(\-?\d+)", ln)
            elapsed = int(me.group(1)) if me else None
            exitcode = ex.group(1) if ex else None

    # Fall back to first/last parseable timestamp for total wall.
    all_ts = [t for t in (ts(l) for l in lines) if t]
    if elapsed is None and len(all_ts) >= 2:
        elapsed = (all_ts[-1] - all_ts[0]).total_seconds()

    out.append(f"  total wall : {fmt_dur(elapsed)}"
               + (f"   exit={exitcode}" if exitcode is not None else ""))

    # --- workflow phases: 'Starting workflow: NAME' (+ task_name suffix) ---
    # Record top-level workflow starts (ignore the per-task 'evals task_name:').
    phase_marks = []  # (datetime, name, line_idx)
    for i, ln in enumerate(lines):
        m = re.search(r"Starting workflow:\s*([a-z_]+)\s*$", ln)
        if m:
            t = ts(ln)
            if t:
                phase_marks.append((t, m.group(1), i))
    # phase end = next phase start, or RELEASE end / last ts
    end_t = all_ts[-1] if all_ts else None
    if phase_marks:
        out.append("\n  PHASES")
        for k, (t, name, _) in enumerate(phase_marks):
            nxt = phase_marks[k + 1][0] if k + 1 < len(phase_marks) else end_t
            dur = (nxt - t).total_seconds() if nxt else None
            out.append(f"    {name:<12} {t.strftime('%H:%M:%S')}  {fmt_dur(dur)}")

    # --- trace-capture spans (warmup) ---
    cap = [ts(l) for i, l in enumerate(lines) if "trace capture: input_seq_len" in l and ts(l)]
    # group spans where consecutive capture lines are within a few min
    spans = []
    for t in cap:
        if spans and (t - spans[-1][1]).total_seconds() <= 600:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    if spans:
        out.append("\n  TRACE CAPTURE (warmup)")
        for a, b in spans:
            out.append(f"    {a.strftime('%H:%M:%S')} -> {b.strftime('%H:%M:%S')}  "
                       f"~{fmt_dur((b - a).total_seconds())}")

    # --- evals: per task ---
    # task start: 'Starting workflow: evals task_name: NAME'
    # decode start: next 'Running generate_until requests'
    # task end: next 'Saving per-task samples'
    task_starts = [(i, re.search(r"task_name:\s*(\S+)", l).group(1))
                   for i, l in enumerate(lines) if "evals task_name:" in l]
    saves = [i for i, l in enumerate(lines) if "Saving per-task samples" in l]
    gens = [i for i, l in enumerate(lines) if "Running generate_until requests" in l]
    # results table rows: |task| ver |filter| n |exact_match|↑| value |± stderr|
    scores = {}
    for l in lines:
        m = re.match(r"\|([a-z0-9_]+)\s*\|.*exact_match.*?\|\s*([0-9.]+)\s*\|", l)
        if m:
            scores[m.group(1)] = m.group(2)
    if task_starts:
        out.append("\n  EVALS")
        out.append(f"    {'task':<20}{'decode wall':<12}{'timeouts':<10}{'score'}")
        for k, (si, name) in enumerate(task_starts):
            nxt_start = task_starts[k + 1][0] if k + 1 < len(task_starts) else len(lines)
            gen = next((g for g in gens if g >= si), None)
            sav = next((s for s in saves if s >= (gen or si)), None)
            gt = first_ts_at_or_after(lines, gen) if gen else None
            st = first_ts_at_or_after(lines, sav) if sav else None
            dur = (st - gt).total_seconds() if (gt and st) else None
            tos = sum(1 for j in range(si, sav or nxt_start)
                      if "TimeoutError" in lines[j] and "Streaming interrupted" in lines[j])
            sc = scores.get(name, "?")
            out.append(f"    {name:<20}{fmt_dur(dur):<12}{str(tos):<10}{sc}")

    # --- benchmarks: 'Running benchmark Qwen3-8B: X/N' grouped ---
    bm = []
    for i, l in enumerate(lines):
        m = re.search(r"Running benchmark .*?:\s*(\d+)/(\d+)", l)
        if m and ts(l):
            bm.append((ts(l), int(m.group(1)), int(m.group(2)), i))
    if bm:
        groups = []  # (start_ts, N, count)
        for (t, x, n, i) in bm:
            if x == 1 or not groups or groups[-1][1] != n:
                groups.append([t, n, 1, t])  # start, N, count, last_ts
            else:
                groups[-1][2] += 1
                groups[-1][3] = t
        out.append("\n  BENCHMARKS (groups)")
        for gi, (gstart, n, cnt, glast) in enumerate(groups):
            nxt = groups[gi + 1][0] if gi + 1 < len(groups) else end_t
            dur = (nxt - gstart).total_seconds() if nxt else None
            out.append(f"    {cnt}/{n} runs   start {gstart.strftime('%H:%M:%S')}   "
                       f"~{fmt_dur(dur)}")

    # --- acceptance ---
    acc = next((l.strip() for l in lines if "Acceptance criteria enforcement" in l), None)
    if acc:
        m = re.search(r"enforcement:\s*(\w+)", acc)
        out.append(f"\n  ACCEPTANCE: {m.group(1) if m else acc}")
    fails = [l.strip().lstrip("- ") for l in lines
             if re.search(r"`(evals|benchmarks)\.", l) and "failed" in l]
    for fl in fails:
        out.append(f"    - {fl[:110]}")

    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        try:
            print(summarize(p))
        except FileNotFoundError:
            print(f"\nLOG: {p}\n  (not found)")


if __name__ == "__main__":
    main()
