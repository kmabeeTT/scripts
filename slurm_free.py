#!/usr/bin/env python3
"""
List Slurm-managed machines and their state, optionally filtered by a
regex applied to the machine (node) name. Cross-references squeue so
busy machines show who is holding them and for how long, and shows the
drain/down reason where Slurm has one recorded.

Usage:
  slurm_free.py                 # show all machines, grouped by state
  slurm_free.py -f              # show only free (idle) machines
  slurm_free.py bh-lb           # filter machine names by regex "bh-lb"
  slurm_free.py -f 'c0[1-3]'    # free machines whose name matches c01-c03
  slurm_free.py -p bh-glx-110   # also show each machine's partitions
  slurm_free.py -v              # also print the slurm commands being run
  slurm_free.py -f --reserve    # print an example salloc for a free match
  slurm_free.py --reserve HOST  # print an example salloc for a specific host
  slurm_free.py -q              # only show machines with a pending queue
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

STATE_ORDER = ["FREE", "BUSY", "DRAINING", "DOWN", "OTHER"]

JOBNAME_MAX = 22  # truncate long job names so one job doesn't blow out the column


def truncate(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"

# Machines currently allocated to the forge team, as of 2026-09-21.
# Update this set (and FORGE_NOTES below) as allocations change.
FORGE_MACHINES = {
    # 120-C-Quad1 (8kW, 2AM-12PM ET window only)
    "bh-glx-120-c01u02", "bh-glx-120-c01u08", "bh-glx-120-c01u14", "bh-glx-120-c01u20",
    # 120-C-Quad6 (8kW, 2AM-12PM ET window only)
    "bh-glx-120-c07u02", "bh-glx-120-c07u08", "bh-glx-120-c07u14", "bh-glx-120-c07u20",
    # 120-C-Quad7 (8kW, 2AM-12PM ET window only)
    "bh-glx-120-c08u02", "bh-glx-120-c08u08", "bh-glx-120-c08u14", "bh-glx-120-c08u20",
    # 120-C-Quad8 (8kW, 2AM-12PM ET window only)
    "bh-glx-120-c09u02", "bh-glx-120-c09u08", "bh-glx-120-c09u14", "bh-glx-120-c09u20",
    # 120-C-Quad2 (14kW, 24 hours)
    "bh-glx-120-c02u02", "bh-glx-120-c02u08", "bh-glx-120-c03u02", "bh-glx-120-c03u08",
}

# Per-machine annotation shown in --forge output: (power, availability window).
FORGE_NOTES = {
    **{n: "8kW, 2AM-12PM ET" for n in (
        "bh-glx-120-c01u02", "bh-glx-120-c01u08", "bh-glx-120-c01u14", "bh-glx-120-c01u20",
        "bh-glx-120-c07u02", "bh-glx-120-c07u08", "bh-glx-120-c07u14", "bh-glx-120-c07u20",
        "bh-glx-120-c08u02", "bh-glx-120-c08u08", "bh-glx-120-c08u14", "bh-glx-120-c08u20",
        "bh-glx-120-c09u02", "bh-glx-120-c09u08", "bh-glx-120-c09u14", "bh-glx-120-c09u20",
    )},
    **{n: "14kW, 24h" for n in (
        "bh-glx-120-c02u02", "bh-glx-120-c02u08", "bh-glx-120-c03u02", "bh-glx-120-c03u08",
    )},
}

COLOR = {
    "FREE": "\033[32m",      # green
    "BUSY": "\033[36m",      # cyan
    "DRAINING": "\033[33m",  # yellow
    "DOWN": "\033[31m",      # red
    "OTHER": "\033[35m",     # magenta
    "QUEUED": "\033[93m",    # bright yellow
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}


VERBOSE = False
_shown_commands = set()


def run(cmd, note=None):
    """Run a slurm command. If VERBOSE is set, print the command (once per
    distinct `note`, so a command that repeats per-job doesn't spam)."""
    if VERBOSE:
        key = note or " ".join(cmd)
        if key not in _shown_commands:
            _shown_commands.add(key)
            repeat = "  # repeated once per running job" if note else ""
            print(f"$ {' '.join(shlex.quote(c) for c in cmd)}{repeat}")
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def classify(raw_state):
    """Collapse Slurm's compound node states (e.g. ALLOCATED+DRAIN) into
    one bucket we care about for "can I use this machine" purposes."""
    s = raw_state.upper()
    if "DOWN" in s:
        return "DOWN"
    if "DRAIN" in s:
        return "DRAINING"
    if "IDLE" in s:
        return "FREE"
    if "ALLOCATED" in s or "MIXED" in s or "COMPLETING" in s:
        return "BUSY"
    return "OTHER"


def parse_nodes():
    """One dict per physical machine, deduped (scontrol -o gives exactly
    one line per node, unlike sinfo -N which repeats a node once per
    partition it belongs to)."""
    text = run(["scontrol", "show", "node", "-o"])
    nodes = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name = re.search(r"NodeName=(\S+)", line)
        state = re.search(r"\bState=(\S+)", line)
        parts = re.search(r"Partitions=(\S+)", line)
        reason = re.search(r"Reason=(.*)$", line)
        if not name or not state:
            continue
        n = name.group(1)
        nodes[n] = {
            "name": n,
            "raw_state": state.group(1),
            "state": classify(state.group(1)),
            "partitions": parts.group(1).split(",") if parts else [],
            "reason": reason.group(1).strip() if reason else "",
        }
    return nodes


def parse_jobs():
    """Map node name -> dict(user, jobname, elapsed, jobid, time_limit, time_left) for running jobs.
    time_limit/time_left are Slurm-formatted strings (e.g. "1-00:00:00", "UNLIMITED", "NOT_SET")."""
    text = run(["squeue", "-h", "-a", "-o", "%i|%j|%u|%t|%M|%l|%L|%N", "--states=R"])
    owner = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        jobid, jobname, user, st, elapsed, time_limit, time_left, nodelist = line.split("|", 7)
        if not nodelist:
            continue
        hosts = run(["scontrol", "show", "hostnames", nodelist],
                    note="scontrol show hostnames <nodelist>").splitlines()
        for h in hosts:
            h = h.strip()
            if h:
                owner[h] = {
                    "user": user, "jobname": jobname, "elapsed": elapsed, "jobid": jobid,
                    "time_limit": time_limit, "time_left": time_left,
                }
    return owner


def format_duration(delta):
    total = int(delta.total_seconds())
    if total < 60:
        return f"{total}s"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def parse_pending():
    """Map node name -> list of pending jobs that explicitly requested it
    (via --nodelist), so you can tell if reserving that node would put you
    behind someone else already waiting on it."""
    text = run(["squeue", "-h", "-a", "-o", "%i|%u|%j|%r|%V|%n", "--states=PD"])
    pending = {}
    now = datetime.now()
    for line in text.splitlines():
        if not line.strip():
            continue
        jobid, user, jobname, reason, submit, reqnodes = line.split("|", 5)
        if not reqnodes:
            continue
        hosts = run(["scontrol", "show", "hostnames", reqnodes],
                    note="scontrol show hostnames <nodelist>").splitlines()
        try:
            waiting = format_duration(now - datetime.fromisoformat(submit))
        except ValueError:
            waiting = "?"
        for h in hosts:
            h = h.strip()
            if h:
                pending.setdefault(h, []).append({
                    "jobid": jobid, "user": user, "jobname": jobname,
                    "reason": reason, "waiting": waiting,
                })
    return pending


def compact_slurm_time(s):
    """Compact a squeue %l/%L duration string ("7-00:00:00", "23:29:34",
    "UNLIMITED", "NOT_SET") into a short form like "7d0h" / "23h29m"."""
    if s in ("UNLIMITED", "NOT_SET", "INVALID"):
        return s.lower()
    days = 0
    rest = s
    if "-" in s:
        days_str, rest = s.split("-", 1)
        days = int(days_str)
    parts = rest.split(":")
    hours = int(parts[0]) if len(parts) > 0 else 0
    minutes = int(parts[1]) if len(parts) > 1 else 0
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default=None,
                     help="regex (grep -E style) to filter machine names, e.g. 'bh-lb' or 'c0[1-3]'")
    ap.add_argument("-f", "--free-only", action="store_true",
                     help="only show free (idle, usable) machines")
    ap.add_argument("-b", "--busy-only", action="store_true",
                     help="only show busy (allocated) machines")
    ap.add_argument("-p", "--show-partitions", action="store_true",
                     help="also print the partition(s) each machine belongs to")
    ap.add_argument("--forge", action="store_true",
                     help="only show machines currently allocated to the forge team")
    ap.add_argument("-i", "--ignore-case", action="store_true", default=True,
                     help="case-insensitive pattern match (default: on)")
    ap.add_argument("--no-color", action="store_true", help="disable colored output")
    ap.add_argument("-v", "--show-commands", action="store_true",
                     help="print the underlying slurm commands as they're run (for learning)")
    ap.add_argument("--reserve", nargs="?", const="__auto__", default=None, metavar="MACHINE",
                     help="print an example salloc command to reserve MACHINE (or the first free "
                          "match in the current results if no name given). Does not run anything.")
    ap.add_argument("-q", "--pending-only", action="store_true",
                     help="only show machines that have a pending (queued) request against them")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.show_commands

    use_color = sys.stdout.isatty() and not args.no_color

    def c(state, text):
        if not use_color:
            return text
        return f"{COLOR.get(state, '')}{text}{COLOR['RESET']}"

    try:
        nodes = parse_nodes()
        owners = parse_jobs() if not args.free_only else {}
        pending = parse_pending()
    except subprocess.CalledProcessError as e:
        print(f"error running slurm command: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("error: slurm commands (scontrol/squeue) not found on this host", file=sys.stderr)
        sys.exit(1)

    rx = None
    if args.pattern:
        flags = re.IGNORECASE if args.ignore_case else 0
        rx = re.compile(args.pattern, flags)

    def passes_filters(n):
        if rx and not rx.search(n["name"]):
            return False
        if args.forge and n["name"] not in FORGE_MACHINES:
            return False
        return True

    rows = []
    for n in nodes.values():
        if not passes_filters(n):
            continue
        if args.free_only and n["state"] != "FREE":
            continue
        if args.busy_only and n["state"] != "BUSY":
            continue
        if args.pending_only and n["name"] not in pending:
            continue
        rows.append(n)

    rows.sort(key=lambda n: natural_key(n["name"]))

    counts = {s: 0 for s in STATE_ORDER}
    for n in nodes.values():
        if not passes_filters(n):
            continue
        counts[n["state"]] += 1
    total = sum(counts.values())

    filter_desc = []
    if args.pattern:
        filter_desc.append(f"pattern '{args.pattern}'")
    if args.forge:
        filter_desc.append("forge team")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""

    summary = " | ".join(c(s, f"{s}: {counts[s]}") for s in STATE_ORDER if counts[s])
    print(f"{c('BOLD', 'Machines')} ({total} match{filter_str}): {summary}\n")

    queued_matching = [n for n in nodes.values() if passes_filters(n) and n["name"] in pending]
    if queued_matching:
        print(f"note: {len(queued_matching)} machine(s) matching filter have a pending queue "
              f"(marked QUEUED below)\n")

    if args.forge:
        missing = FORGE_MACHINES - set(nodes.keys())
        if missing:
            print(f"note: {len(missing)} forge machine(s) not known to slurm: {', '.join(sorted(missing))}\n")

    if not rows:
        print("(no machines matched)")
    else:
        col_names = ["MACHINE", "STATE", "USER", "JOB", "ELAPSED", "JOBID", "LIMIT", "LEFT", "REASON"]
        if args.show_partitions:
            col_names.append("PARTITIONS")
        if args.forge:
            col_names.append("NOTE")
        col_names.append("QUEUED")

        cells = []
        for n in rows:
            cell = {name: "" for name in col_names}
            cell["MACHINE"] = n["name"]
            cell["STATE"] = n["state"]
            if n["state"] == "BUSY" and n["name"] in owners:
                o = owners[n["name"]]
                cell["USER"] = o["user"]
                cell["JOB"] = truncate(o["jobname"], JOBNAME_MAX)
                cell["ELAPSED"] = o["elapsed"]
                cell["JOBID"] = f"#{o['jobid']}"
                cell["LIMIT"] = compact_slurm_time(o["time_limit"])
                cell["LEFT"] = compact_slurm_time(o["time_left"])
            elif n["state"] in ("DOWN", "DRAINING") and n["reason"]:
                cell["REASON"] = n["reason"]
            if args.show_partitions:
                cell["PARTITIONS"] = ",".join(n["partitions"])
            if args.forge and n["name"] in FORGE_NOTES:
                cell["NOTE"] = FORGE_NOTES[n["name"]]
            if n["name"] in pending:
                jobs = pending[n["name"]]
                first = jobs[0]
                extra = f" (+{len(jobs) - 1} more)" if len(jobs) > 1 else ""
                cell["QUEUED"] = f"{first['user']} waiting {first['waiting']}{extra}"
            cells.append(cell)

        widths = {name: max(len(name), max((len(rc[name]) for rc in cells), default=0)) for name in col_names}

        header = "  ".join(f"{name:<{widths[name]}}" for name in col_names)
        print(c("BOLD", header.rstrip()))
        for n, rc in zip(rows, cells):
            parts = []
            for name in col_names:
                text = rc[name]
                if name == "STATE":
                    parts.append(c(n["state"], f"{text:<{widths[name]}}"))
                elif name == "QUEUED" and text:
                    parts.append(c("QUEUED", f"{text:<{widths[name]}}"))
                else:
                    parts.append(f"{text:<{widths[name]}}")
            print("  ".join(parts).rstrip())

    if args.reserve is not None:
        print()
        if args.reserve == "__auto__":
            target = next((n["name"] for n in rows if n["state"] == "FREE"), None)
            if target is None:
                print("no free machine in the current results to build a --reserve example "
                      "from; pass one explicitly, e.g. --reserve bh-glx-110-a01u02", file=sys.stderr)
                return
        else:
            target = args.reserve

        node = nodes.get(target)
        if node is None:
            print(f"unknown machine '{target}' (not in `scontrol show node`)", file=sys.stderr)
            return

        partition = node["partitions"][0] if node["partitions"] else "<partition>"
        print(c("BOLD", f"Example reservation command for {target}") +
              " (this only prints the command, it doesn't run it):\n")
        job_name = f"{os.environ.get('USER', 'me')}-prefill"
        print(f"  salloc --no-shell -w {target} -p {partition} --time=24:00:00 --job-name={job_name}")
        if node["state"] != "FREE":
            print(f"\n  note: {target} is currently {node['state']}, not free -- salloc will "
                  f"queue (PD) until it's released.")
        if target in pending:
            jobs = pending[target]
            print(f"\n  heads up: {len(jobs)} job(s) already queued for {target} -- you'd be "
                  f"joining behind:")
            for j in jobs:
                print(f"    {j['user']:<14} job={j['jobname']:<20} waiting {j['waiting']:<8} "
                      f"(jobid {j['jobid']})")
        print("\n  # --no-shell backgrounds the allocation immediately -- it survives an SSH")
        print("  # disconnect (unlike a bare `salloc` with no flags, which dies with your shell).")
        print(f"  # Connect to the machine once it's allocated:")
        print(f"  ssh {target}")
        print(f"  # When done, release it explicitly: scancel $(squeue -h -u $USER -w {target} -o %i)")
        print("  # Also: node assignments rotate by team/time -- your job can be killed if it")
        print("  # runs outside your team's assigned window (see the shared allocation sheet).")


if __name__ == "__main__":
    main()
