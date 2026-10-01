#!/usr/bin/env python3
"""tt_activity — when did someone last use the TT devices, and is anyone using them now?

Two parts:
  1. A "last use" report from whatever evidence survives on the box (works across users,
     even under hidepid).
  2. A watch: poll every --interval seconds for --duration, print ONE line when activity
     starts and ONE when it stops, keep a live status line + timeline, and end with a
     summary. Ctrl-C ends early and still prints the summary.

Usage:
    python3 ~/scripts/tt_activity.py                    # report, then watch 1h every 30s
    python3 ~/scripts/tt_activity.py --once             # report only
    python3 ~/scripts/tt_activity.py -d 15m -i 10       # watch 15 min, 10 s polls
    python3 ~/scripts/tt_activity.py --ignore-me        # don't count your own jobs

SIGNALS (all readable by any user; none needs /proc of the other user)
  pcie   /sys/class/tenstorrent/*/pcie_perf_counters — cumulative PCIe word counters per
         chip. The WRITE counters (host->device and device->host) sit at exactly 0 on an
         idle chip, while the read counters tick from firmware telemetry, so only writes
         count. A real job moves 1e5..1e8 words; a `tt-smi -s` moves ~55 per chip, so
         small deltas are labelled "probe" rather than "job".
  lock   /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_<N>_* — UMD's per-chip mutex; owner_pid at
         byte 52 (see tt-devs.sh). Held from device open to close. The holder's user comes
         from cgroup v2 (user-<uid>.slice), which hidepid does not filter.
  clk    tt_aiclk > 900 MHz means a chip is open (idle is 800).
  Users are named only from the lock holder. PCIe/aiclk activity with no lock (tt-smi,
  a containerized job, a tool that bypasses UMD) is reported as "user unknown".

LAST-USE EVIDENCE (startup report)
  - Episodes this tool recorded in earlier watches (~/.cache/tt-activity/events.log).
  - Newest /dev/shm file per user (MPI sm_segment, tt_device_*_memory, ...). Its mtime is
    when the file was created/written, so it bounds a job's start, not its end.
  - Stale CHIP_IN_USE locks: the owner crashed without releasing (time unknown).
  - Which chips were written to at all since boot (pcie counters are cumulative).
  There is no record of "last device close" on the box itself; the watch is the only way
  to get exact times, which is why it logs what it sees for the next run.
"""

from __future__ import annotations

import argparse
import glob
import os
import pwd
import re
import shutil
import signal
import struct
import sys
import time
from dataclasses import dataclass, field

SYS = "/sys/class/tenstorrent"
SHM = "/dev/shm"
CGROUP_USERS = "/sys/fs/cgroup/user.slice"
LOG = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                   "tt-activity", "events.log")
WRITE_COUNTERS = ("mst_nonposted_wr_data_word_sent", "mst_posted_wr_data_word_sent",
                  "slv_nonposted_wr_data_word_received", "slv_posted_wr_data_word_received")
JOB_WORDS = 10_000      # per-poll write words on any chip above which it is a "job"
PROBE_HINT = "likely tt-smi -s or a telemetry read, not a job"
IDLE_CLK = 900
ME = pwd.getpwuid(os.getuid()).pw_name

TTY = sys.stdout.isatty()
def _c(code): return (lambda s: f"\033[{code}m{s}\033[0m") if TTY else (lambda s: str(s))
BOLD, DIM, RED, GRN, YEL, CYN = _c(1), _c(2), _c(31), _c(32), _c(33), _c(36)


# ---- helpers ---------------------------------------------------------------
def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def user_name(uid):
    try:
        return pwd.getpwuid(int(uid)).pw_name
    except (KeyError, ValueError):
        return f"uid{uid}"


def hms(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


def when(ts):
    """'21:32 (14m ago)', or with the date when it is not today."""
    lt = time.localtime(ts)
    fmt = "%H:%M" if time.strftime("%F", lt) == time.strftime("%F") else "%b %d %H:%M"
    return f"{time.strftime(fmt, lt)} ({dur(time.time() - ts)} ago)"


def dur(s):
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def chip_ranges(chips):
    """{0,1,2,5,7,8} -> '0-2,5,7-8'."""
    out, run = [], []
    for c in sorted(chips):
        if run and c == run[-1] + 1:
            run.append(c)
        else:
            if run:
                out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def parse_duration(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration {s!r} (e.g. 90s, 15m, 1h)")
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "": 60}[m.group(2)]


def chips():
    out = []
    for d in glob.glob(f"{SYS}/tenstorrent!*"):
        n = d.rsplit("!", 1)[1]
        if n.isdigit():
            out.append(int(n))
    return sorted(out)


# ---- signals ---------------------------------------------------------------
def pcie_writes(chip):
    tot = 0
    for name in WRITE_COUNTERS:
        for lane in (0, 1):
            v = read_int(f"{SYS}/tenstorrent!{chip}/pcie_perf_counters/{name}{lane}")
            tot += v or 0
    return tot


def aiclk(chip):
    return read_int(f"{SYS}/tenstorrent!{chip}/tt_aiclk")


def pid_state(pid):
    """'alive' (ours or hidden) or 'dead'. Signal permission checks ignore hidepid."""
    try:
        os.kill(pid, 0)
        return "alive"
    except PermissionError:
        return "alive"
    except ProcessLookupError:
        return "dead"


def chip_locks():
    """{chip: owner_pid} for every CHIP_IN_USE lock with a non-zero owner."""
    out = {}
    for f in glob.glob(f"{SHM}/TT_UMD_LOCK.CHIP_IN_USE_*"):
        m = re.search(r"CHIP_IN_USE_(\d+)_", f)
        if not m:
            continue
        try:
            with open(f, "rb") as fh:
                data = fh.read(56)
        except OSError:
            continue
        if len(data) != 56:
            continue
        pid = struct.unpack_from("<i", data, 52)[0]
        if pid:
            out[int(m.group(1))] = pid
    return out


def cgroup_pid_users():
    """{pid: user} from cgroup v2 user slices (not subject to hidepid)."""
    out = {}
    for procs in glob.glob(f"{CGROUP_USERS}/user-*.slice/**/cgroup.procs", recursive=True):
        m = re.search(r"user-(\d+)\.slice", procs)
        try:
            with open(procs) as f:
                pids = f.read().split()
        except OSError:
            continue
        u = user_name(m.group(1))
        for p in pids:
            out[int(p)] = u
    return out


@dataclass
class Snap:
    t: float
    writes: dict
    clk: dict
    locks: dict      # chip -> pid (live only)


def snapshot(chip_list):
    locks = {c: p for c, p in chip_locks().items() if pid_state(p) == "alive"}
    return Snap(time.time(), {c: pcie_writes(c) for c in chip_list},
                {c: aiclk(c) for c in chip_list}, locks)


# ---- last-use report -------------------------------------------------------
def read_log():
    try:
        with open(LOG) as f:
            return [l.rstrip("\n").split("\t") for l in f if l.count("\t") >= 4]
    except OSError:
        return []


def shm_evidence():
    """{user: (newest_mtime, kinds)} for /dev/shm files, minus root's boot-time locks."""
    per = {}
    for f in os.listdir(SHM):
        p = os.path.join(SHM, f)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if st.st_uid == 0 or not os.path.isfile(p):
            continue
        if f.startswith("TT_UMD_LOCK."):
            kind = "UMD lock created"
        elif f.startswith("sm_segment."):
            kind = "MPI shm segment"
        elif f.startswith("tt_device_") and f.endswith("_memory"):
            kind = "tt-metal device memory"
        else:
            kind = f[:40]
        u = user_name(st.st_uid)
        newest, kinds = per.get(u, (0, {}))
        kinds[kind] = max(kinds.get(kind, 0), st.st_mtime)
        per[u] = (max(newest, st.st_mtime), kinds)
    return per


def boot_time():
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("btime"):
                return int(line.split()[1])
    return None


def report(chip_list, snap, ignore_me):
    host = os.uname().nodename
    print(BOLD(f"TT device activity — {host}, {len(chip_list)} chips"))
    print(DIM("-" * 64))

    # now
    pu = cgroup_pid_users() if snap.locks else {}
    held = {}
    for c, p in snap.locks.items():
        held.setdefault(pu.get(p, "?"), set()).add(c)
    open_clk = {c for c, v in snap.clk.items() if v and v > IDLE_CLK}
    if held:
        for u, cs in held.items():
            print(f"  {RED('IN USE NOW')}  chips {chip_ranges(cs)} ({len(cs)}) by {BOLD(u)}")
    elif open_clk:
        print(f"  {YEL('OPEN NOW')}    chips {chip_ranges(open_clk)} clocked up, no lock holder "
              f"(starting up, or a container)")
    else:
        print(f"  Now:  {GRN('all chips free')}")

    # recorded by earlier watches
    log = [e for e in read_log() if not (ignore_me and e[3] == ME)]
    if log:
        last = max(log, key=lambda e: float(e[1]))
        kind = last[5] if len(last) > 5 else "job"
        ran = dur(float(last[1]) - float(last[0]))
        print(f"  Last seen by this tool:  {BOLD(when(float(last[1])))}  {last[3]}  "
              f"chips {last[4]}  {DIM(f'({kind}, ran {ran})')}")
    else:
        print(f"  Last seen by this tool:  {DIM('nothing recorded yet (run a watch)')}")

    # /dev/shm leftovers
    ev = shm_evidence()
    if ignore_me:
        ev.pop(ME, None)
    if ev:
        print(f"  Newest /dev/shm file per user {DIM('(mtime = created/written; bounds job START)')}:")
        for u, (newest, kinds) in sorted(ev.items(), key=lambda kv: -kv[1][0]):
            top = sorted(kinds.items(), key=lambda kv: -kv[1])[:3]
            what = ", ".join(k for k, _ in top)
            print(f"    {CYN(f'{u:<12}')} {when(newest):<22} {DIM(what)}")

    # stale locks
    stale = {c: p for c, p in chip_locks().items() if pid_state(p) == "dead"}
    if stale:
        print(f"  {YEL('Stale locks')} on chips {chip_ranges(stale)} — owner crashed without "
              f"releasing (UMD recovers on next open)")

    bt = boot_time()
    used = [c for c, w in snap.writes.items() if w > 0]
    print(f"  Since boot {DIM(when(bt) if bt else '')}: "
          f"{len(used)}/{len(chip_list)} chips have been written to")
    print(DIM("-" * 64))


# ---- watch -----------------------------------------------------------------
@dataclass
class Episode:
    start: float
    end: float
    chips: set = field(default_factory=set)
    users: set = field(default_factory=set)
    peak: int = 0
    kind: str = "probe"


def lock_users(cur, active, pid_users):
    """Users holding a CHIP_IN_USE lock on any active chip."""
    users = {pid_users.get(p, "?") for c, p in cur.locks.items() if c in active}
    users.discard("?")
    return users


def watch(chip_list, duration, interval, ignore_me):
    polls = max(1, int(duration // interval))
    marks = []                     # one char per poll for the timeline
    episodes: list[Episode] = []
    cur_ep: Episode | None = None
    t0 = time.time()
    prev = snapshot(chip_list)
    width = shutil.get_terminal_size((100, 20)).columns

    print(f"Watching {dur(duration)} every {interval:g}s ({polls} polls) from {hms(t0)}. "
          f"{DIM('Ctrl-C to stop early.')}")
    if ignore_me:
        print(DIM(f"  ignoring activity attributed to {ME}"))

    def status(i):
        if not TTY:
            return
        elapsed = time.time() - t0
        tail = "".join(marks)[-max(10, width - 62):]
        state = (RED(f"ACTIVE chips {chip_ranges(cur_ep.chips)}") if cur_ep else GRN("idle"))
        last = f"last {hms(episodes[-1].end)}" if episodes else "no activity yet"
        if cur_ep:
            last = f"since {hms(cur_ep.start)}"
        line = f"\r\033[K[{dur(elapsed):>6}/{dur(duration)}] {i:>3}/{polls} {tail}  {state} {DIM(last)}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def emit(msg):
        if TTY:
            sys.stdout.write("\r\033[K")
        print(msg)

    def close_episode(ep, now):
        episodes.append(ep)
        who = ",".join(sorted(ep.users)) or "user unknown"
        emit(f"{hms(now)}  {GRN('■ stopped')}  chips {chip_ranges(ep.chips)}  {who}  "
             f"{DIM(f'ran ~{dur(ep.end - ep.start)}, {ep.kind}, peak {ep.peak:,} words/poll')}")
        try:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            with open(LOG, "a") as f:
                f.write(f"{ep.start:.0f}\t{ep.end:.0f}\t{os.uname().nodename}\t{who}\t"
                        f"{chip_ranges(ep.chips)}\t{ep.kind}\n")
        except OSError:
            pass

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    i = 0
    status(i)
    try:
        while i < polls and not stop["now"]:
            deadline = t0 + (i + 1) * interval
            while time.time() < deadline and not stop["now"]:
                time.sleep(min(1.0, max(0.0, deadline - time.time())))
                status(i)
            if stop["now"]:
                break
            cur = snapshot(chip_list)
            i += 1
            delta = {c: cur.writes[c] - prev.writes.get(c, cur.writes[c]) for c in chip_list}
            active = {c for c in chip_list
                      if delta[c] > 0 or c in cur.locks or (cur.clk[c] or 0) > IDLE_CLK}
            pid_users = cgroup_pid_users() if cur.locks else {}
            users = lock_users(cur, active, pid_users) if active else set()
            if ignore_me and active and users == {ME}:
                active = set()
            peak = max(delta.values(), default=0)
            kind = "job" if (peak >= JOB_WORDS or cur.locks) else "probe"

            if active:
                marks.append("█" if kind == "job" else "▪")
                who = ",".join(sorted(users)) or "user unknown"
                if cur_ep is None:
                    cur_ep = Episode(prev.t, cur.t)
                    src = "lock" if cur.locks else ("pcie" if peak else "aiclk")
                    hint = f" - {PROBE_HINT}" if kind == "probe" else ""
                    emit(f"{hms(cur.t)}  {RED('▶ started')}  chips {chip_ranges(active)} "
                         f"({len(active)})  {BOLD(who)}  {DIM(f'[{src}, {peak:,} words{hint}]')}")
                cur_ep.end = cur.t
                cur_ep.chips |= active
                cur_ep.users |= users
                cur_ep.peak = max(cur_ep.peak, peak)
                if kind == "job":
                    cur_ep.kind = "job"
            else:
                marks.append("·")
                if cur_ep is not None:
                    close_episode(cur_ep, cur.t)
                    cur_ep = None
            prev = cur
            status(i)
    finally:
        if cur_ep is not None:
            close_episode(cur_ep, prev.t)
        if TTY:
            sys.stdout.write("\r\033[K")
        summary(t0, prev.t, interval, marks, episodes)


def summary(t0, t1, interval, marks, episodes):
    print(DIM("-" * 64))
    print(BOLD(f"Summary {hms(t0)} -> {hms(t1)}  ({dur(t1 - t0)}, {len(marks)} polls)"))
    # timeline, 60 polls per row (30 min at 30s), labelled with each row's start time
    row = 60
    for k in range(0, len(marks), row):
        print(f"  {DIM(hms(t0 + k * interval)[:5])}  {''.join(marks[k:k + row])}")
    print(DIM("  · idle   ▪ probe (tiny PCIe writes, e.g. tt-smi -s)   █ job (lock held / heavy traffic)"))
    if not episodes:
        print(f"  {GRN('No device activity seen.')}")
        return
    busy = sum(e.end - e.start for e in episodes)
    print(f"  {len(episodes)} episode(s), ~{dur(busy)} active:")
    for e in episodes:
        hint = f"  {DIM(f'({PROBE_HINT})')}" if e.kind == "probe" else ""
        print(f"    {hms(e.start)}-{hms(e.end)}  {e.kind:<5}  chips {chip_ranges(e.chips):<10} "
              f"{','.join(sorted(e.users)) or 'user unknown'}{hint}")
    print(f"  Last activity: {BOLD(hms(episodes[-1].end))}")
    if all(e.kind == "probe" for e in episodes):
        print(f"  {GRN('No jobs ran')} - every episode was probe-sized (tt-smi -s moves only tens to hundreds of words/chip "
              f"and takes no lock); the chips stayed free.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--duration", type=parse_duration, default=3600,
                    help="how long to watch: 90s, 15m, 1h (bare number = minutes). Default 1h")
    ap.add_argument("-i", "--interval", type=float, default=30, help="poll seconds (default 30)")
    ap.add_argument("--once", action="store_true", help="last-use report only, no watch")
    ap.add_argument("--ignore-me", action="store_true",
                    help=f"don't count activity attributed to {ME}")
    a = ap.parse_args()

    chip_list = chips()
    if not chip_list:
        sys.exit(f"no {SYS}/tenstorrent!* devices found")
    snap = snapshot(chip_list)
    report(chip_list, snap, a.ignore_me)
    if not a.once:
        watch(chip_list, a.duration, a.interval, a.ignore_me)


if __name__ == "__main__":
    main()
