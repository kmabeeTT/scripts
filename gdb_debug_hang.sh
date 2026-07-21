#!/usr/bin/env bash
# gdb_debug_hang.sh -- diagnose a hung/stuck native process (e.g. a wedged
# VLLM::EngineCore stuck on a TT device read) by finding the actively
# spinning thread(s) via a /proc CPU-time delta sample, then dumping both
# a native backtrace (gdb) and a Python-level stack (py-spy) for those
# threads.
#
# Why CPU-delta first: a genuine native-level hang (e.g. a TT device
# completion-queue wait that never returns) shows as ONE thread pegged near
# 100% CPU while every other thread (RPC/gloo/tqdm workers, etc.) sits
# idle. Sampling utime+stime from /proc/<pid>/task/*/stat twice, N seconds
# apart, finds that thread directly instead of guessing which of 100+
# threads to inspect.
#
# Usage:
#   gdb_debug_hang.sh [--pid PID] [--pattern PATTERN] [--sample-secs N]
#                      [--top N] [-h|--help]
#
# Flags:
#   --pid PID        target process (default: auto-detect via --pattern).
#   --pattern PAT    process name/cmdline pattern for auto-detect via
#                     `pgrep -f` (default: "VLLM::EngineCore"). Errors out
#                     if it matches zero or more than one process.
#   --sample-secs N  CPU-delta sample window in seconds (default 3).
#   --top N          number of hottest threads to dump full gdb backtraces
#                     for (default 1).
#   -h|--help        this help
#
# Needs: passwordless `sudo -n gdb` and `sudo -n py-spy` (ptrace access to
# another user's/root-owned process). py-spy path defaults to the tt-xla
# venv's copy; override with PY_SPY=/path/to/py-spy.
#
# Also reports which /dev/tenstorrent/* devices the target holds, since a
# hung engine still holding its device is usually the reason a fresh
# server/test can't start until this one is killed.
set -uo pipefail

PID=""
PATTERN="VLLM::EngineCore"
SAMPLE_SECS=3
TOP_N=1
PY_SPY="${PY_SPY:-$HOME/tt-xla/venv/bin/py-spy}"

while [ $# -gt 0 ]; do
  case "$1" in
    --pid) PID=${2:?--pid needs a value}; shift 2 ;;
    --pid=*) PID="${1#*=}"; shift ;;
    --pattern) PATTERN=${2:?--pattern needs a value}; shift 2 ;;
    --pattern=*) PATTERN="${1#*=}"; shift ;;
    --sample-secs) SAMPLE_SECS=${2:?--sample-secs needs a value}; shift 2 ;;
    --sample-secs=*) SAMPLE_SECS="${1#*=}"; shift ;;
    --top) TOP_N=${2:?--top needs a value}; shift 2 ;;
    --top=*) TOP_N="${1#*=}"; shift ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; YEL=$'\e[33m'; CYN=$'\e[36m'; RST=$'\e[0m'
else
  BOLD=; DIM=; YEL=; CYN=; RST=
fi

if [ -z "$PID" ]; then
  mapfile -t matches < <(pgrep -f "$PATTERN")
  case "${#matches[@]}" in
    0) echo "ERROR: no process matching pattern '$PATTERN' (try --pid or --pattern)" >&2; exit 1 ;;
    1) PID="${matches[0]}" ;;
    *) echo "ERROR: pattern '$PATTERN' matched multiple PIDs: ${matches[*]} (pick one with --pid)" >&2; exit 1 ;;
  esac
fi
[ -d "/proc/$PID" ] || { echo "ERROR: no such process $PID" >&2; exit 1; }

echo "${BOLD}Target:${RST} PID $PID ($(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-100))"

# ---- 1. per-thread CPU-time delta sample -----------------------------------
# Parse /proc/<pid>/task/<tid>/stat robustly: comm (field 2, in parens) can
# itself contain spaces (seen in the wild: "Tracy Sampling", "RealtimeProfile"
# threads), which shifts naive whitespace-field-counting. Split on the LAST
# ") " instead -- pid and comm are the only fields before it, so what follows
# always starts at state(3), making utime/stime the 12th/13th fields of that
# remainder.
read_ticks() {
  local stat="$1" rest
  rest="${stat##*) }"
  read -r _ _ _ _ _ _ _ _ _ _ _ utime stime _ <<<"$rest"
  echo $((utime + stime))
}

echo
echo "${BOLD}Sampling per-thread CPU time over ${SAMPLE_SECS}s...${RST}"
declare -A T0
for t in /proc/"$PID"/task/*/; do
  tid=$(basename "$t")
  stat=$(cat "$t/stat" 2>/dev/null) || continue
  T0["$tid"]=$(read_ticks "$stat")
done
sleep "$SAMPLE_SECS"
declare -A DELTA
for tid in "${!T0[@]}"; do
  stat=$(cat "/proc/$PID/task/$tid/stat" 2>/dev/null) || continue
  DELTA["$tid"]=$(( $(read_ticks "$stat") - T0["$tid"] ))
done

mapfile -t sorted < <(for tid in "${!DELTA[@]}"; do echo "${DELTA[$tid]} $tid"; done | sort -rn)

echo "${DIM}top threads by CPU ticks used during the sample window:${RST}"
count=0
HOT_TIDS=()
for line in "${sorted[@]}"; do
  read -r delta tid <<<"$line"
  comm=$(cat "/proc/$PID/task/$tid/comm" 2>/dev/null)
  printf '  %sTID %-8s%s delta=%-6s comm=%s\n' "$CYN" "$tid" "$RST" "$delta" "$comm"
  count=$((count + 1))
  if [ "$delta" -gt 0 ] && [ "${#HOT_TIDS[@]}" -lt "$TOP_N" ]; then
    HOT_TIDS+=("$tid")
  fi
  [ "$count" -ge 15 ] && break
done

if [ ${#HOT_TIDS[@]} -eq 0 ]; then
  echo "${YEL}No thread accumulated CPU time during the sample window -- process looks fully idle/blocked (e.g. waiting on a lock/queue), not spinning. Skipping gdb (nothing to target); py-spy dump below may still show which Python frame everything is parked in.${RST}"
fi

# ---- 2. py-spy: python-level stacks -----------------------------------------
echo
echo "${BOLD}py-spy dump (Python-level stacks)${RST}"
if [ -x "$PY_SPY" ]; then
  sudo -n "$PY_SPY" dump --pid "$PID" 2>&1
else
  echo "${YEL}py-spy not found/executable at $PY_SPY -- set PY_SPY=/path/to/py-spy${RST}"
fi

# ---- 3. gdb: native backtrace for the hot thread(s) -------------------------
if [ ${#HOT_TIDS[@]} -gt 0 ]; then
  echo
  echo "${BOLD}gdb native backtrace(s) for hot thread(s)${RST}"
  ALL_BT=$(sudo -n gdb -p "$PID" -batch -ex "thread apply all bt" 2>&1)
  for tid in "${HOT_TIDS[@]}"; do
    echo "${DIM}--- TID $tid ---${RST}"
    printf '%s\n' "$ALL_BT" | awk -v tid="$tid" '
      /^Thread [0-9]+ \(/ { in_block = ($0 ~ ("LWP " tid ")")) }
      in_block { print }
    '
  done
fi

# ---- 4. TT device holders ---------------------------------------------------
echo
echo "${BOLD}TT device holders${RST}"
for d in /dev/tenstorrent/*; do
  holders=$(fuser "$d" 2>/dev/null)
  [ -n "$holders" ] && printf '  %s: PID(s)%s\n' "$d" "$holders"
done
exit 0
