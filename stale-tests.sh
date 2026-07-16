#!/usr/bin/env bash
# stale-tests — find and report stale pytest/python/test processes, AND
# stale processes holding TCP listen ports.
#
# Two independent sections:
#   1. Stale pytest/python/test processes: pytest, or any python process
#      whose command line mentions "test" -- flagged if zombie/defunct,
#      stopped (e.g. forgotten after Ctrl-Z), orphaned (reparented to
#      init), or simply running longer than a threshold.
#   2. Stale port holders: ANY process (regardless of name) holding an open
#      TCP listen socket, with the same staleness flags. This catches the
#      other common cause of "why won't my new server start" -- a leftover
#      front-end (uvicorn, vllm serve, etc.) or its orphaned engine child
#      still squatting the port after a crash, e.g. a uvicorn that died on
#      "address already in use" but whose spawned worker process is still
#      alive and holding the port's socket fd.
# Both sections cross-reference whether a flagged process is also holding a
# Tenstorrent device (see tt-devs.sh), since that's usually the actual
# problem a stale process causes here.
#
# Suggested alias:
#     alias staletests='~/scripts/stale-tests.sh'
#
# Usage:
#   stale-tests.sh [--min-minutes N] [--all] [--no-ports] [--kill] [-h|--help]
#
# Flags:
#   --min-minutes N   flag running (non-zombie/stopped) processes older than
#                      N minutes (default 15). Zombie/stopped/orphaned
#                      processes are always flagged regardless of age.
#   --all             also list matching/port-holding processes that were
#                      NOT flagged -- useful to sanity-check the match, or to
#                      see everything currently listening.
#   --no-ports        skip section 2 (port holders) -- faster, test-only.
#   --kill            after reporting, interactively offer to kill -9 each
#                      flagged process (asks once per process; default no).
#   -h|--help         this help
set -uo pipefail

MIN_MINUTES=15
SHOW_ALL=0
DO_KILL=0
DO_PORTS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --min-minutes) MIN_MINUTES=${2:?--min-minutes needs a value}; shift 2 ;;
    --min-minutes=*) MIN_MINUTES="${1#*=}"; shift ;;
    --all) SHOW_ALL=1; shift ;;
    --no-ports) DO_PORTS=0; shift ;;
    --kill) DO_KILL=1; shift ;;
    -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'
  CYN=$'\e[36m'; RST=$'\e[0m'
else
  BOLD=; DIM=; RED=; GRN=; YEL=; CYN=; RST=
fi

MIN_SECONDS=$((MIN_MINUTES * 60))

# ---- helpers ---------------------------------------------------------------
cmd_of()   { [ -r "/proc/$1/cmdline" ] && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/  *$//'; }
fmt_elapsed() {
  local s=$1 h m
  h=$((s/3600)); m=$(((s%3600)/60)); s=$((s%60))
  if [ "$h" -gt 0 ]; then printf '%dh%dm' "$h" "$m"
  elif [ "$m" -gt 0 ]; then printf '%dm%ds' "$m" "$s"
  else printf '%ds' "$s"
  fi
}
# Reasons a process (given its ps fields) looks stale/abandoned. Populates
# the global array `reasons`.
stale_reasons() {
  local ppid=$1 stat=$2 etimes=$3
  reasons=()
  case "$stat" in *Z*) reasons+=("ZOMBIE") ;; esac
  case "$stat" in *T*) reasons+=("STOPPED") ;; esac
  [ "$ppid" = "1" ] && reasons+=("ORPHANED")
  [ "${etimes:-0}" -ge "$MIN_SECONDS" ] && reasons+=("LONG-RUNNING(>${MIN_MINUTES}m)")
}

# ---- one /proc scan: TT device holders + listening-socket owners ----------
declare -A DEV_OF_PID           # pid -> "N N ..." (device numbers)
declare -A SOCK_INODES_OF_PID   # pid -> " inode inode ... " (this pid's own socket fds)
shopt -s nullglob
for fdpath in /proc/[0-9]*/fd/*; do
  tgt=$(readlink "$fdpath" 2>/dev/null) || continue
  pid=${fdpath#/proc/}; pid=${pid%%/*}
  case "$tgt" in
    /dev/tenstorrent/*)
      DEV_OF_PID["$pid"]+="${tgt##*/} "
      ;;
    socket:\[*)
      inode=${tgt#socket:[}; inode=${inode%]}
      SOCK_INODES_OF_PID["$pid"]+="$inode "
      ;;
  esac
done
if command -v fuser >/dev/null 2>&1; then
  for dev in /dev/tenstorrent/*; do
    for pid in $(fuser "$dev" 2>/dev/null); do
      DEV_OF_PID["$pid"]+="${dev##*/} "
    done
  done
fi

# For each pid holding at least one socket fd, check its OWN
# /proc/<pid>/net/tcp{,6} (that process's view of its network namespace --
# safe even if it's in a container netns different from ours) for LISTEN
# (state 0A) sockets whose inode it itself owns.
declare -A PORT_OF_PID   # pid -> "port port ..."
for pid in "${!SOCK_INODES_OF_PID[@]}"; do
  inodes=" ${SOCK_INODES_OF_PID[$pid]}"
  for tcpfile in "/proc/$pid/net/tcp" "/proc/$pid/net/tcp6"; do
    [ -r "$tcpfile" ] || continue
    while read -r _sl local_addr _rem st _txrx _trtm _retr _uid _timeout inode _rest; do
      [ "$st" = "0A" ] || continue
      case "$inodes" in *" $inode "*) : ;; *) continue ;; esac
      hexport=${local_addr##*:}
      port=$((16#$hexport))
      PORT_OF_PID["$pid"]+="$port "
    done < <(tail -n +2 "$tcpfile" 2>/dev/null)
  done
done

# ---- Section 1: stale pytest/python/test processes -------------------------
FLAGGED1=0
LISTED1=0
echo "${BOLD}Stale pytest/python/test process report${RST}  ${DIM}($(date '+%Y-%m-%d %H:%M:%S') — threshold ${MIN_MINUTES}m)${RST}"
echo "${DIM}------------------------------------------------------------${RST}"

while IFS= read -r line; do
  read -r pid ppid user stat etimes comm <<<"$line"
  [ -n "${pid:-}" ] || continue
  args=$(cmd_of "$pid")
  [ -n "$args" ] || args="[$comm]"

  # Match: pytest by name, or any python-family process whose args mention
  # "test" (case-insensitive) -- covers pytest invocations, ad hoc
  # test_*.py / *_test.py scripts, unittest, hang/repro scripts, etc.
  case "$comm" in
    pytest) matched=1 ;;
    python*)
      case "$args" in *[Tt]est*) matched=1 ;; *) matched=0 ;; esac
      ;;
    *) matched=0 ;;
  esac
  [ "$matched" = 1 ] || continue

  stale_reasons "$ppid" "$stat" "$etimes"
  if [ ${#reasons[@]} -eq 0 ]; then
    [ "$SHOW_ALL" -eq 1 ] || continue
  else
    FLAGGED1=$((FLAGGED1+1))
  fi
  LISTED1=$((LISTED1+1))

  el=$(fmt_elapsed "${etimes:-0}")
  devs="${DEV_OF_PID[$pid]:-}"
  ports="${PORT_OF_PID[$pid]:-}"
  if [ ${#reasons[@]} -gt 0 ]; then tag="${RED}${reasons[*]}${RST}"; else tag="${DIM}(unflagged)${RST}"; fi

  printf '%sPID %-8s%s user=%-10s stat=%-5s elapsed=%-8s %s\n' \
    "$BOLD" "$pid" "$RST" "$user" "$stat" "$el" "$tag"
  cmdshow="$args"
  [ ${#cmdshow} -gt 110 ] && cmdshow="${cmdshow:0:110}…"
  printf '    %scmd:%s %s\n' "$DIM" "$RST" "$cmdshow"
  [ -n "$devs" ]  && printf '    %sHOLDS TT DEVICE:%s %s\n' "$YEL" "$RST" "$devs"
  [ -n "$ports" ] && printf '    %sHOLDS PORT(S):%s %s\n' "$YEL" "$RST" "$ports"
done < <(ps -eo pid=,ppid=,user=,stat=,etimes=,comm= 2>/dev/null)

echo "${DIM}------------------------------------------------------------${RST}"
if [ "$FLAGGED1" -eq 0 ]; then
  echo "${GRN}No stale pytest/python/test processes found.${RST}"
else
  echo "${BOLD}${FLAGGED1}${RST} flagged (of ${LISTED1} matching process(es) shown)."
fi

# ---- Section 2: stale TCP port holders (any process) -----------------------
if [ "$DO_PORTS" -eq 1 ]; then
  echo
  FLAGGED2=0
  LISTED2=0
  echo "${BOLD}Stale TCP port-holder report${RST}"
  echo "${DIM}------------------------------------------------------------${RST}"

  while IFS= read -r line; do
    read -r pid ppid user stat etimes comm <<<"$line"
    [ -n "${pid:-}" ] || continue
    ports="${PORT_OF_PID[$pid]:-}"
    [ -n "$ports" ] || continue

    stale_reasons "$ppid" "$stat" "$etimes"
    if [ ${#reasons[@]} -eq 0 ]; then
      [ "$SHOW_ALL" -eq 1 ] || continue
    else
      FLAGGED2=$((FLAGGED2+1))
    fi
    LISTED2=$((LISTED2+1))

    args=$(cmd_of "$pid"); [ -n "$args" ] || args="[$comm]"
    el=$(fmt_elapsed "${etimes:-0}")
    devs="${DEV_OF_PID[$pid]:-}"
    if [ ${#reasons[@]} -gt 0 ]; then tag="${RED}${reasons[*]}${RST}"; else tag="${DIM}(unflagged)${RST}"; fi

    printf '%sPID %-8s%s user=%-10s stat=%-5s elapsed=%-8s ports=%s%-12s%s%s\n' \
      "$BOLD" "$pid" "$RST" "$user" "$stat" "$el" "$CYN" "$ports" "$RST" "  $tag"
    cmdshow="$args"
    [ ${#cmdshow} -gt 110 ] && cmdshow="${cmdshow:0:110}…"
    printf '    %scmd:%s %s\n' "$DIM" "$RST" "$cmdshow"
    [ -n "$devs" ] && printf '    %sHOLDS TT DEVICE:%s %s\n' "$YEL" "$RST" "$devs"
  done < <(ps -eo pid=,ppid=,user=,stat=,etimes=,comm= 2>/dev/null)

  echo "${DIM}------------------------------------------------------------${RST}"
  if [ "$FLAGGED2" -eq 0 ]; then
    echo "${GRN}No stale port-holding processes found.${RST}"
  else
    echo "${BOLD}${FLAGGED2}${RST} flagged (of ${LISTED2} port-holding process(es) shown)."
  fi
fi

# ---- optional interactive kill ---------------------------------------------
if [ "$DO_KILL" -eq 1 ]; then
  echo
  echo "${BOLD}Kill flagged processes${RST} ${DIM}(pytest/python/test matches + port holders)${RST}"
  declare -A OFFERED
  while IFS= read -r line; do
    read -r pid ppid user stat etimes comm <<<"$line"
    [ -n "${pid:-}" ] || continue
    [ -n "${OFFERED[$pid]:-}" ] && continue

    args=$(cmd_of "$pid")
    is_test=0
    case "$comm" in
      pytest) is_test=1 ;;
      python*) case "$args" in *[Tt]est*) is_test=1 ;; esac ;;
    esac
    has_port=0
    [ -n "${PORT_OF_PID[$pid]:-}" ] && has_port=1
    [ "$is_test" = 1 ] || [ "$has_port" = 1 ] || continue

    stale_reasons "$ppid" "$stat" "$etimes"
    [ ${#reasons[@]} -gt 0 ] || continue
    OFFERED["$pid"]=1

    read -r -p "Kill PID $pid ($comm, stat=$stat, reasons=${reasons[*]})? [y/N] " ans
    case "$ans" in
      y|Y) kill -9 "$pid" 2>&1 && echo "  killed $pid" ;;
      *) echo "  skipped $pid" ;;
    esac
  done < <(ps -eo pid=,ppid=,user=,stat=,etimes=,comm= 2>/dev/null)
fi
