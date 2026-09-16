#!/usr/bin/env bash
# tt-devs — report Tenstorrent device usage, including other users' jobs.
#
# Shows, for each /dev/tenstorrent/<N>: whether it is claimed, which PID(s) hold
# it, the owning user + command where visible, and (best-effort) the TCP port a
# model server in that process group is listening on.
#
# Suggested alias:
#     alias ttdev='~/scripts/tt-devs.sh'
#
# WHY THIS IS MORE THAN A `fuser` WRAPPER
#   Since 2026-09-15 these boxes mount /proc with hidepid=2, so /proc/<pid> for
#   other users is hidden outright: ps, fuser and a /proc/*/fd scan see ONLY
#   your own processes. An empty `fuser /dev/tenstorrent/N` therefore no longer
#   means "free" — it means "not held by me". This script detects that mode and
#   falls back to sources that still work across users.
#
# SOURCES (each holder line is tagged with the source that found it)
#   umd    /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_<N>_* — UMD's per-chip robust mutex.
#          World-readable (0666) and its trailing struct fields carry the holder:
#              struct pthread_mutex_wrapper {   // 56 bytes, x86_64
#                  pthread_mutex_t mutex;       //  0..39
#                  uint64_t        initialized; // 40..47  ("TTUMDMTX")
#                  pid_t           owner_tid;   // 48..51
#                  pid_t           owner_pid;   // 52..55
#              };
#          (see tt-metal .../umd/device/api/umd/device/utils/robust_mutex.hpp)
#          CHIP_IN_USE is taken in LocalChip::start_device and released in
#          close_device, so it spans the whole device session. THIS IS THE ONLY
#          SOURCE THAT SEES OTHER USERS. owner_pid is 0 after a clean release.
#   fd     scan of /proc/<pid>/fd symlinks pointing at /dev/tenstorrent/*.
#   fuser  same information, same blindness. Both are own-process-only under
#          hidepid, and both report a device as open even when UMD has not
#          started the chip (UMD opens every device node in the cluster, so a
#          1-chip job still shows 32 open fds — umd is the accurate signal for
#          "is this chip actually claimed").
#   clk    /sys/class/tenstorrent/tenstorrent!<N>/tt_aiclk. Pure sysfs, no /proc,
#          so it works across users, but carries no PID. An unused chip sits at
#          800MHz; UMD raises it (~1350) during device bring-up and it stays
#          there until close, so it tracks "a chip is OPEN", not "a chip is
#          computing" - verified by holding a mesh open and idle for 60s with
#          the clock pinned at 1350 the whole time. It lags the claim only
#          during bring-up, where chips ramp one at a time. Labelled "active"
#          because it cannot name a holder, but it is a claim-grade signal and
#          it is what contradicts a dead-looking owner pid below.
#
# HIDDEN vs STALE HOLDERS
#   For a PID we cannot see, `kill -0` still distinguishes the two cases that
#   matter, because signal permission checks are not subject to hidepid:
#       EPERM ("Operation not permitted") -> alive, owned by another user
#       ESRCH ("No such process")         -> dead; the lock is STALE
#   A stale CHIP_IN_USE means the owner crashed without unlocking. UMD recovers
#   it via EOWNERDEAD on the next acquire, so the chip is usable — it is not a
#   holder you need to chase.
#
# KNOWN BLIND SPOTS (all verified, not guesses)
#   containers   A job in a container writes its PID-NAMESPACED pid into the
#                lock. With --ipc=host the lock is shared but that pid means
#                nothing here; without it the container gets a private /dev/shm
#                and its locks are invisible entirely, so the chip reads "free"
#                while held. tt-media-server runs containerized, so this is not
#                hypothetical. Partly mitigated: a dead-looking owner pid on a
#                chip whose aiclk is up is reported as "holder unresolvable"
#                rather than STALE. A private-/dev/shm container remains
#                invisible - there is no host-side signal for it besides aiclk.
#   JTAG         LocalChip::start_device returns before taking the lock for
#                IODeviceType::JTAG, so a JTAG-attached chip never appears.
#   remote chips RemoteChip::start_device is empty - only MMIO-local chips take
#                CHIP_IN_USE. Moot on a BH Galaxy (32 nodes = 32 local chips),
#                but on Wormhole a remote chip behind an MMIO chip has no lock
#                of its own. It also has no /dev/tenstorrent node, so it is out
#                of this script's scope either way.
#   pre-start    A process that opened the device node but has not yet called
#                start_device holds no lock. The fd/fuser sources cover that,
#                but only for your own processes.
#   PID reuse    A recycled pid makes a genuinely stale lock look live. Errs
#                toward a false "IN USE", which is the safe direction.
#
# Flags
#   --no-ports   skip the port/model lookup (faster)
#   --blind      ignore own-process sources (fd/fuser/ps), exercising only the
#                cross-user path. Use this to test the hidepid code path, since
#                you cannot become another user to check it for real.
#   --idle-clk N treat aiclk > N as active (default 900)
#   -h|--help    this help
set -uo pipefail

NO_PORTS=0
BLIND=0
IDLE_CLK=${TT_DEVS_IDLE_CLK:-900}
SHM_DIR=${TT_DEVS_SHM_DIR:-/dev/shm}   # override is for testing only

usage() { sed -n '2,/^set -uo/p' "$0" | sed -e 's/^# \{0,1\}//' -e '/^set -uo/d'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --no-ports)  NO_PORTS=1 ;;
    --blind)     BLIND=1 ;;
    --idle-clk)  shift; IDLE_CLK=${1:?--idle-clk needs a value} ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'
  CYN=$'\e[36m'; MAG=$'\e[35m'; RST=$'\e[0m'
else
  BOLD=; DIM=; RED=; GRN=; YEL=; CYN=; MAG=; RST=
fi

shopt -s nullglob

# ---- detect /proc visibility ----------------------------------------------
# Two independent checks: the mount option, and an empirical probe (PID 1 is
# root-owned, so if we cannot read its stat we cannot read any other user's).
HIDEPID=$(awk '$2=="/proc"{n=split($4,o,","); for(i=1;i<=n;i++) if(o[i]~/^hidepid=/){sub(/^hidepid=/,"",o[i]); print o[i]}}' /proc/mounts 2>/dev/null | head -1)
if [ -r /proc/1/stat ]; then PROC_BLIND=0; else PROC_BLIND=1; fi
RESTRICTED=0
case "$HIDEPID" in
  1|2|invisible|noaccess) RESTRICTED=1 ;;
esac
[ "$PROC_BLIND" -eq 1 ] && RESTRICTED=1
[ "$BLIND" -eq 1 ] && RESTRICTED=1

# ---- enumerate device nodes (numeric basenames only) ----------------------
DEVICES=()
while read -r d; do
  [ -n "$d" ] && DEVICES+=("$d")
done < <(for d in /dev/tenstorrent/*; do
           b=${d##*/}
           [[ $b =~ ^[0-9]+$ ]] && printf '%s\n' "$d"
         done | sort -t/ -k4 -n)
if [ ${#DEVICES[@]} -eq 0 ]; then
  echo "${RED}No /dev/tenstorrent/* devices found.${RST}" >&2
  exit 1
fi

# ---- source: UMD CHIP_IN_USE robust mutexes (cross-user) -------------------
declare -A UMD_PID UMD_TID UMD_CREATOR
for f in "$SHM_DIR"/TT_UMD_LOCK.CHIP_IN_USE_*; do
  idx=${f##*CHIP_IN_USE_}; idx=${idx%%_*}
  [[ $idx =~ ^[0-9]+$ ]] || continue
  [ "$(stat -c %s "$f" 2>/dev/null)" = 56 ] || continue   # unexpected layout; skip
  pid=$(od -An -tu4 -j52 -N4 "$f" 2>/dev/null | tr -d ' ')
  tid=$(od -An -tu4 -j48 -N4 "$f" 2>/dev/null | tr -d ' ')
  [ -n "$pid" ] && [ "$pid" != 0 ] || continue
  UMD_PID["/dev/tenstorrent/$idx"]=$pid
  UMD_TID["/dev/tenstorrent/$idx"]=$tid
  UMD_CREATOR["/dev/tenstorrent/$idx"]=$(stat -c %U "$f" 2>/dev/null)
done

# ---- source: own-process fd scan + fuser (blind to other users) -----------
declare -A DEV_PIDS
if [ "$BLIND" -eq 0 ]; then
  for fdpath in /proc/[0-9]*/fd/*; do
    tgt=$(readlink "$fdpath" 2>/dev/null) || continue
    case "$tgt" in
      /dev/tenstorrent/*)
        pid=${fdpath#/proc/}; pid=${pid%%/*}
        DEV_PIDS["$tgt"]+="$pid "
        ;;
    esac
  done
  if command -v fuser >/dev/null 2>&1; then
    for dev in "${DEVICES[@]}"; do
      fp=$(fuser "$dev" 2>/dev/null) || true
      [ -n "$fp" ] && DEV_PIDS["$dev"]+="$fp "
    done
  fi
fi

# ---- source: sysfs aiclk (cross-user, no PID) -----------------------------
declare -A DEV_CLK
for dev in "${DEVICES[@]}"; do
  n=${dev##*/}
  v=$(cat "/sys/class/tenstorrent/tenstorrent!$n/tt_aiclk" 2>/dev/null)
  [[ $v =~ ^[0-9]+$ ]] && DEV_CLK["$dev"]=$v
done

# ---- helpers ---------------------------------------------------------------
uniq_pids() { tr ' ' '\n' <<<"$1" | grep -E '^[0-9]+$' | sort -un; }
comm_of()   { [ "$BLIND" -eq 1 ] && return; ps -o comm= -p "$1" 2>/dev/null | head -1; }
user_of()   { [ "$BLIND" -eq 1 ] && return; ps -o user= -p "$1" 2>/dev/null | head -1; }
pgid_of()   { [ "$BLIND" -eq 1 ] && return; ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '; }
cmd_of()    { [ "$BLIND" -eq 1 ] && return; tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/  *$//'; }

# Liveness of a PID we may not be able to see. LC_ALL=C keeps the errno strings
# stable so the EPERM/ESRCH distinction stays parseable.
# Prints: mine | hidden | dead
pid_state() {
  local pid=$1 err
  # --blind reports our own live PIDs as hidden, so the output matches what
  # another user's job would actually render as. Dead stays dead.
  err=$(LC_ALL=C kill -0 "$pid" 2>&1) && {
    [ "$BLIND" -eq 1 ] && { echo hidden; return; }
    echo mine; return
  }
  case "$err" in
    *"not permitted"*)  echo hidden ;;
    *"o such process"*) echo dead ;;
    *)                  echo hidden ;;
  esac
}

# all PIDs sharing a process group id
pids_in_pgid() {
  [ "$BLIND" -eq 1 ] && return
  ps -e -o pid=,pgid= 2>/dev/null | awk -v g="$1" '$2==g{print $1}'
}

# best-effort model name from a PID's environ
model_for_pids() {
  local p v
  for p in "$@"; do
    [ -r "/proc/$p/environ" ] || continue
    v=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null \
        | grep -E '^(MODEL|MODEL_NAME)=' | head -1 | cut -d= -f2-)
    [ -n "$v" ] && { echo "$v"; return; }
  done
}

# Given a holder PID, print serving lines: "<port> <pid> <comm>".
# The serving port is the --port of the server front-end (uvicorn main:app /
# vllm api server) in the holder's process group — NOT the many internal
# gloo/RPC/metrics LISTEN sockets, which we deliberately ignore.
serving_lines() {
  local holder=$1 pgid grp p cl port comm
  pgid=$(pgid_of "$holder"); [ -n "$pgid" ] || return
  grp=$(pids_in_pgid "$pgid"); [ -n "$grp" ] || grp=$holder
  for p in $grp; do
    cl=$(cmd_of "$p"); [ -n "$cl" ] || continue
    case "$cl" in
      *uvicorn*|*main:app*|*api_server*|*"vllm serve"*) : ;;
      *) continue ;;
    esac
    comm=$(comm_of "$p")
    while read -r port; do
      [ -n "$port" ] && echo "$port $p $comm"
    done < <(grep -oE -- '--port[= ]+[0-9]+' <<<"$cl" | grep -oE '[0-9]+$')
  done | sort -n | awk '!seen[$1]++'
}

# One holder line (+ cmd line when we can see it).
print_holder() {
  local pid=$1 src=$2 state comm usr cmd
  state=$(pid_state "$pid")
  comm=$(comm_of "$pid"); usr=$(user_of "$pid")
  if [ -z "$usr" ]; then
    case "$state" in
      hidden) usr="? ${DIM}(hidden)${RST}"; comm=${comm:-"? (hidden)"} ;;
      dead)   usr="-";                      comm=${comm:-"(exited - stale lock)"} ;;
      *)      usr="?";                      comm=${comm:-"?"} ;;
    esac
  fi
  printf '    %sPID %-8s%s %s%-18s%s user=%s %s[%s]%s\n' \
    "$BOLD" "$pid" "$RST" "$CYN" "$comm" "$RST" "$usr" "$DIM" "$src" "$RST"
  cmd=$(cmd_of "$pid")
  if [ -n "$cmd" ]; then
    [ ${#cmd} -gt 96 ] && cmd="${cmd:0:96}..."
    printf '        %scmd:%s %s\n' "$DIM" "$RST" "$cmd"
  fi
}

# ---- report ----------------------------------------------------------------
host=$(hostname 2>/dev/null || echo "?")
claimed=0 opened=0
for dev in "${DEVICES[@]}"; do
  [ -n "${UMD_PID[$dev]:-}" ] && claimed=$((claimed+1))
  [ -n "${DEV_PIDS[$dev]:-}" ] && opened=$((opened+1))
done

printf '%sTenstorrent device usage%s  %s(%s - %d/%d chips claimed)%s\n' \
  "$BOLD" "$RST" "$DIM" "$host" "$claimed" "${#DEVICES[@]}" "$RST"

if [ "$RESTRICTED" -eq 1 ]; then
  why="hidepid=${HIDEPID:-?}"
  [ "$BLIND" -eq 1 ] && why="--blind"
  printf '%s! restricted /proc (%s): ps/fuser see only your own processes.%s\n' \
    "$YEL" "$why" "$RST"
  printf '%s  Cross-user holders come from UMD CHIP_IN_USE locks; other users%s\n' "$DIM" "$RST"
  printf '%s  appear as user=? (hidden) with no command line.%s\n' "$DIM" "$RST"
else
  printf '%s  full /proc visibility: all processes are visible.%s\n' "$DIM" "$RST"
fi
printf '%s%s%s\n' "$DIM" "------------------------------------------------------------" "$RST"

for dev in "${DEVICES[@]}"; do
  upid=${UMD_PID[$dev]:-}
  pids=$(uniq_pids "${DEV_PIDS[$dev]:-}")
  clk=${DEV_CLK[$dev]:-}
  active=0
  if [ -n "$clk" ] && [ "$clk" -gt "$IDLE_CLK" ] 2>/dev/null; then active=1; fi

  if [ -z "$upid" ] && [ -z "$pids" ]; then
    if [ "$active" -eq 1 ]; then
      printf '%-22s %sactive%s %s(aiclk %s, no lock holder - starting up or between claims)%s\n' \
        "$dev" "$MAG" "$RST" "$DIM" "$clk" "$RST"
    else
      printf '%-22s %sfree%s\n' "$dev" "$GRN" "$RST"
    fi
    continue
  fi

  # A stale UMD lock with no other holder means the chip is recoverable.
  ustate=
  [ -n "$upid" ] && ustate=$(pid_state "$upid")
  if [ "$ustate" = dead ] && [ -z "$pids" ]; then
    if [ "$active" -eq 1 ]; then
      # The owner PID does not exist in our namespace, yet the chip is clocked
      # up: the holder is almost certainly alive in a PID namespace we cannot
      # see (a containerized job writes its namespaced PID into the shared
      # lock). Calling this STALE would wrongly imply the chip is available.
      printf '%-22s %sIN USE%s %s(holder unresolvable: lock pid %s not in our%s\n' \
        "$dev" "$RED" "$RST" "$DIM" "$upid" "$RST"
      printf '                       %snamespace but aiclk %s - likely a container)%s\n' \
        "$DIM" "$clk" "$RST"
    else
      printf '%-22s %sSTALE%s %s(lock owner pid %s is dead; UMD recovers on next open)%s\n' \
        "$dev" "$YEL" "$RST" "$DIM" "$upid" "$RST"
    fi
    continue
  fi

  note=
  [ "$active" -eq 1 ] && note=" ${DIM}aiclk ${clk}${RST}"
  printf '%-22s %sIN USE%s%s\n' "$dev" "$RED" "$RST" "$note"

  if [ -n "$upid" ]; then
    print_holder "$upid" "umd CHIP_IN_USE"
    if [ "$ustate" = hidden ]; then
      creator=${UMD_CREATOR[$dev]:-}
      # The shm file's owner is whoever first created the lock, not necessarily
      # the current holder - a weak hint for who to ask, nothing more.
      if [ -n "$creator" ] && [ "$creator" != "$(id -un)" ]; then
        printf '        %shint:%s lock file created by %s %s(may be a previous job)%s\n' \
          "$DIM" "$RST" "$creator" "$DIM" "$RST"
      fi
    fi
  fi

  while read -r pid; do
    [ -n "$pid" ] || continue
    [ "$pid" = "$upid" ] && continue    # already printed from the umd source
    print_holder "$pid" "fd/fuser"
    if [ "$NO_PORTS" -eq 0 ]; then
      mapfile -t svc < <(serving_lines "$pid")
      if [ "${#svc[@]}" -gt 0 ]; then
        model=$(model_for_pids $(pids_in_pgid "$(pgid_of "$pid")"))
        for s in "${svc[@]}"; do
          read -r port lpid lcomm <<<"$s"
          printf '        %sserving:%s port %s%s%s' \
            "$YEL" "$RST" "$BOLD" "$port" "$RST"
          printf ' (%s, pid %s)' "${lcomm:-?}" "$lpid"
          [ -n "${model:-}" ] && printf ' model=%s' "$model"
          printf '\n'
        done
      fi
    fi
  done <<<"$pids"
done

if [ "$RESTRICTED" -eq 1 ] && [ "$claimed" -eq 0 ] && [ "$opened" -eq 0 ]; then
  printf '%s%s%s\n' "$DIM" "------------------------------------------------------------" "$RST"
  printf '%sNo holders found. Under restricted /proc this is trustworthy only%s\n' "$DIM" "$RST"
  printf '%sbecause the UMD lock source is cross-user - but a job that opened a%s\n' "$DIM" "$RST"
  printf '%schip without starting it (or a non-UMD user of the device) is invisible.%s\n' "$DIM" "$RST"
fi
