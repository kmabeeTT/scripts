#!/usr/bin/env bash
# tt-devs — report Tenstorrent device usage.
#
# Shows, for each /dev/tenstorrent/<N>: whether it is in use, which PID(s) hold
# it open, the owning user + command, and (best-effort) the TCP port a model
# server in that process group is listening on.
#
# Suggested alias:
#     alias ttdev='~/scripts/tt-devs.sh'
#
# Notes / how it works:
#  - Holders are found via `fuser` + a scan of /proc/<pid>/fd symlinks pointing
#    at /dev/tenstorrent/*. As a normal user you only see your own processes
#    (and any visible via ps); run with sudo to see other users' holders.
#  - Port detection is netns-aware: tt-media-server runs inside a container, so
#    host `ss` sees the wrong network namespace. Instead we read the holder's
#    own /proc/<pid>/net/tcp{,6} LISTEN table and keep only sockets whose inode
#    is owned by a PID sharing the holder's process group (pgid) — i.e. the
#    uvicorn front-end that launched the engine. This is best-effort: a
#    standalone job (no server) correctly shows no port.
#
# Flags:
#   --no-ports   skip the port/model lookup (faster)
#   -h|--help    this help
set -uo pipefail

NO_PORTS=0
for a in "$@"; do
  case "$a" in
    --no-ports) NO_PORTS=1 ;;
    -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $a (try --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'
  CYN=$'\e[36m'; RST=$'\e[0m'
else
  BOLD=; DIM=; RED=; GRN=; YEL=; CYN=; RST=
fi

shopt -s nullglob

# ---- enumerate device nodes (numeric basenames only) ----------------------
DEVICES=()
for d in /dev/tenstorrent/*; do
  b=${d##*/}
  [[ $b =~ ^[0-9]+$ ]] && DEVICES+=("$d")
done
if [ ${#DEVICES[@]} -eq 0 ]; then
  echo "${RED}No /dev/tenstorrent/* devices found.${RST}" >&2
  exit 1
fi

# ---- build device -> holder PIDs map (one /proc scan + fuser) --------------
declare -A DEV_PIDS
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

# ---- helpers ---------------------------------------------------------------
uniq_pids() { tr ' ' '\n' <<<"$1" | grep -E '^[0-9]+$' | sort -un; }
comm_of()   { ps -o comm= -p "$1" 2>/dev/null | head -1; }
user_of()   { ps -o user= -p "$1" 2>/dev/null | head -1; }
pgid_of()   { ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '; }
cmd_of()    { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/  *$//'; }

# all PIDs sharing a process group id
pids_in_pgid() {
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

# ---- report ----------------------------------------------------------------
host=$(hostname 2>/dev/null || echo "?")
now=$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
in_use=0
for dev in "${DEVICES[@]}"; do
  [ -n "${DEV_PIDS[$dev]:-}" ] && in_use=$((in_use+1))
done

printf '%sTenstorrent device usage%s  %s(%s — %d/%d in use)%s\n' \
  "$BOLD" "$RST" "$DIM" "$host" "$in_use" "${#DEVICES[@]}" "$RST"
printf '%s%s%s\n' "$DIM" "------------------------------------------------------------" "$RST"

for dev in "${DEVICES[@]}"; do
  pids=$(uniq_pids "${DEV_PIDS[$dev]:-}")
  if [ -z "$pids" ]; then
    printf '%-22s %sfree%s\n' "$dev" "$GRN" "$RST"
    continue
  fi
  printf '%-22s %sIN USE%s\n' "$dev" "$RED" "$RST"
  while read -r pid; do
    [ -n "$pid" ] || continue
    comm=$(comm_of "$pid"); [ -n "$comm" ] || comm="(gone/no-access)"
    usr=$(user_of "$pid");  [ -n "$usr" ]  || usr="?"
    printf '    %sPID %-8s%s %s%-18s%s user=%s\n' \
      "$BOLD" "$pid" "$RST" "$CYN" "$comm" "$RST" "$usr"
    cmd=$(cmd_of "$pid")
    if [ -n "$cmd" ]; then
      [ ${#cmd} -gt 96 ] && cmd="${cmd:0:96}…"
      printf '        %scmd:%s %s\n' "$DIM" "$RST" "$cmd"
    fi
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
