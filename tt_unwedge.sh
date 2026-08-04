#!/usr/bin/env bash
# tt_unwedge.sh -- detect and recover Tenstorrent boards left wedged by a
# device hang (dispatch/fetch-queue timeout, killed EngineCore, crashed
# pytest), WITHOUT rebooting the host.
#
# The symptom this exists for:
#
#   $ tt-smi
#   Error in detecting devices!
#   Read 0xffffffff over PCIe ID 1: the board should be reset.
#   ...
#   Exiting...
#
#   $ tt-smi -r        # <-- ALSO FAILS, with the exact same error
#
# THE KEY INSIGHT (this is why the obvious fix doesn't work):
#
#   `tt-smi -r` with no targets runs full topology discovery across ALL
#   devices BEFORE it issues any reset. Discovery reads the wedged board,
#   gets 0xffffffff, and aborts -- so it never reaches the reset it was
#   invoked to perform. Chicken-and-egg: the tool that fixes the board
#   refuses to start because the board is broken.
#
#   The fix is to NAME the target, which skips the broad discovery, plus
#   --no_reinit to skip the post-reset re-detection that trips on the same
#   path:
#
#       tt-smi -r <logical_id> --no_reinit
#
#   That single command has resolved every wedge seen so far. Everything
#   else in this script is preflight, verification, and escalation for when
#   it doesn't.
#
# ALSO WORTH KNOWING: `0xffffffff` reads look like a dead card but usually
# aren't. Check `/sys/bus/pci/devices/<BDF>/vendor` -- if it still reads
# 0x1e52, the board is enumerated and on the bus, and only the ASIC is
# wedged. That is recoverable in seconds. Do not reboot on 0xffffffff
# alone; reboot only after the escalation ladder below is exhausted.
#
# Usage:
#   tt_unwedge.sh                 # detect + fix + verify (the normal case)
#   tt_unwedge.sh --check         # report health only, change nothing
#   tt_unwedge.sh --kill          # also kill processes holding the devices
#   tt_unwedge.sh --device 1      # only touch this logical id
#   tt_unwedge.sh --dry-run       # print the ladder, execute nothing
#
# Flags:
#   --check         probe only. exit 0 = healthy, 1 = wedged. Good as a
#                    precondition before a long test run.
#   --kill          kill anything holding /dev/tenstorrent/* first. A reset
#                    silently fails while fds are open, so a stale
#                    EngineCore will make this script look ineffective.
#   --device N      restrict to one UMD logical id (repeatable).
#   --dry-run       show what would run without running it.
#   --no-escalate   stop after the tt-smi step; skip sudo FLR / modprobe.
#   -h|--help       this help
#
# Needs: tt-smi on PATH (or ~/.local/bin). Escalation steps 3 and 4 need
# passwordless `sudo -n`; without it they are skipped with a printed
# command you can run by hand.
#
# Exit codes: 0 recovered/healthy, 1 still wedged (see printed next steps),
# 2 usage/environment error.

set -o pipefail

TT_SMI="${TT_SMI:-$(command -v tt-smi || echo "$HOME/.local/bin/tt-smi")}"
TT_VENDOR_ID="0x1e52"
PROBE_TIMEOUT=180
RESET_TIMEOUT=180

MODE=fix
DO_KILL=0
DRY_RUN=0
ESCALATE=1
TARGETS=()

die()  { echo "ERROR: $*" >&2; exit 2; }
info() { echo "[tt-unwedge] $*"; }
run()  {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] $*"
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)       MODE=check; shift ;;
        --kill)        DO_KILL=1; shift ;;
        --device)      [[ -n "${2:-}" ]] || die "--device needs an id"
                       TARGETS+=("$2"); shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --no-escalate) ESCALATE=0; shift ;;
        -h|--help)     sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)             die "unknown flag: $1 (try --help)" ;;
    esac
done

[[ -x "$TT_SMI" ]] || die "tt-smi not found (looked at: $TT_SMI). Set TT_SMI=/path/to/tt-smi"

# ---------------------------------------------------------------- helpers

# Map UMD logical id -> PCI BDF. tt-smi -ls prints this, but it can't run on
# a wedged board -- which is exactly when we need the map. So derive it from
# sysfs: Tenstorrent functions sorted by BDF, index == logical id. That has
# matched tt-smi's own numbering on every box seen (0->01:00.0, 1->02:00.0,
# ...). Only used for the sudo FLR fallback, so a mismatch costs a wasted
# reset of a healthy neighbour, not data.
bdf_for_id() {
    local want="$1" i=0 dev vendor
    for dev in $(ls -1 /sys/bus/pci/devices 2>/dev/null | sort); do
        vendor=$(cat "/sys/bus/pci/devices/$dev/vendor" 2>/dev/null)
        [[ "$vendor" == "$TT_VENDOR_ID" ]] || continue
        if [[ "$i" == "$want" ]]; then echo "$dev"; return 0; fi
        i=$((i + 1))
    done
    return 1
}

holders() {
    local d pids all=""
    for d in /dev/tenstorrent/*; do
        [[ -e "$d" ]] || continue
        pids=$(fuser "$d" 2>/dev/null)
        [[ -n "$pids" ]] && all="$all $pids"
    done
    echo "$all" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' '
}

# Probe health. Sets PROBE_OUT and WEDGED_IDS. Returns 0 healthy, 1 wedged.
PROBE_OUT=""
WEDGED_IDS=()
probe() {
    PROBE_OUT=$(timeout "$PROBE_TIMEOUT" "$TT_SMI" -ls 2>&1)
    WEDGED_IDS=()
    if grep -qE "Error in detecting devices|should be reset|0xffffffff" <<<"$PROBE_OUT"; then
        # "Read 0xffffffff over PCIe ID 1: the board should be reset."
        mapfile -t WEDGED_IDS < <(grep -oP 'over PCIe ID \K[0-9]+' <<<"$PROBE_OUT" | sort -u)
        return 1
    fi
    grep -q "All available boards" <<<"$PROBE_OUT" || return 1
    return 0
}

have_sudo() { sudo -n true 2>/dev/null; }

# ---------------------------------------------------------------- preflight

info "tt-smi: $TT_SMI"

HOLD=$(holders)
if [[ -n "${HOLD// /}" ]]; then
    info "processes holding /dev/tenstorrent/*: $HOLD"
    ps -o pid=,comm= -p ${HOLD} 2>/dev/null | sed 's/^/           /'
    if [[ $DO_KILL -eq 1 ]]; then
        info "killing them (--kill)"
        run kill -9 ${HOLD} 2>/dev/null
        sleep 5
        HOLD=$(holders)
        [[ -n "${HOLD// /}" ]] && info "WARNING: still held by: $HOLD"
    else
        info "NOTE: a reset will not take while fds are open. Re-run with --kill,"
        info "      or: pkill -9 -f 'VLLM::EngineCore'; pkill -9 -f pytest"
    fi
else
    info "no processes hold the devices"
fi

# ---------------------------------------------------------------- probe

if probe; then
    info "all boards healthy:"
    grep -E '^\│ [0-9]' <<<"$PROBE_OUT" | head -20 | sed 's/^/           /'
    exit 0
fi

info "WEDGED. tt-smi reports:"
grep -E "Error in detecting|should be reset|0xffffffff" <<<"$PROBE_OUT" | sed 's/^/           /'

if [[ $MODE == check ]]; then
    info "--check: not touching anything. Re-run without --check to recover."
    exit 1
fi

# Which ids to reset: parsed from the error, else --device, else all present.
if [[ ${#TARGETS[@]} -gt 0 ]]; then
    IDS=("${TARGETS[@]}")
    info "targets from --device: ${IDS[*]}"
elif [[ ${#WEDGED_IDS[@]} -gt 0 ]]; then
    IDS=("${WEDGED_IDS[@]}")
    info "targets parsed from tt-smi error: ${IDS[*]}"
else
    mapfile -t IDS < <(ls -1 /dev/tenstorrent/ 2>/dev/null | sort -n)
    info "could not parse a specific id; trying all present: ${IDS[*]}"
fi

# ------------------------------------------------- ladder 1: targeted reset
# The one that actually works. Named target + --no_reinit sidesteps the
# discovery-before-reset abort described in the header.

info "--- step 1: targeted tt-smi reset ---"
for id in "${IDS[@]}"; do
    info "tt-smi -r $id --no_reinit"
    run timeout "$RESET_TIMEOUT" "$TT_SMI" -r "$id" --no_reinit 2>&1 | sed 's/^/           /'
done

if [[ $DRY_RUN -eq 0 ]] && probe; then
    info "RECOVERED after targeted reset."
    exit 0
fi
[[ $DRY_RUN -eq 1 ]] || info "still wedged after step 1"

if [[ $ESCALATE -eq 0 ]]; then
    info "--no-escalate set; stopping here."
    exit 1
fi

# ------------------------------------------------------- ladder 2: reset all
info "--- step 2: tt-smi reset all (--no_reinit) ---"
run timeout "$RESET_TIMEOUT" "$TT_SMI" -r all --no_reinit 2>&1 | sed 's/^/           /'
if [[ $DRY_RUN -eq 0 ]] && probe; then
    info "RECOVERED after reset-all."
    exit 0
fi

# --------------------------------------------------- ladder 3: PCIe FLR
info "--- step 3: PCIe function-level reset via sysfs ---"
for id in "${IDS[@]}"; do
    bdf=$(bdf_for_id "$id") || { info "no BDF for id $id, skipping"; continue; }
    vendor=$(cat "/sys/bus/pci/devices/$bdf/vendor" 2>/dev/null)
    info "id $id -> $bdf (vendor $vendor)"
    if [[ ! -e "/sys/bus/pci/devices/$bdf/reset" ]]; then
        info "  no reset node; skipping"
        continue
    fi
    if have_sudo; then
        run sudo sh -c "echo 1 > /sys/bus/pci/devices/$bdf/reset"
    else
        info "  no passwordless sudo. Run by hand:"
        info "    sudo sh -c 'echo 1 > /sys/bus/pci/devices/$bdf/reset'"
    fi
done
if [[ $DRY_RUN -eq 0 ]] && probe; then
    info "RECOVERED after PCIe FLR."
    exit 0
fi

# ----------------------------------------------- ladder 4: driver reload
info "--- step 4: driver reload ---"
HOLD=$(holders)
if [[ -n "${HOLD// /}" ]]; then
    info "cannot unload: devices still held by $HOLD (kill them, or pass --kill)"
elif have_sudo; then
    run sudo modprobe -r tenstorrent && run sudo modprobe tenstorrent
    sleep 3
    if [[ $DRY_RUN -eq 0 ]] && probe; then
        info "RECOVERED after driver reload."
        exit 0
    fi
else
    info "no passwordless sudo. Run by hand:"
    info "  sudo modprobe -r tenstorrent && sudo modprobe tenstorrent"
fi

# ---------------------------------------------------------------- give up
info "STILL WEDGED after the full ladder. Now a reboot is justified."
info "Before rebooting, capture state for a bug report:"
info "  tt-smi -ls 2>&1 | tail -30"
info "  dmesg | grep -iE 'tenstorrent|pcie|aer' | tail -40"
info "  for b in \$(ls /sys/bus/pci/devices); do \\"
info "    [ \"\$(cat /sys/bus/pci/devices/\$b/vendor 2>/dev/null)\" = $TT_VENDOR_ID ] \\"
info "      && echo \"\$b \$(cat /sys/bus/pci/devices/\$b/current_link_speed 2>/dev/null)\"; done"
info "A board whose sysfs 'vendor' no longer reads $TT_VENDOR_ID has actually"
info "dropped off the bus -- that one needs the reboot (or a cold power cycle)."
exit 1
