#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Disaggregated prefill -> decode demo, Mistral Small 4 119B, one Blackhole galaxy.
#
#   turn 1 : tt-metal prefills your prompt on a 32-chip 8x4 mesh, writes the KV cache to
#            disk, exits; tt-blaze seeds those rows into its reload ring and answers.
#   turn 2+: decode only, in the same cache. ~2.4 ms/token.
#
# Both legs want all 32 chips, so they run in sequence. Turn 1 therefore costs a prefill
# (~9 min) plus a ring capture (~9 min); everything after it is instant. That is the demo:
# the handoff, not the startup.
#
#   ./disagg_demo.sh
set -uo pipefail

TT_METAL=${TT_METAL:-/data/kmabee/tt-metal}
BLAZE=${BLAZE:-/data/kmabee/tt-blaze}
WEIGHTS=${WEIGHTS:-/data/kmabee/models/Mistral-Small-4-119B-2603}
PREFILL_CACHE=${PREFILL_CACHE:-/data/kmabee/mistral4_caches/ttnn_cache_8x4}
LOGDIR=${LOGDIR:-$HOME/disagg_demo_logs}
STAMP=$(date +%Y%m%d_%H%M%S)
DUMP=${DUMP:-/data/kmabee/disagg_kv/demo_$STAMP}
mkdir -p "$LOGDIR" "$DUMP"

export TT_METAL BLAZE WEIGHTS PREFILL_CACHE LOGDIR DUMP STAMP
export M4_DISAGG_FIFO=/tmp/m4_disagg_demo.in
export M4_DISAGG_OUT=/tmp/m4_disagg_demo.out

# The python below is extracted to a temp file rather than fed to `python3 -` on stdin:
# a heredoc IS stdin, so input() would read the exhausted script and EOF instantly.
PYSRC=$(mktemp /tmp/disagg_demo.XXXXXX.py)
trap 'rm -f "$PYSRC"' EXIT
cat > "$PYSRC" <<'PY'
import json, os, re, signal, subprocess, sys, time
from pathlib import Path

TT_METAL, BLAZE = os.environ["TT_METAL"], os.environ["BLAZE"]
DUMP, LOGDIR, STAMP = os.environ["DUMP"], os.environ["LOGDIR"], os.environ["STAMP"]
FIFO, OUT = os.environ["M4_DISAGG_FIFO"], os.environ["M4_DISAGG_OUT"]
C = dict(dim="\033[2m", b="\033[1m", g="\033[92m", y="\033[93m", r="\033[91m", off="\033[0m")


def note(msg, c="dim"):
    print(f"{C[c]}{msg}{C['off']}", flush=True)


#: pytest section headers, which appear ONLY on a real failure. Deliberately not
#: "short test summary": pytest prints that header on a PASSING run too (with PASSED under
#: it), so using it as a failure marker fails every successful run.
FAIL_MARKS = ("= FAILURES =", "= ERRORS =", "Traceback (most recent call last)")


def wait_for(log, done, label, proc=None):
    """Tail `log` until it succeeds or fails. Returns True on success.

    Success is checked FIRST, and the process exit code is the final authority: the done
    marker is written before pytest's summary, so any ordering that consults failure text
    first can trip over a run that already succeeded.

    Prints the elapsed clock rather than a spinner: these phases run for minutes, and the
    question a watcher actually has is "how long has this been going", not "is it alive".
    """
    t0, last = time.time(), ""
    while True:
        try:
            txt = Path(log).read_text(errors="ignore")
        except OSError:
            txt = ""
        if any(m in txt for m in done):
            print(f"\r  {label} done in {time.time()-t0:5.0f}s{' '*30}", flush=True)
            return True
        if proc is not None and proc.poll() is not None:
            print()
            note(f"  process exited (rc={proc.returncode}) without finishing -- see {log}", "r")
            return False
        if any(m in txt for m in FAIL_MARKS):
            print()
            note(f"  FAILED -- see {log}", "r")
            return False
        tail = [l for l in txt.splitlines() if "image '" in l or "seeded L" in l]
        if tail and tail[-1] != last:
            last = tail[-1]
        hint = last.split("] ")[-1][:56] if last else ""
        print(f"\r  {label} {time.time()-t0:5.0f}s  {C['dim']}{hint}{C['off']}", end="", flush=True)
        time.sleep(2)


def stream(fh, first_token_banner=None):
    """Render the blaze event stream until the turn ends. Returns the stats dict."""
    printed = False
    while True:
        line = fh.readline()
        if not line:
            time.sleep(0.02)
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        k = ev.get("ev")
        if k == "handoff" and first_token_banner:
            note(f"\n  handed off: {ev['seeded']} layers x {ev['positions']} positions "
                 f"seeded into the ring -- decode owns it from here\n", "g")
        elif k == "tok":
            if not printed:
                print(f"{C['b']}Response:{C['off']} ", end="", flush=True)
                printed = True
            print(ev["text"], end="", flush=True)
        elif k == "end":
            print()
            g = f"{ev['gen_tok_s']:.1f} tok/s" if ev.get("gen_tok_s") else "n/a"
            note(f"[{ev['n_tokens']} tokens | TTFT: {ev['ttft_s']*1000:.0f}ms | "
                 f"{ev['elapsed_s']:.2f}s | {g} | stop:{ev['stopped']}]")
            if ev.get("stopped") == "max_new":
                note(f"  ^ CUT OFF at the M4_DISAGG_MAX_NEW={ev['n_tokens']} cap -- the model did not "
                     f"stop, the harness did.\n    Re-run with a bigger cap for long answers, e.g. "
                     f"M4_DISAGG_MAX_NEW=1024 M4_DISAGG_BUDGET=4096.\n    Note a truncated reply stays "
                     f"in the KV verbatim, so later turns continue from the broken text.", "y")
            print()
            return ev
        elif k == "ready":
            return None


def _shutdown(ring, grace=120):
    """Quit the ring and GUARANTEE the chips come back, gracefully if possible.

    `/quit` asks the walk to retire by feeding pad pages to the end of its firmware-fixed
    `rounds`, which is the clean path -- but it is NOT reliable: observed stalling in
    `pipeline_block.read_output` waiting for a D2H page that never arrives (2026-09-20, twice).
    When that happens the old code had already returned, leaving eight ranks spinning at 100%
    CPU and all 32 chips claimed, with nothing telling the user.

    So: ask nicely, WAIT, and if the ring has not exited, kill its process GROUP (Popen used
    start_new_session, so the group covers ttrun/mpirun/all 8 ranks) and clear the stale UMD
    locks the hard kill leaves behind. A hard kill also tends to leave the board needing
    `tt-smi -glx_reset`, so say so rather than letting the next run discover it as
    `Read 0xffffffff over PCIe`.
    """
    try:
        with open(FIFO, "w") as f:
            f.write("/quit\n")
    except OSError:
        pass
    note(f"\nquitting: asked the ring to retire; waiting up to {grace}s")
    for _ in range(grace):
        if ring.poll() is not None:
            note("  ring exited cleanly", "g")
            return
        time.sleep(1)

    note("  ring did NOT retire (the known read_output stall) -- killing its process group", "y")
    try:
        os.killpg(os.getpgid(ring.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError) as e:
        note(f"  killpg: {e!r}")
    for _ in range(30):
        if ring.poll() is not None:
            break
        time.sleep(1)
    # Stale CHIP_IN_USE locks survive the kill with dead owner pids, so every later holder
    # check reads the chips as claimed. UMD would recover them via EOWNERDEAD, but the
    # reporting stays wrong until they are gone.
    import glob
    stale = glob.glob("/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*_PCIe")
    for f in stale:
        try:
            os.unlink(f)
        except OSError:
            pass
    note(f"  killed; cleared {len(stale)} stale UMD lock(s)", "y")
    note("  RUN `tt-smi -glx_reset` BEFORE THE NEXT RUN -- a hard kill usually leaves the board "
         "failing at mesh open with 'Read 0xffffffff over PCIe'.", "y")


# ── turn 1: the prompt that gets prefilled ────────────────────────────────────────
print(f"\n{C['b']}Disaggregated prefill -> decode demo{C['off']}  (Mistral Small 4 119B, 1 galaxy)")
note("turn 1 runs tt-metal prefill + a ring capture (~18 min). every turn after is instant.\n")
# A finished dump is fully reusable -- it is just 36 tensors plus meta.json, and nothing in
# it depends on the decode side. Pointing DUMP at one skips the whole prefill leg, which is
# what makes a re-demo of the SAME prompt start in ~9 min instead of ~14.
reuse = Path(DUMP, "meta.json").exists()
if reuse:
    prompt = json.loads(Path(DUMP, "meta.json").read_text())["prompt"]
    note(f"reusing the prefill dump at {DUMP}", "g")
    print(f"First prompt: {prompt}   {C['dim']}(from the dump; prefill leg skipped){C['off']}")
else:
    try:
        prompt = input("First prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
    if not prompt:
        sys.exit(0)

# ── leg 1: tt-metal prefill on all 32 chips, then it exits ────────────────────────
plog = f"{LOGDIR}/prefill_{STAMP}.log"
if not reuse:
    print(f"\n{C['b']}[1/2]{C['off']} tt-metal prefill, 8x4 mesh, 36 layers, real weights")
    note(f"  log: {plog}")
    env = {**os.environ,
           "TT_METAL_HOME": TT_METAL, "PYTHONPATH": TT_METAL,
           "LD_LIBRARY_PATH": f"{TT_METAL}/build_Release/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
           "MISTRAL4_HF_MODEL": os.environ["WEIGHTS"],
           "TT_MISTRAL4_PREFILL_TTNN_CACHE": os.environ["PREFILL_CACHE"],
           "PREFILL_SERVE_SEQ_LEN": os.environ.get("PREFILL_SERVE_SEQ_LEN", "1024"),
           "M4_DUMP_DIR": DUMP, "M4_DUMP_PROMPT": prompt}
    with open(plog, "w") as fh:
        pre = subprocess.Popen(
            [f"{TT_METAL}/python_env/bin/pytest",
             "models/demos/deepseek_v3_d_p/demo/dump_mistral4_prefill_kv.py", "-k", "dump", "-s", "-x"],
            cwd=TT_METAL, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    if not wait_for(plog, ["[m4-dump] DONE"], "prefilling", proc=pre):
        sys.exit(1)
    pre.wait()

meta = json.loads(Path(DUMP, "meta.json").read_text())
mb = sum(f.stat().st_size for f in Path(DUMP).iterdir()) / 1e6
note(f"  {meta['prompt_len']} tokens -> {meta['num_layers']} x [{meta['prompt_len']}, "
     f"{meta['kvpe_dim']}] = {mb:.0f} MB on disk; prefill exited, chips released", "g")

# ── leg 2: tt-blaze reload ring, seeded from that dump, then stays live ───────────
blog = f"{LOGDIR}/blaze_{STAMP}.log"
print(f"\n{C['b']}[2/2]{C['off']} tt-blaze reload ring, 8 stages x 2x2, seeding + capture")
note(f"  log: {blog}")
for f in (FIFO, OUT):
    Path(f).unlink(missing_ok=True)
with open(blog, "w") as fh:
    ring = subprocess.Popen(["./run_ring_chat.sh"], cwd=BLAZE,
                     env={**os.environ, "M4_DISAGG_DUMP": DUMP},
                     stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
# run_ring_chat.sh REDIRECTS ITS OWN OUTPUT into a log of its own and prints only a banner
# to stdout, so `blog` (our capture of that stdout) holds four lines and never another. The
# ring's output -- captures, seeding, tokens -- is in the file that banner names. Watching
# blog instead sits at "starting" forever while the run succeeds in the other file.
inner, t_wait = None, time.time()
while inner is None:
    m = re.search(r"^log: (\S+)$", Path(blog).read_text(errors="ignore"), re.M)
    if m:
        inner = m.group(1)
    elif time.time() - t_wait > 120:
        note(f"  the wrapper never named its log -- see {blog}", "r")
        sys.exit(1)
    else:
        time.sleep(1)
note(f"  ring log: {inner}")
if not wait_for(inner, ["capture starts now", "disagg-chat] tok "], "starting"):
    sys.exit(1)
if not wait_for(inner, ["disagg-chat] tok "], "capturing 25 images"):
    sys.exit(1)

fh = open(OUT)
stream(fh, first_token_banner=True)          # turn 1: the disaggregated answer
while stream(fh) is not None:
    pass

# ── turns 2+: decode only, same cache ─────────────────────────────────────────────
note("subsequent turns are DECODE ONLY -- your tokens go straight into the seeded cache;\n"
     "tt-metal is not involved again, but every reply still attends over its KV.\n", "y")
while True:
    try:
        nxt = input(f"{C['b']}Prompt (q to quit):{C['off']} ").strip()
    except (EOFError, KeyboardInterrupt):
        nxt = "q"
    if nxt.lower() in ("q", "quit", "exit"):
        _shutdown(ring)
        break
    if not nxt:
        continue
    with open(FIFO, "w") as f:
        f.write(nxt + "\n")
    stream(fh)
    while stream(fh) is not None:
        pass
PY
python3 -u "$PYSRC"
