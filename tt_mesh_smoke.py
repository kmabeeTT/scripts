#!/usr/bin/env python3
"""tt_mesh_smoke — prove every chip in a mesh is alive and computing.

Runs a tiny eltwise-add and a tiny matmul on each mesh shape, with *different data on
every device*, then checks each device's slab independently. A dead, mis-mapped, or
silently-wrong chip shows up as a named device index rather than a vague failure.

Deliberately local-only: no CCL, no fabric, no all-gather. Those test the interconnect,
hang when the topology doesn't match, and are the usual reason a "is the board OK" check
turns into a 20-minute debugging session. This asks one question — does each chip compute?

Runtime is dominated by device open (~3s for 32 chips); the math itself is microseconds.

There is also an opt-in interconnect ladder (--interconnect) that does exercise the fabric:
an all_gather on a 2-chip submesh, then on every row of the mesh at once, then every
column. Each device is checked for having received exactly the slabs it should have, by
content, so it proves data physically crossed links rather than just that ops returned.

Two things about the fabric path, both learned the hard way on a BH Galaxy:

  * TT_METAL_OPERATION_TIMEOUT_SECONDS bounds *dispatch ops*, not fabric bring-up. A
    stalled all_gather raises cleanly; a fabric that cannot route blocks forever inside
    open_mesh_device, where nothing rescues you. Killing that is what wedges an ethernet
    core and costs a `tt-smi -glx_reset`.
  * FABRIC_1D (line) works on this box. FABRIC_1D_RING (--ring) hangs at bring-up, even
    though the physical grouping descriptor advertises TORUSX/TORUSY/TORUSXY matches —
    the descriptor matching a torus does not mean the ring routes. Treat --ring as an
    experiment on a box you can afford to reset.

Usage (standalone):
    python3 ~/scripts/tt_mesh_smoke.py                # per-chip liveness, every shape that fits
    python3 ~/scripts/tt_mesh_smoke.py --shape 8x4    # one shape
    python3 ~/scripts/tt_mesh_smoke.py --list         # what fits on this host
    python3 ~/scripts/tt_mesh_smoke.py --interconnect # fabric ladder instead of liveness
    python3 ~/scripts/tt_mesh_smoke.py --stage pair   # one interconnect stage
    python3 ~/scripts/tt_mesh_smoke.py --all          # liveness, then the fabric ladder

Usage (pytest):
    pytest ~/scripts/tt_mesh_smoke.py -v
    pytest ~/scripts/tt_mesh_smoke.py -v -k 8x4
    TT_MESH_SMOKE_INTERCONNECT=1 pytest ~/scripts/tt_mesh_smoke.py -v -k interconnect

Needs TT_METAL_HOME + PYTHONPATH set and the tt-metal venv active, same as any ttnn run.
Exit code is 0 only if every chip in every requested shape passed.
"""

import argparse
import os
import sys
import time

import torch

# Bound every device-side op *before* ttnn opens anything. This is the single most
# important line in the interconnect path: a stalled fabric op then raises a normal
# Python exception and unwinds cleanly. Killing a hung fabric op from outside (kill -9)
# is what leaves an ethernet RISC wedged and the board needing a reset.
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")

# Topology for this box. Fabric needs a mesh graph descriptor; without one the runtime
# falls back to auto-discovery, which is what leaves ring/torus configs unroutable. This
# has to be set before ttnn is imported — the control plane reads it when the cluster is
# constructed, so setting it later silently has no effect.
DEFAULT_MGD = "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"
MGD_AUTOSET = False


def _looks_like_galaxy():
    """32 chips visible? Cheap arch-free check — ttnn isn't importable yet at this point.

    Deliberately not `os.path.exists(descriptor)`: that file ships in every tt-metal
    checkout, so existence says nothing about the hardware. Exporting a Galaxy topology
    on a LoudBox or a P150 is precisely how you get an unroutable fabric and an
    unbounded hang in open_mesh_device.
    """
    try:
        return sum(1 for n in os.listdir("/dev/tenstorrent") if n.isdigit()) == 32
    except OSError:
        return False


if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    # abspath is load-bearing. With TT_METAL_HOME unset, join() yields a *relative* path,
    # and os.path.exists() still says True whenever cwd happens to be the repo root — so
    # a relative path gets exported, and metal resolves it from / instead, dying with
    # "Custom mesh graph descriptor file not found: /tt_metal/fabric/...".
    _mgd = os.path.abspath(os.path.join(os.environ.get("TT_METAL_HOME", ""), DEFAULT_MGD))
    if os.path.exists(_mgd) and _looks_like_galaxy():
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = _mgd
        MGD_AUTOSET = True
else:
    # An inherited descriptor path is trusted by the runtime and only validated deep in
    # cluster construction, where it surfaces as a TT_FATAL from metal_env.cpp long after
    # you've forgotten you exported it. The classic bad value is
    # "/tt_metal/fabric/..." — the result of `export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/...`
    # with TT_METAL_HOME unset. Catch it here, where the fix is obvious.
    _inherited = os.environ["TT_MESH_GRAPH_DESC_PATH"]
    if not os.path.exists(_inherited):
        sys.exit(
            f"tt_mesh_smoke: TT_MESH_GRAPH_DESC_PATH points at a file that does not exist:\n"
            f"    {_inherited}\n"
            f"  If it starts at /tt_metal/, it was built from an unset TT_METAL_HOME.\n"
            f"  Fix with:  export TT_METAL_HOME=/path/to/tt-metal\n"
            f"             export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/{DEFAULT_MGD}\n"
            f"  Or just `unset TT_MESH_GRAPH_DESC_PATH` — this script picks the right\n"
            f"  descriptor itself on a 32-chip Galaxy."
        )

import ttnn

# (rows, cols). 1x1 = single chip; 8x1 = one column of a Galaxy; 8x4 = whole 32-chip mesh.
SHAPES = {"1x1": (1, 1), "8x1": (8, 1), "8x4": (8, 4)}

TILE = 32  # one tile per device — smallest thing the matmul path will take

# Per-op thresholds. With well-conditioned inputs and a bf16-rounded reference, healthy
# Blackhole silicon measures 1.00000 (add) and 0.99999 (matmul) on every chip, so these
# gates sit far below observed accuracy while still catching real degradation. A dead or
# wrong chip scores near 0 — nowhere near either.
PCC_ADD = 0.999
PCC_MATMUL = 0.999


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors, flattened. 1.0 == identical."""
    x, y = a.flatten().float(), b.flatten().float()
    if torch.equal(x, y):
        return 1.0
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        return 1.0 if x.norm() == y.norm() else 0.0
    return float((x @ y) / denom)


def make_inputs(num_devices):
    """Two [1,1,32,32*N] tensors whose every 32-wide slab is unique to its device.

    Each device gets its own seeded N(0,1) block, so a wrong result can only have come
    from that device — no slab can be mistaken for another's, and the seed keeps runs
    reproducible.

    Two things this data deliberately avoids, both of which read as hardware faults when
    they are really numerics:
      * an additive per-device offset (ramp + i) — past ~4.0 the bf16 step exceeds the
        ramp's, quantizing the signal away on exactly the higher-numbered devices;
      * a constant operand — it makes the matmul output nearly DC, and PCC subtracts the
        mean, so ordinary rounding noise dominates what little variance is left.
    """
    a = torch.zeros(1, 1, TILE, TILE * num_devices)
    b = torch.zeros(1, 1, TILE, TILE * num_devices)
    for i in range(num_devices):
        sl = slice(i * TILE, (i + 1) * TILE)
        gen = torch.Generator().manual_seed(1000 + i)
        a[0, 0, :, sl] = torch.randn(TILE, TILE, generator=gen)
        b[0, 0, :, sl] = torch.randn(TILE, TILE, generator=gen)
    return a, b


def as_device_sees(t: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and back, so the reference starts from the values the chip actually got.

    Without this the reference carries fp32 inputs the device never saw, and the resulting
    input-quantization error is indistinguishable from a compute fault.
    """
    return t.to(torch.bfloat16).float()


def check_per_device(got, ref, num_devices, op_name, threshold):
    """Compare each device's slab separately. Returns list of (device_index, pcc) failures."""
    bad, worst = [], 1.0
    for i in range(num_devices):
        sl = slice(i * TILE, (i + 1) * TILE)
        p = pcc(got[0, 0, :, sl], ref[0, 0, :, sl])
        worst = min(worst, p)
        if p < threshold:
            bad.append((i, p))
    if bad:
        detail = ", ".join(f"dev{i} pcc={p:.4f}" for i, p in bad)
        print(f"    {op_name}: FAILED on {len(bad)}/{num_devices} device(s) -> {detail}")
    else:
        print(f"    {op_name}: all {num_devices} devices correct (worst pcc {worst:.5f})")
    return bad


def run_shape(rows, cols, verbose=True):
    """Open the mesh, run both ops, verify every device. Returns True if all chips passed."""
    num_devices = rows * cols
    label = f"{rows}x{cols}"
    t0 = time.time()

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    try:
        opened = time.time() - t0
        ids = list(mesh.get_device_ids()) if hasattr(mesh, "get_device_ids") else []
        # Physical ids are not contiguous on a Galaxy (e.g. [0, 4, 12, 8, ...]), so print
        # the list rather than a range — a range would misreport which chips were used.
        if verbose:
            shown = ",".join(str(i) for i in ids[:8]) + ("..." if len(ids) > 8 else "")
            print(f"  mesh {label}: opened {num_devices} devices in {opened:.1f}s"
                  + (f" (chips {shown})" if ids else ""))

        torch_a, torch_b = make_inputs(num_devices)
        mapper = ttnn.shard_tensor_to_mesh_mapper(mesh, dim=-1)
        composer = ttnn.concat_mesh_to_tensor_composer(mesh, dim=-1)

        kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
        tt_a = ttnn.from_torch(torch_a, **kw)
        tt_b = ttnn.from_torch(torch_b, **kw)

        q_a, q_b = as_device_sees(torch_a), as_device_sees(torch_b)

        # Eltwise add: exercises the compute grid on every chip.
        out_add = ttnn.to_torch(ttnn.add(tt_a, tt_b), mesh_composer=composer)
        ref_add = q_a + q_b

        # Matmul: exercises a different kernel path (and the matmul engine) per chip.
        out_mm = ttnn.to_torch(ttnn.matmul(tt_a, tt_b), mesh_composer=composer)
        ref_mm = torch.zeros_like(ref_add)
        for i in range(num_devices):
            sl = slice(i * TILE, (i + 1) * TILE)
            ref_mm[0, 0, :, sl] = q_a[0, 0, :, sl] @ q_b[0, 0, :, sl]

        bad = check_per_device(out_add, ref_add, num_devices, "eltwise add", PCC_ADD)
        bad += check_per_device(out_mm, ref_mm, num_devices, "matmul    ", PCC_MATMUL)
    finally:
        ttnn.close_mesh_device(mesh)

    elapsed = time.time() - t0
    status = "PASS" if not bad else "FAIL"
    print(f"  mesh {label}: {status} ({elapsed:.1f}s total)\n")
    return not bad


def shapes_that_fit():
    avail = ttnn.get_num_devices()
    return {k: v for k, v in SHAPES.items() if v[0] * v[1] <= avail}


# ------------------------------------------------------------------ interconnect stages
#
# Escalating deliberately: a failure at "pair" means something very different from a
# failure only at "mesh", and stopping at the first failure keeps a broken fabric from
# being exercised at full width. cluster_axis=None gathers across the whole (1-D) mesh;
# on the 2-D mesh we gather along each axis separately so a bad row and a bad column are
# distinguishable.
#
# FABRIC_1D is a line topology and is the default everywhere here. FABRIC_1D_RING needs
# the mesh to be physically ring/torus cabled — on a plain-mesh Galaxy it does not route
# and the op hangs, so it is opt-in via --ring rather than part of the ladder.

# Fabric firmware initializes across the whole mesh, not a slice of it: opening a bare 1x2
# with fabric enabled fails in fabric_firmware_initializer with cores stuck at STARTED.
# So the parent mesh is opened once, and each stage runs on a submesh carved out of it —
# the same shape tt-metal's own CCL tests use. "sub" stages carve; "axis" stages gather
# along one axis of the parent.
STAGES = [
    {"name": "pair",   "sub": (1, 2), "axis": None, "why": "two adjacent chips exchange at all"},
    {"name": "mesh-x", "sub": None,   "axis": 1,    "why": "every row of the full mesh at once"},
    {"name": "mesh-y", "sub": None,   "axis": 0,    "why": "every column of the full mesh at once"},
]

# Only "pair" carves a submesh. Separate row/column stages were dropped: gathering on the
# parent with cluster_axis covers both axes at full width anyway, and two live submeshes
# overlapping the same devices poisons dispatch — the second one dies in
# system_memory_manager after burning the whole op timeout, though it passes in 0.1s when
# it runs first. Each stage therefore gets a freshly opened parent mesh.


def expected_groups(rows, cols, axis):
    """Which source slabs each device should end up holding after the gather.

    Device order from get_device_tensors is mesh-linear (row-major). axis=None means the
    whole mesh gathers together; axis=1 groups within a row; axis=0 groups within a column.
    """
    groups = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if axis is None:
            groups.append(set(range(rows * cols)))
        elif axis == 1:
            groups.append({r * cols + cc for cc in range(cols)})
        else:
            groups.append({rr * cols + c for rr in range(rows)})
    return groups


def check_gathered(out_tt, q_input, groups, label):
    """Verify each device received exactly the slabs it should have.

    Matching is by slab signature (sum) rather than by position, so this does not depend
    on the concatenation order the op happens to use — it asks the question that matters:
    did this chip physically receive the other chips' data, and is it intact?
    """
    sigs = [float(q_input[0, 0, :, i * TILE:(i + 1) * TILE].sum()) for i in range(q_input.shape[-1] // TILE)]
    per_device = ttnn.get_device_tensors(out_tt)
    bad = []

    for dev_i, dev_t in enumerate(per_device):
        got = ttnn.to_torch(dev_t).float()
        n_slabs = got.shape[-1] // TILE
        received, corrupt = set(), []
        for s in range(n_slabs):
            slab = got[0, 0, :, s * TILE:(s + 1) * TILE]
            src = min(range(len(sigs)), key=lambda k: abs(sigs[k] - float(slab.sum())))
            p = pcc(slab, q_input[0, 0, :, src * TILE:(src + 1) * TILE])
            if p < PCC_ADD:
                corrupt.append((src, p))
            else:
                received.add(src)
        want = groups[dev_i]
        if received != want or corrupt:
            missing = sorted(want - received)
            note = f"missing {missing}" if missing else ""
            if corrupt:
                note += (" " if note else "") + "corrupt " + ",".join(f"src{s}(pcc={p:.3f})" for s, p in corrupt)
            bad.append((dev_i, note or f"got {sorted(received)} want {sorted(want)}"))

    if bad:
        detail = "; ".join(f"dev{i}: {n}" for i, n in bad[:6])
        more = f" (+{len(bad) - 6} more)" if len(bad) > 6 else ""
        print(f"    {label}: FAILED on {len(bad)}/{len(per_device)} device(s) -> {detail}{more}")
    else:
        width = ttnn.to_torch(per_device[0]).shape[-1] // TILE
        print(f"    {label}: all {len(per_device)} devices received their {width} slabs intact")
    return bad


def run_stage(parent, stage, verbose=True):
    """Run one stage on the already-open parent mesh. Returns True on pass."""
    label = stage["name"]
    axis = stage["axis"]
    t0 = time.time()
    submesh = None

    try:
        if stage["sub"]:
            rows, cols = stage["sub"]
            submesh = parent.create_submesh(ttnn.MeshShape(rows, cols), offset=ttnn.MeshCoordinate(0, 0))
            dev, shape_note = submesh, f"submesh {rows}x{cols}"
        else:
            rows, cols = tuple(parent.shape)
            dev, shape_note = parent, f"full {rows}x{cols}"

        num_devices = rows * cols
        if verbose:
            axis_note = "gather across all" if axis is None else f"cluster_axis={axis}"
            print(f"  stage {label:>7} [{shape_note}, {axis_note}]: {stage['why']}")

        torch_in, _ = make_inputs(num_devices)
        q_in = as_device_sees(torch_in)
        mapper = ttnn.shard_tensor_to_mesh_mapper(dev, dim=-1)
        tt_in = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev, mesh_mapper=mapper)

        kwargs = {"dim": 3} if axis is None else {"dim": 3, "cluster_axis": axis}
        out = ttnn.all_gather(tt_in, **kwargs)

        bad = check_gathered(out, q_in, expected_groups(rows, cols, axis), "all_gather")
    except Exception as exc:
        print(f"    all_gather: ERROR — {type(exc).__name__}: {str(exc).splitlines()[0][:180]}")
        bad = [(-1, "exception")]

    # Submeshes are deliberately *not* closed here. Closing one and then carving another
    # from the same parent corrupts dispatch state — the next stage dies in
    # system_memory_manager after burning the full op timeout, while that same stage
    # passes in 0.1s when run first. tt-metal's own CCL tests hold submeshes open and
    # release them all at parent teardown; run_interconnect does the same.

    print(f"  stage {label:>7}: {'PASS' if not bad else 'FAIL'} ({time.time() - t0:.1f}s)\n")
    return not bad


def run_interconnect(selected=None, ring=False, verbose=True):
    """Open the full mesh with fabric once, then run the ladder, stopping at first failure."""
    avail = ttnn.get_num_devices()
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D

    mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    print(f"  mesh graph descriptor: {os.path.basename(mgd) if mgd else 'NONE (auto-discovery)'}")

    # The auto-set descriptor is Blackhole-Galaxy-specific. If we guessed it from the chip
    # count alone and the arch disagrees, refuse rather than risk an unroutable fabric —
    # a bad bring-up hangs with nothing to rescue it.
    if MGD_AUTOSET and ttnn.get_arch_name() != "blackhole":
        print(f"  REFUSING: auto-selected a BH Galaxy descriptor but arch is "
              f"{ttnn.get_arch_name()}. Set TT_MESH_GRAPH_DESC_PATH explicitly for this box.\n")
        return {"topology-check": False}
    if ring:
        print("  WARNING: FABRIC_1D_RING bring-up is NOT covered by the op timeout — that bounds\n"
              "           dispatch ops only. If the ring does not route, open_mesh_device blocks\n"
              "           indefinitely and killing it can wedge an ethernet core. Recovery is\n"
              "           `tt-smi -glx_reset`. Observed to hang on this box (see --help).")

    results = {}
    for stage in STAGES:
        if selected and stage["name"] not in selected:
            continue
        need = (stage["sub"][0] * stage["sub"][1]) if stage["sub"] else avail
        if need > avail:
            print(f"  stage {stage['name']:>7}: SKIP (needs {need} devices, host has {avail})\n")
            results[stage["name"]] = None
            continue

        # Fresh fabric + parent per stage. Costs ~2.5s each, and buys independence: a
        # stage can't be poisoned by dispatch state its predecessor left behind, and
        # re-running bring-up each time exercises it repeatedly rather than once.
        ttnn.set_fabric_config(fabric)
        parent = None
        try:
            shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape() if mgd else ttnn.MeshShape(8, 4)
            t0 = time.time()
            parent = ttnn.open_mesh_device(mesh_shape=shape)
            print(f"  fabric {fabric.name} up on {tuple(parent.shape)} "
                  f"({parent.get_num_devices()} devices) in {time.time() - t0:.1f}s")
            ok = run_stage(parent, stage, verbose)
        except Exception as exc:
            print(f"  fabric bring-up FAILED — {type(exc).__name__}: {str(exc).splitlines()[0][:180]}\n")
            ok = False
        finally:
            if parent is not None:
                for sub in parent.get_submeshes():
                    ttnn.close_mesh_device(sub)
                ttnn.close_mesh_device(parent)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

        results[stage["name"]] = ok
        if not ok:
            print(f"  stopping: {stage['name']} failed — not escalating to a wider mesh\n")
            break

    return results


# ---------------------------------------------------------------- pytest entry points

def pytest_generate_tests(metafunc):
    if "mesh_shape_name" in metafunc.fixturenames:
        metafunc.parametrize("mesh_shape_name", list(SHAPES), ids=list(SHAPES))
    if "stage_name" in metafunc.fixturenames:
        names = [s["name"] for s in STAGES]
        metafunc.parametrize("stage_name", names, ids=names)


def test_interconnect(stage_name):
    """One fabric stage: every device receives exactly the slabs it should have.

    Off by default — this brings fabric up, which is slower and carries the bring-up hang
    risk described in the module docstring. Enable with TT_MESH_SMOKE_INTERCONNECT=1.
    """
    import pytest

    if os.environ.get("TT_MESH_SMOKE_INTERCONNECT") != "1":
        pytest.skip("interconnect stages are opt-in: set TT_MESH_SMOKE_INTERCONNECT=1")

    stage = next(s for s in STAGES if s["name"] == stage_name)
    avail = ttnn.get_num_devices()
    need = (stage["sub"][0] * stage["sub"][1]) if stage["sub"] else avail
    if need > avail:
        pytest.skip(f"stage {stage_name} needs {need} devices, host has {avail}")

    results = run_interconnect(selected=[stage_name], ring=False)
    assert results.get(stage_name), f"interconnect stage {stage_name} failed"


def test_mesh_alive(mesh_shape_name):
    """Every chip in the mesh computes an eltwise add and a matmul correctly."""
    import pytest

    rows, cols = SHAPES[mesh_shape_name]
    avail = ttnn.get_num_devices()
    if rows * cols > avail:
        pytest.skip(f"mesh {mesh_shape_name} needs {rows * cols} devices, host has {avail}")
    assert run_shape(rows, cols), f"mesh {mesh_shape_name}: at least one chip produced wrong results"


# ---------------------------------------------------------------- standalone entry point

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", action="append", choices=list(SHAPES),
                    help="mesh shape to test (repeatable); default is every shape that fits")
    ap.add_argument("--list", action="store_true", help="show shapes that fit on this host and exit")
    ap.add_argument("--interconnect", action="store_true",
                    help="run the fabric ladder (pair -> every row -> every column) instead of "
                         "the per-chip liveness pass")
    ap.add_argument("--all", action="store_true",
                    help="run liveness for every shape that fits, then the fabric ladder")
    ap.add_argument("--stage", action="append", choices=[s["name"] for s in STAGES],
                    help="run only these interconnect stages (implies --interconnect)")
    ap.add_argument("--ring", action="store_true",
                    help="use FABRIC_1D_RING instead of FABRIC_1D; needs ring/torus cabling and "
                         "hangs without it — opt in only if you know the box is cabled for it")
    ap.add_argument("--op-timeout", type=int, metavar="SEC",
                    help="per-op dispatch timeout in seconds (default 30); a stalled fabric op "
                         "then raises instead of wedging the board")
    args = ap.parse_args()

    if args.op_timeout:
        os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = str(args.op_timeout)

    avail = ttnn.get_num_devices()
    fits = shapes_that_fit()

    if args.list:
        print(f"{avail} devices visible ({ttnn.get_arch_name()})")
        for name, (r, c) in SHAPES.items():
            print(f"  {name:>4}  {r * c:>2} devices  {'ok' if name in fits else 'does not fit'}")
        return 0

    want_ccl = args.all or args.interconnect or bool(args.stage)
    want_liveness = args.all or not want_ccl
    print(f"tt_mesh_smoke: {avail} devices visible ({ttnn.get_arch_name()}), "
          f"op timeout {os.environ['TT_METAL_OPERATION_TIMEOUT_SECONDS']}s\n")

    results = {}

    if want_liveness:
        for name in (args.shape or list(fits)):
            rows, cols = SHAPES[name]
            if rows * cols > avail:
                print(f"  mesh {name}: SKIP (needs {rows * cols} devices, host has {avail})\n")
                results[name] = None
                continue
            try:
                results[name] = run_shape(rows, cols)
            except Exception as exc:  # a failed open is a result, not a crash
                print(f"  mesh {name}: ERROR — {type(exc).__name__}: {exc}\n")
                results[name] = False

    if want_ccl:
        # Only reached with --all / --interconnect / --stage, never by default: fabric
        # bring-up is not covered by the op timeout, so it stays something you ask for.
        print(f"interconnect ladder ({'FABRIC_1D_RING' if args.ring else 'FABRIC_1D'}):\n")
        results.update(run_interconnect(selected=args.stage, ring=args.ring))

    ran = {k: v for k, v in results.items() if v is not None}
    passed = sum(1 for v in ran.values() if v)
    summary = "  ".join(f"{k}={'pass' if v else 'FAIL' if v is not None else 'skip'}"
                        for k, v in results.items())
    print(f"summary: {summary}")
    return 0 if ran and passed == len(ran) else 1


if __name__ == "__main__":
    sys.exit(main())
