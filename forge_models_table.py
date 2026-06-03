#!/usr/bin/env python3
"""Summarize model configs from the tt-inference-server models-ci-config.json.

Each model in the config either declares an inference engine directly, or carries
an "implementations" list of engine-specific configs. This script flattens those
into one row per (model, engine) config and prints a table.

By default it shows only FORGE models. Use --engine to change/disable that filter.
"""
import argparse
import json
import sys


def iter_configs(models):
    """Yield (model_name, engine, stages_dict) for every model implementation."""
    for name, entry in models.items():
        impls = entry.get("implementations", [entry])
        for impl in impls:
            yield name, impl.get("inference_engine", "?"), impl.get("ci", {})


def collect_rows(models, engine=None, stage=None, device=None):
    rows = []
    for name, eng, ci in iter_configs(models):
        if engine and eng.upper() != engine.upper():
            continue

        # Gather stages -> runner devices, applying optional stage filter.
        stage_devices = {
            st: cfg.get("devices", [])
            for st, cfg in ci.items()
            if not stage or st.lower() == stage.lower()
        }
        if stage and not stage_devices:
            continue  # this config has no matching CI stage

        all_devices = sorted({d for ds in stage_devices.values() for d in ds})
        if device and not any(device.upper() == d.upper() for d in all_devices):
            continue

        rows.append(
            {
                "model": name,
                "engine": eng,
                "stages": sorted(stage_devices.keys()),
                "devices": all_devices,
                "num_devices": len(all_devices),
            }
        )
    return rows


def print_table(rows):
    if not rows:
        print("No matching model configs found.")
        return

    headers = ["MODEL", "ENGINE", "CI STAGES", "RUNNERS (DEVICES)", "#RUN"]
    table = [
        [
            r["model"],
            r["engine"],
            ", ".join(r["stages"]),
            ", ".join(r["devices"]),
            str(r["num_devices"]),
        ]
        for r in rows
    ]

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in table))
        for i in range(len(headers))
    ]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in table:
        print(fmt.format(*row))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config",
                    help="path to models-ci-config.json")
    ap.add_argument("-e", "--engine", default="FORGE",
                    help="filter by inference engine, e.g. FORGE, vLLM, MEDIA "
                         "(default: FORGE; use 'all' for no filter)")
    ap.add_argument("-s", "--stage",
                    help="filter by CI stage: nightly, weekly, or release")
    ap.add_argument("-d", "--device",
                    help="filter by runner/device, e.g. N150, T3K, GALAXY")
    ap.add_argument("--sort", choices=["model", "engine", "num_devices"],
                    default="model", help="column to sort by (default: model)")
    args = ap.parse_args()

    try:
        with open(args.config) as f:
            models = json.load(f)["models"]
    except (OSError, KeyError, json.JSONDecodeError) as e:
        sys.exit(f"error reading config {args.config}: {e}")

    engine = None if args.engine.lower() == "all" else args.engine
    rows = collect_rows(models, engine=engine, stage=args.stage, device=args.device)

    reverse = args.sort == "num_devices"
    rows.sort(key=lambda r: r[args.sort], reverse=reverse)

    print_table(rows)
    print()
    label = (engine or "all").upper() if engine else "all"
    print(f"Total unique model configs ({label}"
          f"{', stage=' + args.stage if args.stage else ''}"
          f"{', device=' + args.device if args.device else ''}"
          f"): {len(rows)}")
    print(f"Distinct model names: {len({r['model'] for r in rows})}")


if __name__ == "__main__":
    main()
