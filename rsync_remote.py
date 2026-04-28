#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Rsync a directory from a remote machine to the current directory, preserving
the trailing portion of the remote path as the local destination.

Example:
    Remote path: ~/kmabee_demo/tt-xla-2/tracy_sampler_nongreedy_tt_sampling_pad32_batch2/reports/2026_04_28_18_26_05
    Local dest:  ./tracy_sampler_nongreedy_tt_sampling_pad32_batch2/reports/2026_04_28_18_26_05/

Usage:
    python3 rsync_remote.py <remote_path>
    python3 rsync_remote.py <remote_path> --host 10.32.48.99
    python3 rsync_remote.py <remote_path> --user ttuser
    python3 rsync_remote.py <remote_path> --strip "~/some/other/prefix/"
    python3 rsync_remote.py <remote_path> --dest ./my_local_dir
    python3 rsync_remote.py <remote_path> --dry-run
"""
import argparse
import os
import subprocess
import sys
from pathlib import PurePosixPath

DEFAULT_HOST = "10.32.48.16"
DEFAULT_STRIP = "~/kmabee_demo/tt-xla-2/"


def derive_local_dest(remote_path: str, strip: str) -> str:
    remote = remote_path.rstrip("/")
    candidates = [strip.rstrip("/") + "/"]
    # Also try with `~/` and `~` stripped, since rsync expands `~` server-side
    # but the literal path in the arg keeps it.
    if strip.startswith("~/"):
        candidates.append(strip[2:].rstrip("/") + "/")
    for cand in candidates:
        if remote.startswith(cand):
            return remote[len(cand):]
    # Fallback: just use the basename
    return PurePosixPath(remote).name


def main():
    parser = argparse.ArgumentParser(
        description="Rsync a remote directory into the current working directory."
    )
    parser.add_argument("remote_path", help="Path on the remote machine (e.g. ~/foo/bar)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Remote host (default: {DEFAULT_HOST})")
    parser.add_argument("--user", default=os.environ.get("USER"),
                        help="Remote SSH user (default: $USER)")
    parser.add_argument("--strip", default=DEFAULT_STRIP,
                        help=f"Prefix to strip from the remote path to form the local "
                             f"destination (default: {DEFAULT_STRIP!r})")
    parser.add_argument("--dest", default=None,
                        help="Explicit local destination directory (overrides --strip)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be transferred without doing it")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable compression (useful on a fast LAN)")
    args = parser.parse_args()

    if args.dest is not None:
        local_dest = args.dest
    else:
        local_dest = derive_local_dest(args.remote_path, args.strip)
        local_dest = os.path.join(".", local_dest)

    os.makedirs(local_dest, exist_ok=True)

    flags = "-avh"
    if not args.no_compress:
        flags += "z"
    cmd = ["rsync", flags, "--partial", "--info=progress2"]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += [
        f"{args.user}@{args.host}:{args.remote_path.rstrip('/')}/",
        local_dest.rstrip("/") + "/",
    ]

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
