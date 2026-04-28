#!/usr/bin/env python3
"""
Download nightly CI artifacts from tt-xla across multiple runs.

Given an example job URL, derives the artifact name pattern and downloads
matching artifacts for every nightly run in the specified date range.

Usage:
  python fetch_ci_logs.py --job-url URL
  python fetch_ci_logs.py --job-url URL --since 2026-04-01 --until 2026-04-28
  python fetch_ci_logs.py --job-url URL --since 2026-04-01 --output-dir my_logs
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


REPO = "tenstorrent/tt-xla"


def gh_api(endpoint: str) -> dict:
    result = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def get_run_artifacts(repo: str, run_id: int) -> list[dict]:
    artifacts = []
    page = 1
    while True:
        data = gh_api(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}")
        artifacts.extend(data["artifacts"])
        if len(data["artifacts"]) < 100:
            break
        page += 1
    return artifacts


def find_artifact_prefix(repo: str, run_id: int, job_id: int, contains: str | None) -> str:
    """
    Return artifact name prefix for the artifact ending with -{job_id}.
    If multiple candidates exist, requires `contains` to disambiguate.
    """
    artifacts = get_run_artifacts(repo, run_id)
    suffix = f"-{job_id}"
    candidates = [a["name"] for a in artifacts if a["name"].endswith(suffix)]

    if not candidates:
        raise SystemExit(
            f"ERROR: no artifact ending in -{job_id} found in run {run_id}.\n"
            f"Available artifacts:\n" + "\n".join(f"  {a['name']}" for a in artifacts)
        )

    if contains:
        candidates = [n for n in candidates if contains in n]
        if not candidates:
            raise SystemExit(
                f"ERROR: no artifact ending in -{job_id} with {contains!r} in name.\n"
                f"Candidates (before filter):\n" + "\n".join(f"  {n}" for n in
                    [a["name"] for a in artifacts if a["name"].endswith(suffix)])
            )

    if len(candidates) > 1:
        raise SystemExit(
            f"ERROR: multiple artifacts match job {job_id} — use --name-contains to pick one:\n"
            + "\n".join(f"  {n}" for n in candidates)
        )

    name = candidates[0]
    return name[: -len(str(job_id))]  # keep trailing dash


def get_workflow_id(repo: str, run_id: int) -> tuple[int, str]:
    data = gh_api(f"repos/{repo}/actions/runs/{run_id}")
    return data["workflow_id"], data["name"]


def list_runs_in_range(repo: str, workflow_id: int, since: date, until: date) -> list[dict]:
    """Return all scheduled workflow runs with created_at in [since, until]."""
    runs = []
    page = 1
    # GitHub date range filter uses YYYY-MM-DD..YYYY-MM-DD
    date_filter = f"{since}..{until}"
    while True:
        endpoint = (
            f"repos/{repo}/actions/workflows/{workflow_id}/runs"
            f"?per_page=100&page={page}&event=schedule&created={date_filter}"
        )
        data = gh_api(endpoint)
        runs.extend(data["workflow_runs"])
        if len(data["workflow_runs"]) < 100:
            break
        page += 1
    return runs


def find_matching_artifact(repo: str, run_id: int, prefix: str) -> str | None:
    """Return the artifact name that starts with prefix, or None."""
    artifacts = get_run_artifacts(repo, run_id)
    for a in artifacts:
        if a["name"].startswith(prefix):
            return a["name"]
    return None


def download_artifact(repo: str, run_id: int, artifact_name: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gh", "run", "download", str(run_id),
         "--repo", repo, "--name", artifact_name, "-D", str(dest)],
        check=True,
    )


def parse_job_url(url: str) -> tuple[int, int]:
    m = re.search(r"/runs/(\d+)/job/(\d+)", url)
    if not m:
        raise SystemExit(f"ERROR: cannot parse run_id/job_id from URL: {url}")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-url", required=True,
                        help="Example GitHub Actions job URL to derive artifact pattern from")
    parser.add_argument("--since", metavar="YYYY-MM-DD", default=None,
                        help="Start date inclusive (default: 2 weeks ago)")
    parser.add_argument("--until", metavar="YYYY-MM-DD", default=None,
                        help="End date inclusive (default: today)")
    parser.add_argument("--output-dir", default="ci_logs", metavar="DIR",
                        help="Root directory for downloads (default: ci_logs)")
    parser.add_argument("--name-contains", default=None, metavar="STR",
                        help="Substring to disambiguate when multiple artifacts share a job_id "
                             "(e.g. 'test-log' vs 'test-reports')")
    parser.add_argument("--repo", default=REPO,
                        help=f"GitHub repo slug (default: {REPO})")
    args = parser.parse_args()

    today = date.today()
    since = date.fromisoformat(args.since) if args.since else today - timedelta(weeks=2)
    until = date.fromisoformat(args.until) if args.until else today

    ref_run_id, ref_job_id = parse_job_url(args.job_url)
    print(f"Reference run {ref_run_id}, job {ref_job_id}")

    prefix = find_artifact_prefix(args.repo, ref_run_id, ref_job_id, args.name_contains)
    print(f"Artifact prefix: {prefix!r}")

    workflow_id, workflow_name = get_workflow_id(args.repo, ref_run_id)
    print(f"Workflow: {workflow_name!r} (id {workflow_id})")
    print(f"Date range: {since} → {until}")

    runs = list_runs_in_range(args.repo, workflow_id, since, until)
    print(f"Found {len(runs)} runs in range\n")

    if not runs:
        print("Nothing to download.")
        return

    output_dir = Path(args.output_dir)
    downloaded = skipped = missed = errored = 0

    for run in sorted(runs, key=lambda r: r["created_at"]):
        run_id = run["id"]
        run_date = run["created_at"][:10]
        dest = output_dir / f"{run_date}_{run_id}"

        # Skip if already populated
        if dest.exists() and any(dest.iterdir()):
            print(f"  SKIP  {run_date}  run={run_id}  (already present at {dest})")
            skipped += 1
            continue

        artifact_name = find_matching_artifact(args.repo, run_id, prefix)
        if artifact_name is None:
            print(f"  MISS  {run_date}  run={run_id}  (no matching artifact)")
            missed += 1
            continue

        print(f"  GET   {run_date}  run={run_id}  → {artifact_name}")
        try:
            download_artifact(args.repo, run_id, artifact_name, dest)
            files = list(dest.iterdir())
            print(f"        → {dest}/ ({len(files)} file(s))")
            downloaded += 1
        except subprocess.CalledProcessError as exc:
            print(f"        ERROR: {exc}", file=sys.stderr)
            errored += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {missed} missing, {errored} errors")


if __name__ == "__main__":
    main()
