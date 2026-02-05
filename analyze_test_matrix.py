#!/usr/bin/env python3
"""
Analyze test matrix JSON file and estimate durations.

Usage:
    python analyze_test_matrix.py <path-to-matrix.json> [--arch ARCH]

Example:
    python analyze_test_matrix.py .github/workflows/test-matrix-presets/model-test-passing.json --arch qb2-blackhole
"""

import json
import subprocess
import sys
import os
import re
from pathlib import Path
from datetime import datetime

def load_matrix_file(matrix_file):
    """Load the test matrix JSON file."""
    with open(matrix_file, 'r') as f:
        return json.load(f)

def load_test_durations(duration_file):
    """Load test durations from JSON file."""
    if not os.path.exists(duration_file):
        print(f"Warning: {duration_file} not found. Will use default durations.")
        return {}

    with open(duration_file, 'r') as f:
        return json.load(f)

def run_pytest_collect(entry_idx, entry, arch, log_dir, repo_root):
    """Run pytest --collect-only for an entry and return test count and list of test names."""
    test_dir = entry['dir']
    test_mark = entry.get('test-mark', '')

    # Build pytest command
    cmd = [
        'python', '-m', 'pytest',
        '-q', '--collect-only',
        test_dir,
        '--arch', arch
    ]

    if test_mark:
        cmd.extend(['-m', test_mark])

    log_file = log_dir / f"entry_{entry_idx+1}_collect.log"

    print(f"  Entry {entry_idx+1}: Collecting tests... ", end='', flush=True)

    try:
        # Run from repo root
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root
        )

        output = result.stdout + result.stderr

        # Write to log file
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {repo_root}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            f.write(output)

        # Parse test count
        # Look for pattern like "64/3906 tests collected"
        match = re.search(r'(\d+)/\d+ tests? collected', output)
        if not match:
            print("FAILED (see log)")
            return None, []

        test_count = int(match.group(1))

        # Parse individual test names
        # Look for lines like: tests/runner/test_models.py::test_all_models_torch[model-name]
        test_names = []
        for line in output.split('\n'):
            line = line.strip()
            # Match test lines (start with test path, contains ::)
            if '::' in line and line.startswith('tests/'):
                # Remove any trailing comments or extra text
                test_name = line.split()[0] if ' ' in line else line
                test_names.append(test_name)

        print(f"{test_count} tests")
        return test_count, test_names

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return None, []
    except Exception as e:
        print(f"ERROR: {e}")
        return None, []

def calculate_actual_duration(test_names, durations, default_duration=180):
    """Calculate actual total duration by looking up each test's duration."""
    total_duration = 0
    found_count = 0
    missing_tests = []

    for test_name in test_names:
        if test_name in durations:
            total_duration += durations[test_name]
            found_count += 1
        else:
            # Try to find with different path formats
            found = False
            for duration_key in durations.keys():
                if test_name in duration_key or duration_key.endswith(test_name):
                    total_duration += durations[duration_key]
                    found_count += 1
                    found = True
                    break

            if not found:
                total_duration += default_duration
                missing_tests.append(test_name)

    avg_duration = total_duration / len(test_names) if test_names else default_duration

    return total_duration, avg_duration, found_count, missing_tests

def format_duration(seconds):
    """Format duration in readable format."""
    hours = seconds / 3600
    minutes = seconds / 60
    return f"{seconds:,.0f}s ({minutes:.1f}m, {hours:.2f}h)"

def format_seconds(seconds):
    """Format seconds for table column."""
    return f"{seconds:.1f}s"

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_matrix.py <path-to-matrix.json> [--arch ARCH]")
        print("\nExample:")
        print("  python analyze_test_matrix.py .github/workflows/test-matrix-presets/model-test-passing.json --arch qb2-blackhole")
        sys.exit(1)

    matrix_file = sys.argv[1]

    # Parse arch argument
    arch = 'qb2-blackhole'  # default
    if '--arch' in sys.argv:
        arch_idx = sys.argv.index('--arch')
        if arch_idx + 1 < len(sys.argv):
            arch = sys.argv[arch_idx + 1]

    # Get repo root (assume script is run from repo root or has activate sourced)
    repo_root = os.getcwd()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(repo_root) / f"test_matrix_analysis_{timestamp}"
    log_dir.mkdir(exist_ok=True)

    print(f"Test Matrix Analysis")
    print(f"=" * 80)
    print(f"Matrix file: {matrix_file}")
    print(f"Architecture: {arch}")
    print(f"Log directory: {log_dir}")
    print(f"=" * 80)
    print()

    # Load matrix file
    try:
        entries = load_matrix_file(matrix_file)
        print(f"Loaded {len(entries)} entries from matrix file\n")
    except Exception as e:
        print(f"Error loading matrix file: {e}")
        sys.exit(1)

    # Load test durations
    duration_file = Path(repo_root) / '.test_durations'
    durations = load_test_durations(duration_file)
    print(f"Loaded {len(durations)} test durations from .test_durations\n")

    # Collect tests for each entry
    print("Collecting tests for each entry:")
    print("-" * 80)

    results = []
    for idx, entry in enumerate(entries):
        test_count, test_names = run_pytest_collect(idx, entry, arch, log_dir, repo_root)

        if test_count is not None:
            # Calculate actual duration from test names
            total_duration, avg_duration, num_matching, missing_tests = calculate_actual_duration(
                test_names, durations
            )

            results.append({
                'entry_num': idx + 1,
                'name': entry.get('name', f"Entry {idx+1}"),
                'test_mark': entry.get('test-mark', ''),
                'test_count': test_count,
                'avg_duration': avg_duration,
                'total_duration': total_duration,
                'num_matching': num_matching,
                'num_missing': len(missing_tests),
                'parallel_groups': entry.get('parallel-groups', 1)
            })
        else:
            results.append({
                'entry_num': idx + 1,
                'name': entry.get('name', f"Entry {idx+1}"),
                'test_mark': entry.get('test-mark', ''),
                'test_count': None,
                'avg_duration': None,
                'total_duration': None,
                'num_matching': 0,
                'num_missing': 0,
                'parallel_groups': entry.get('parallel-groups', 1)
            })

    print()
    print("=" * 80)
    print()

    # Print summary table
    print("Test Counts and Duration Estimates (Serial Execution):")
    print("=" * 130)
    print(f"{'Entry':<8} {'Tests':<8} {'Parallel':<9} {'Avg/Test':<12} {'Total Duration':<25} {'Found':<8} {'Missing':<8} {'Name':<30}")
    print("=" * 130)

    total_tests = 0
    total_duration = 0
    total_found = 0
    total_missing = 0
    failed_entries = []

    for result in results:
        entry_num = result['entry_num']
        test_count = result['test_count']

        if test_count is None:
            failed_entries.append(entry_num)
            print(f"{entry_num:<8} {'FAILED':<8} {'-':<9} {'-':<12} {'-':<25} {'-':<8} {'-':<8} {result['name'][:30]:<30}")
            continue

        total_tests += test_count
        total_duration += result['total_duration']
        total_found += result['num_matching']
        total_missing += result['num_missing']

        avg_str = format_seconds(result['avg_duration'])
        duration_str = format_duration(result['total_duration'])

        print(f"{entry_num:<8} {test_count:<8} {result['parallel_groups']:<9} {avg_str:<12} {duration_str:<25} {result['num_matching']:<8} {result['num_missing']:<8} {result['name'][:30]:<30}")

    print("=" * 130)
    print()

    if failed_entries:
        print(f"WARNING: {len(failed_entries)} entries failed: {failed_entries}")
        print()

    print(f"TOTAL TESTS: {total_tests}")
    print(f"TOTAL DURATION (SERIAL): {format_duration(total_duration)}")
    if total_tests > 0:
        print(f"AVERAGE PER TEST: {total_duration/total_tests:.1f}s")
    print(f"TESTS WITH ACTUAL DURATIONS: {total_found}/{total_tests} ({100*total_found/total_tests:.1f}%)")
    print(f"TESTS USING DEFAULT (180s): {total_missing}/{total_tests} ({100*total_missing/total_tests:.1f}%)")

    print()
    print(f"Logs saved to: {log_dir}")
    print()

    # Save summary to file
    summary_file = log_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Test Matrix Analysis\n")
        f.write(f"{'='*80}\n")
        f.write(f"Matrix file: {matrix_file}\n")
        f.write(f"Architecture: {arch}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"{'='*80}\n\n")

        f.write("Test Counts and Duration Estimates:\n")
        f.write("="*130 + "\n")
        f.write(f"{'Entry':<8} {'Tests':<8} {'Parallel':<9} {'Avg/Test':<12} {'Total Duration':<25} {'Found':<8} {'Missing':<8} {'Name':<30}\n")
        f.write("="*130 + "\n")

        for result in results:
            entry_num = result['entry_num']
            test_count = result['test_count']

            if test_count is None:
                f.write(f"{entry_num:<8} {'FAILED':<8} {'-':<9} {'-':<12} {'-':<25} {'-':<8} {'-':<8} {result['name'][:30]:<30}\n")
                continue

            avg_str = format_seconds(result['avg_duration'])
            duration_str = format_duration(result['total_duration'])

            f.write(f"{entry_num:<8} {test_count:<8} {result['parallel_groups']:<9} {avg_str:<12} {duration_str:<25} {result['num_matching']:<8} {result['num_missing']:<8} {result['name'][:30]:<30}\n")

        f.write("="*130 + "\n\n")
        f.write(f"TOTAL TESTS: {total_tests}\n")
        f.write(f"TOTAL DURATION (SERIAL): {format_duration(total_duration)}\n")
        if total_tests > 0:
            f.write(f"AVERAGE PER TEST: {total_duration/total_tests:.1f}s\n")
        f.write(f"TESTS WITH ACTUAL DURATIONS: {total_found}/{total_tests} ({100*total_found/total_tests:.1f}%)\n")
        f.write(f"TESTS USING DEFAULT (180s): {total_missing}/{total_tests} ({100*total_missing/total_tests:.1f}%)\n")

    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
