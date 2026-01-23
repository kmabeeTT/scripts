#!/usr/bin/env python3
"""
Analyze TT-XLA test failures from debug logs.

Identifies:
- The failing TTNN operation
- The error message
- The MLIR location tag
- Corresponding TTIR operations (if IR dumps are available)

Usage:
    python analyze_failure.py <log_file> [--output <report_file>]
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime


def find_failing_operation(log_file):
    """Find the failing TTNN operation and error."""

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find TT_THROW error
    error_line = None
    error_msg = None
    for i, line in enumerate(lines):
        if 'TT_THROW' in line or 'Statically allocated circular buffers' in line:
            error_line = i
            error_msg = line.strip()
            break

    if not error_line:
        return None, None, None

    # Find the last executing operation before error
    failing_op = None
    failing_op_line = None
    for i in range(error_line - 1, max(0, error_line - 100), -1):
        if 'Executing operation:' in lines[i]:
            failing_op = lines[i].strip()
            failing_op_line = i
            break

    # Extract MLIR location tag
    location_tag = None
    if failing_op:
        loc_match = re.search(r'loc\("([^"]+)"\)', failing_op)
        if loc_match:
            location_tag = loc_match.group(1)

    return {
        'error_line': error_line + 1,
        'error_msg': error_msg,
        'failing_op': failing_op,
        'failing_op_line': failing_op_line + 1 if failing_op_line else None,
        'location_tag': location_tag
    }


def find_ttir_operations(log_file, location_tag):
    """Find TTIR operations with matching location tag."""

    with open(log_file, 'r') as f:
        content = f.read()

    if not location_tag:
        return []

    # Search for location definition
    loc_pattern = rf'#loc(\d+) = loc\("{re.escape(location_tag)}"\)'
    loc_match = re.search(loc_pattern, content)

    if not loc_match:
        return []

    loc_num = loc_match.group(1)

    # Find operations using this location
    op_pattern = rf'%(\d+) = "ttir\.([a-z_]+)"\([^)]*\).*?loc\(#loc{loc_num}\)'
    ttir_ops = re.findall(op_pattern, content)

    return ttir_ops


def extract_operation_details(failing_op):
    """Extract details from the failing operation string."""

    if not failing_op:
        return {}

    details = {}

    # Extract operation name
    op_match = re.search(r'"ttnn\.([a-z_]+)"', failing_op)
    if op_match:
        details['op_name'] = op_match.group(1)

    # Extract tensor shapes
    shape_pattern = r'tensor<([^>]+)>'
    shapes = re.findall(shape_pattern, failing_op)
    if shapes:
        details['shapes'] = shapes

    # Extract compute config
    if 'compute_config' in failing_op:
        config_match = re.search(r'compute_config = #ttnn\.device_compute_kernel_config<([^>]+)>', failing_op)
        if config_match:
            details['compute_config'] = config_match.group(1)

    return details


def generate_report(log_file, output_file=None):
    """Generate a comprehensive failure report."""

    failure_info = find_failing_operation(log_file)

    if not failure_info:
        print("❌ No failure found in log file.", file=sys.stderr)
        return None

    ttir_ops = find_ttir_operations(log_file, failure_info.get('location_tag'))
    op_details = extract_operation_details(failure_info.get('failing_op'))

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("TT-XLA FAILURE ANALYSIS")
    report.append("=" * 80)
    report.append(f"Log file: {log_file}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## Error Summary")
    report.append(f"Line: {failure_info['error_line']}")
    report.append(f"Error: {failure_info['error_msg']}")
    report.append("")

    report.append("## Failing Operation")
    report.append(f"Line: {failure_info['failing_op_line']}")
    report.append(f"Location tag: {failure_info['location_tag']}")

    if op_details.get('op_name'):
        report.append(f"Operation: ttnn.{op_details['op_name']}")

    if op_details.get('shapes'):
        report.append(f"Tensor shapes: {', '.join(op_details['shapes'])}")

    if op_details.get('compute_config'):
        report.append(f"Compute config: {op_details['compute_config']}")

    report.append("")
    report.append("Full operation:")
    if failure_info['failing_op']:
        # Truncate long operations
        op = failure_info['failing_op']
        if len(op) > 500:
            op = op[:500] + "..."
        report.append(op)
    report.append("")

    if ttir_ops:
        report.append("## Corresponding TTIR Operations")
        report.append(f"Found {len(ttir_ops)} TTIR operation(s) with location '{failure_info['location_tag']}':")
        for var_num, op_name in ttir_ops:
            report.append(f"  %{var_num} = ttir.{op_name}")
    else:
        report.append("## TTIR Graph")
        report.append("⚠️  TTIR operations not found in log.")
        report.append("To see TTIR graph, re-run test with:")
        report.append("  export MLIR_ENABLE_DUMP=1")
        report.append("  pytest <your_test> 2>&1 | tee debug.log")

    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"✅ Report written to: {output_file}")
    else:
        print(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TT-XLA test failures from debug logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', help='Path to debug log file')
    parser.add_argument('-o', '--output', help='Output file for analysis report')

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    result = generate_report(args.log_file, args.output)
    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
