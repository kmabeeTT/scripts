#!/usr/bin/env python3
"""
Show all MLIR modules in a TT-XLA debug log.

Usage:
    python show_mlir_modules.py <log_file>
"""

import re
import sys
import argparse
from pathlib import Path


def show_mlir_modules(log_file):
    """Show all MLIR modules with their line ranges and operation counts."""

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find MLIR Module markers
    mlir_modules = []
    for i, line in enumerate(lines):
        if 'MLIR Module' in line:
            # Extract module type (vhlo, shlo, ttir, ttnn, etc.)
            match = re.search(r'MLIR Module (\w+):', line)
            if match:
                module_type = match.group(1)
                mlir_modules.append((i + 1, module_type))  # +1 for 1-based line numbers

    if not mlir_modules:
        print("❌ No 'MLIR Module' markers found in log file.", file=sys.stderr)
        print("\nThis log may not have IR dumps enabled.", file=sys.stderr)
        return False

    print("=" * 80)
    print("MLIR MODULES IN LOG")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print(f"Total modules: {len(mlir_modules)}\n")

    # Analyze each module
    for idx, (line_num, module_type) in enumerate(mlir_modules):
        # Determine end of module
        if idx + 1 < len(mlir_modules):
            end_line = mlir_modules[idx + 1][0]
        else:
            end_line = len(lines)

        # Count operations in this module
        module_section = ''.join(lines[line_num:end_line])

        # Try to count operations for each dialect
        if module_type in ['vhlo', 'shlo', 'shlo_frontend', 'shlo_compiler']:
            # StableHLO family uses different patterns
            op_pattern = r'%([\w.]+) = (stablehlo|vhlo)\.'
            ops = re.findall(op_pattern, module_section)
            op_count = len(ops)
        elif module_type == 'ttir':
            op_pattern = r'= "ttir\.[a-z_]+"'
            op_count = len(re.findall(op_pattern, module_section))
        elif module_type == 'ttnn':
            op_pattern = r'= "ttnn\.[a-z_]+"'
            op_count = len(re.findall(op_pattern, module_section))
        else:
            op_count = 0

        line_count = end_line - line_num

        # Format output with color-like markers
        marker = "✓" if op_count > 0 else "·"
        print(f"{marker} {module_type.upper():15} Line {line_num:6} - {end_line:6}  "
              f"({line_count:5} lines, {op_count:5} ops)")

    print("\n" + "=" * 80)
    print("EXTRACTION COMMANDS")
    print("=" * 80)

    # Find TTIR module for extraction command
    ttir_found = False
    for line_num, module_type in mlir_modules:
        if module_type == 'ttir':
            ttir_found = True
            print(f"\nExtract TTIR graph:")
            print(f"  python .github/scripts/extract_ttir_graph.py {log_file} -o ttir.mlir")
            break

    if ttir_found:
        print(f"\nSearch for specific operations:")
        print(f"  grep 'multiply.362' ttir.mlir")
        print(f"  grep -c '\"ttir\\.' ttir.mlir")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Show all MLIR modules in TT-XLA debug log',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', help='Path to debug log file')

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    result = show_mlir_modules(args.log_file)
    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
