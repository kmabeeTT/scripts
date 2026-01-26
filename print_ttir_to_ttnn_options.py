#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper script to construct ttir-to-ttnn-backend-pipeline options string.

This shows what options would be passed to ttmlir-opt for manual compilation.
"""

import argparse


def build_pipeline_options(
    system_desc_path: str,
    experimental_bfp8_weights: bool = False,
    enable_bfp8_conversion: bool = False,
    optimization_level: int = 0,
    math_fidelity: str = "hifi4",
    fp32_dest_acc_en: bool = True,
    enable_trace: bool = False,
    enable_const_eval: bool = True,
    mesh_shape: str = "1,1",
) -> str:
    """
    Build the ttir-to-ttnn-backend-pipeline options string.

    Args:
        system_desc_path: Path to system descriptor .ttsys file
        experimental_bfp8_weights: Enable BFP8 weight conversion
        enable_bfp8_conversion: Enable BFP8 conversion for activations
        optimization_level: 0=no optimization, 1=optimizer enabled, 2=optimizer+sharding
        math_fidelity: lofi, hifi2, hifi3, or hifi4
        fp32_dest_acc_en: Enable FP32 destination accumulation
        enable_trace: Enable trace optimization
        enable_const_eval: Enable constant evaluation
        mesh_shape: Device mesh shape as "rows,cols"

    Returns:
        String with pipeline options for ttmlir-opt
    """
    options = []

    # Core options
    options.append(f"system-desc-path={system_desc_path}")
    options.append(f"optimization-level={optimization_level}")

    # BFP8 options
    if experimental_bfp8_weights:
        options.append("experimental-bfp8-weights=true")
    if enable_bfp8_conversion:
        options.append("enable-bfp8-conversion=true")

    # Compute config
    options.append(f"compute-cfg-math-fidelity={math_fidelity}")
    options.append(f"compute-cfg-fp32-dest-acc-en={str(fp32_dest_acc_en).lower()}")

    # Other options
    if enable_trace:
        options.append("enable-trace=true")
    if enable_const_eval:
        options.append("enable-const-eval=true")

    # Mesh shape
    options.append(f"mesh-shape={mesh_shape}")

    # Optimizer is controlled by optimization_level but can be explicit
    if optimization_level >= 1:
        options.append("enable-optimizer=true")
    if optimization_level >= 2:
        options.append("memory-layout-analysis-enabled=true")

    return " ".join(options)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ttmlir-opt pipeline options string",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Default options
  python print_ttir_to_ttnn_options.py ttrt-artifacts/system_desc.ttsys

  # With BFP8 weights enabled
  python print_ttir_to_ttnn_options.py ttrt-artifacts/system_desc.ttsys --bfp8-weights

  # With optimizer enabled (level 1)
  python print_ttir_to_ttnn_options.py ttrt-artifacts/system_desc.ttsys -O1

  # Full example
  python print_ttir_to_ttnn_options.py ttrt-artifacts/system_desc.ttsys \\
      --bfp8-weights -O1 --math-fidelity hifi4

Then use the output with ttmlir-opt:
  ttmlir-opt --ttir-to-ttnn-backend-pipeline="<output>" input.mlir
        """,
    )

    parser.add_argument("system_desc_path", help="Path to system descriptor .ttsys file")
    parser.add_argument(
        "--bfp8-weights",
        action="store_true",
        help="Enable experimental BFP8 weight conversion",
    )
    parser.add_argument(
        "--bfp8-activations",
        action="store_true",
        help="Enable BFP8 conversion for activations",
    )
    parser.add_argument(
        "-O",
        "--optimization-level",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Optimization level: 0=disabled, 1=optimizer, 2=optimizer+sharding",
    )
    parser.add_argument(
        "--math-fidelity",
        choices=["lofi", "hifi2", "hifi3", "hifi4"],
        default="hifi4",
        help="Math fidelity setting",
    )
    parser.add_argument(
        "--no-fp32-dest-acc",
        action="store_true",
        help="Disable FP32 destination accumulation",
    )
    parser.add_argument("--trace", action="store_true", help="Enable trace optimization")
    parser.add_argument(
        "--no-const-eval", action="store_true", help="Disable constant evaluation"
    )
    parser.add_argument(
        "--mesh-shape", default="1,1", help="Device mesh shape (default: 1,1)"
    )

    args = parser.parse_args()

    options_string = build_pipeline_options(
        system_desc_path=args.system_desc_path,
        experimental_bfp8_weights=args.bfp8_weights,
        enable_bfp8_conversion=args.bfp8_activations,
        optimization_level=args.optimization_level,
        math_fidelity=args.math_fidelity,
        fp32_dest_acc_en=not args.no_fp32_dest_acc,
        enable_trace=args.trace,
        enable_const_eval=not args.no_const_eval,
        mesh_shape=args.mesh_shape,
    )

    print("=" * 80)
    print("TTIR to TTNN Backend Pipeline Options:")
    print("=" * 80)
    print(options_string)
    print()
    print("Full ttmlir-opt command:")
    print("=" * 80)
    print(f'ttmlir-opt --ttir-to-ttnn-backend-pipeline="{options_string}" \\')
    print("  -o output.mlir input.mlir")
    print("=" * 80)


if __name__ == "__main__":
    main()
