# TT-XLA Debug Scripts

Quick reference for analyzing TT-XLA compilation failures.

---

## Available Scripts

### 🔍 show_mlir_modules.py
**Show all MLIR modules in a log**

```bash
python show_mlir_modules.py <log_file>
```

**Output**: Overview of all IR modules (VHLO, StableHLO, TTIR, TTNN) with line ranges and operation counts.

**Use when**: You want to understand the structure of a debug log.

---

### 📊 analyze_failure.py
**Identify failing operations and generate failure report**

```bash
python analyze_failure.py <log_file> [-o report.txt]
```

**Output**:
- Failing TTNN operation
- Error message and location
- MLIR location tags
- Tensor shapes and compute configs
- Corresponding TTIR operations (if available)

**Use when**: You have a failing test and want to know what operation caused the failure.

---

### 📦 extract_mlir_graphs.py
**Extract MLIR graphs from debug log**

```bash
python extract_mlir_graphs.py <log_file> [--type ttir|ttnn|all]
```

**Output**: MLIR graphs (VHLO, StableHLO, TTIR, TTNN) written to /tmp/ with summary table.

**Use when**: You need to extract and analyze IR representations at various compilation stages.

**Features**:
- Extract specific IR types or all types
- Filter by TTNN operation patterns
- Multiple graph support
- Summary table with statistics

---

### 🔄 compare_mlir.py
**Compare two MLIR files for functional equivalence**

```bash
python compare_mlir.py <file1.mlir> <file2.mlir> [--diff]
```

**Output**: Comparison report showing if files are functionally equivalent, with normalized files saved to /tmp.

**Use when**: You need to verify that two MLIR IR files represent the same computation graph, ignoring location attributes.

**Features**:
- Strips location attributes for comparison
- Compares operation types and counts
- Verifies function signatures
- Auto-saves normalized files for manual inspection
- Provides ready-to-use diff commands

---

### ⚙️ print_ttir_to_ttnn_options.py
**Generate ttmlir-opt pipeline options for manual TTIR-to-TTNN conversion**

```bash
python print_ttir_to_ttnn_options.py <system_desc.ttsys> [options]
```

**Output**: Complete ttmlir-opt command with properly formatted pipeline options.

**Use when**: You need to manually run TTIR-to-TTNN conversion with the same options that TT-XLA would use.

**Options**:
- `--bfp8-weights`: Enable experimental BFP8 weight conversion
- `--bfp8-activations`: Enable BFP8 conversion for activations
- `-O [0,1,2]`: Optimization level (0=disabled, 1=optimizer, 2=optimizer+sharding)
- `--math-fidelity [lofi|hifi2|hifi3|hifi4]`: Math fidelity setting (default: hifi4)
- `--no-fp32-dest-acc`: Disable FP32 destination accumulation
- `--trace`: Enable trace optimization
- `--no-const-eval`: Disable constant evaluation
- `--mesh-shape`: Device mesh shape (default: 1,1)

**Example**:
```bash
# With BFP8 weights and optimizer enabled
python print_ttir_to_ttnn_options.py ttrt-artifacts/system_desc.ttsys --bfp8-weights -O1

# Output:
# ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=... experimental-bfp8-weights=true ..." \
#   -o output.mlir input.mlir
```

---

### 🐳 find_docker_tags.py
**Find all registry tags pointing to the same digest as a local Docker image**

```bash
python find_docker_tags.py <image>
```

**Output**: All registry tags (e.g. `dt-*`) that resolve to the same manifest digest as the locally pulled image.

**Use when**: You want to know which `dt-` tag (or any other tag) corresponds to a Docker image you have locally, since `docker inspect` only shows the tag you pulled with (e.g. `latest`).

**Example**:
```bash
$ python find_docker_tags.py ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-24-04:latest

Inspecting local image: ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-24-04:latest
Registry:     ghcr.io
Repository:   tenstorrent/tt-xla/tt-xla-ird-ubuntu-24-04
Digest:       sha256:ea054b08...
Checking 31 tags...

Tags pointing to sha256:ea054b08...:
  dt-b5e301d886ca68a24cde226793ed46a2f0e7097ae2ffd688835764bcc46ae5d5
  dt-72057613073a334abadf8709b0793656b030eb15904942b8940f1ff96486ea40
```

**Notes**:
- Works with any OCI-compliant registry that supports anonymous pulls (ghcr.io, Docker Hub, etc.)
- Requires the image to be pulled locally (`docker pull <image>` first)
- Does not work for locally built images (no RepoDigest)

---

### 📥 rsync_remote.py
**Pull a remote directory into the current directory via rsync**

```bash
python rsync_remote.py <remote_path> [--host HOST] [--user USER] [--strip PREFIX] [--dest DIR] [--dry-run] [--no-compress]
```

**Output**: Files synced to `./<remote_path with --strip removed>/` (or `--dest` if given), using `rsync -avhz --partial --info=progress2`.

**Use when**: You need to grab profiler artifacts (`.tracy`, `profile_log_device.csv`, `ops_perf_results_*.csv`, etc.) or any directory off a remote dev machine quickly.

**Defaults**:
- `--host`: `10.32.48.16`
- `--user`: `$USER`
- `--strip`: `~/kmabee_demo/tt-xla-2/` (everything after this becomes the local path)

**Example**:
```bash
# Remote: ~/kmabee_demo/tt-xla-2/tracy_sampler_nongreedy_tt_sampling_pad32_batch2/reports/2026_04_28_18_26_05
# Local:  ./tracy_sampler_nongreedy_tt_sampling_pad32_batch2/reports/2026_04_28_18_26_05/
python rsync_remote.py ~/kmabee_demo/tt-xla-2/tracy_sampler_nongreedy_tt_sampling_pad32_batch2/reports/2026_04_28_18_26_05
```

---

## Typical Workflow

### 1. Run Test with IR Dumps
```bash
export MLIR_ENABLE_DUMP=1
pytest <your_test> -v -s 2>&1 | tee test_debug.log
```

### 2. Get Overview
```bash
python show_mlir_modules.py test_debug.log
```

### 3. Analyze Failure
```bash
python analyze_failure.py test_debug.log
```

### 4. Extract MLIR Graphs (if needed)
```bash
# Extract TTIR (default)
python extract_mlir_graphs.py test_debug.log

# Extract all IR types
python extract_mlir_graphs.py test_debug.log --type all

# Filter by operation
python extract_mlir_graphs.py test_debug.log --filter 'ttnn.rms_norm'
```

### 5. Search in Extracted Graphs
```bash
grep 'multiply.362' /tmp/graph_1_ttir.mlir
grep -c '"ttir\.' /tmp/graph_1_ttir.mlir
```

---

## Quick Examples

### Example 1: Understanding Log Structure
```bash
$ python show_mlir_modules.py test_qwen3_embedding.log

MLIR MODULES IN LOG
Total modules: 6

✓ VHLO            Line     81 -  10985  (10904 lines,     0 ops)
✓ SHLO            Line  10985 -  21164  (10179 lines,  4942 ops)
✓ TTIR            Line  40351 -  50239  ( 9888 lines,  4942 ops)
✓ TTNN            Line  50239 -  63963  (13724 lines,  4319 ops)
```

### Example 2: Finding What Failed
```bash
$ python analyze_failure.py test_qwen3_embedding.log

TT-XLA FAILURE ANALYSIS
==================================================
## Failing Operation
Line: 63690
Location tag: multiply.362
Operation: ttnn.rms_norm
Tensor shapes: 32x4096xf32, 4096xbf16
Compute config: fp32_dest_acc_en=true, packer_l1_acc=true

## Error Summary
Line: 63691
Error: circular buffers grow to 1528960 B which is beyond max L1 size of 1499136 B
```

### Example 3: Extracting MLIR Graphs
```bash
$ python extract_mlir_graphs.py test_qwen3_embedding.log --type all

====================================================================================================
EXTRACTED GRAPHS SUMMARY
====================================================================================================
Graph    IR Type              Lines      Ops   Output File
----------------------------------------------------------------------------------------------------
1        vhlo                 10903    39969   /tmp/graph_1_vhlo.mlir
1        shlo                 10178     5091   /tmp/graph_1_shlo.mlir
1        ttir                  9887     4942   /tmp/graph_1_ttir.mlir
1        ttnn                 13724     4319   /tmp/graph_1_ttnn.mlir
----------------------------------------------------------------------------------------------------
Total: 1 graph(s), 6 IR representation(s), 63,917 operations, 63,877 lines

$ grep 'multiply.362' /tmp/graph_1_ttir.mlir
%144 = "ttir.multiply"(%139, %143) : (tensor<1x32x4096xbf16>, ...) loc(#loc419)
```

---

## Script Features

### Detection Methods

All scripts automatically detect log format:

1. **"MLIR Module" markers** (most reliable)
   - `MLIR Module ttir:`
   - Present in most TT-XLA logs

2. **"IR Dump" markers** (verbose logs)
   - `// -----// IR Dump After ConvertStableHLOToTTIR`
   - Present when high verbosity is enabled

3. **Module scanning** (fallback)
   - Searches for `module @SyncTensorsGraph`
   - Validates by counting TTIR operations

### Error Handling

All scripts provide helpful feedback:
- ✅ Success indicators
- ❌ Error messages
- ℹ️ Information about detection method used
- 💡 Suggestions for fixing issues

---

## Common Issues

### Issue: "No MLIR Module markers found"

**Cause**: IR dumps not enabled in log.

**Solution**:
```bash
export MLIR_ENABLE_DUMP=1
pytest <your_test> -v -s 2>&1 | tee test_debug.log
```

### Issue: "No failure found in log file"

**Cause**: Test passed or didn't reach failure point.

**Solution**: Verify the test actually failed by checking exit code or test output.

### Issue: Extracted graphs seem incomplete

**Cause**: Log was truncated or captured incorrectly.

**Solution**:
- Use `2>&1 | tee` to capture both stdout and stderr
- Ensure test ran to completion

---

## Environment Variables

Useful for controlling log verbosity:

| Variable | Purpose | Values |
|----------|---------|--------|
| `MLIR_ENABLE_DUMP` | Enable IR dumps | `1` (enabled) |
| `TTXLA_LOGGER_LEVEL` | TT-XLA log level | `DEBUG`, `VERBOSE`, `INFO` |
| `TTMLIR_ENABLE_PERF_TRACE` | Performance tracing | `1` (enabled) |

**Recommended for debugging**:
```bash
export MLIR_ENABLE_DUMP=1
export TTXLA_LOGGER_LEVEL=DEBUG
```

**Warning**: Logs can be very large (100K+ lines) with high verbosity.

---

## Help

All scripts support `--help`:

```bash
python show_mlir_modules.py --help
python analyze_failure.py --help
python extract_mlir_graphs.py --help
```

---

## Contributing

To improve these scripts:

1. Test with various log formats
2. Add detection methods for edge cases
3. Improve error messages
4. Add more analysis features

All scripts use Python 3.11+ with only standard library dependencies.
