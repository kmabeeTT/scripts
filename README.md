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

### 📦 extract_ttir_graph.py
**Extract TTIR graph from debug log**

```bash
python extract_ttir_graph.py <log_file> -o ttir.mlir
```

**Output**: Complete TTIR graph in MLIR format.

**Use when**: You need to analyze the high-level IR representation to understand operation decomposition.

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

### 4. Extract TTIR (if needed)
```bash
python extract_ttir_graph.py test_debug.log -o ttir.mlir
```

### 5. Search for Specific Operations
```bash
grep 'multiply.362' ttir.mlir
grep -c '"ttir\.' ttir.mlir
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

### Example 3: Extracting TTIR
```bash
$ python extract_ttir_graph.py test_qwen3_embedding.log -o ttir.mlir

✓ TTIR module at line 40351 (4942 operations, 9887 lines)
✅ Found TTIR graph (simplified log format)
✅ TTIR graph written to: ttir.mlir

$ grep 'multiply.362' ttir.mlir
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

### Issue: "No TTIR graph found"

**Cause**: IR dumps not enabled in log.

**Solution**:
```bash
export MLIR_ENABLE_DUMP=1
pytest <your_test> -v -s 2>&1 | tee test_debug.log
```

### Issue: "No failure found in log file"

**Cause**: Test passed or didn't reach failure point.

**Solution**: Verify the test actually failed by checking exit code or test output.

### Issue: Extracted TTIR seems incomplete

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
python extract_ttir_graph.py --help
```

---

## Documentation

For more detailed information, see:

- **`../../DEBUG_LOG_ANALYSIS_GUIDE.md`**: Complete workflow guide
- **`../../TTIR_EXTRACTION_SUCCESS.md`**: TTIR extraction examples
- **`../../SCRIPT_IMPROVEMENTS_SUMMARY.md`**: Script update history

---

## Contributing

To improve these scripts:

1. Test with various log formats
2. Add detection methods for edge cases
3. Improve error messages
4. Add more analysis features

All scripts use Python 3.11+ with only standard library dependencies.
