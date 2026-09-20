# AMD Open-Source Stack — The Full Descent (reproducible)

Everything in this directory was produced from public repos on a stock machine
(Python 3.14, no EDA tools, no Docker, no AMD hardware) with ~100 lines of
version-skew shims. The full chain is verified bit-exact at every hop.

## The pipeline (top to bottom)

```
PyTorch model (Brevitas quantized)
   │  brevitas.export.export_qonnx
   ▼
QONNX IR  (quantization-as-operator; raw weights + Quant/BipolarQuant nodes)
   │  finn.transformation.qonnx.ConvertQONNXtoFINN  (FoldQuantWeights,
   │                                                ConvertQuantActToMultiThreshold)
   ▼
FINN dialect  (INT4 weights folded, activations → MultiThreshold comparators)
   │  finn.transformation.streamline.Streamline  (absorb scale/shift → integer datapath)
   ▼
Streamlined integers  (INT4×INT4→INT32 MACs; binary XNOR/popcount; UINT32 accum)
   │  step_convert_to_hw → partition → specialize_layers → PrepareIP
   ▼
RTL / HLS code  (mvu_vvu_axi.sv + mvu_4sx4u.sv / top_.cpp + thresh.h + hls_syn_.tcl)
   │  memblock.dat = packed weights (the exact bytes for BRAM init)
   ▼
Software chip simulation  (decode memblock → run datapath in numpy)
   │   → reproduces PyTorch to 1e-8 (int4) / 5e-7 (binary), 33/33 stress trials
   ▼
RTL simulation  (Verilator 4.224 + pyverilator fork, FINN's own rtlsim harness)
   │   → cycle-accurate execution of the generated Verilog
   ▼
Stitched FPGA design  (InsertFIFO/InsertDWC/CreateStitchedIP → make_project.tcl)
   │   → Vivado block design; weights baked into BRAM via memstream IPs
   ▼
[Vivado/Vitis HLS — the only step not run: needs Xilinx account + ~100 GB]
```

## Key findings (each verified, not assumed)

1. **The whole FINN toolchain runs outside its Docker** — two shims for
   qonnx-HEAD version skew (see `finn_shims.py`).
2. **memblock.dat layout** is `[output][input]`-major (decoded from FINN's own
   `matrixvectoractivation.py` packing: `W.T → interleave → reshape → flip`).
3. **qonnx Im2Col window order** is `[kh][kw][c]` (NHWC patch flattened).
4. **`RemoveCNVtoFCFlatten` pre-permutes fc weights** to match the raw NHWC
   stream (invisible to correctness, crucial for simulation).
5. **Binary threshold resolves to `pc >= t`** (verified at boundaries 18/19).
6. **Real bugs found**: torch.export omits `kernel_shape` on Conv (breaks FINN
   lowering); empty node names → `module  #(` with no name in generated RTL.
7. **The NPU stack**: MLIR-AIR is NOT the shared substrate — MLIR-AIE is;
   IRON bypasses AIR entirely (`zero` air.* imports).
8. **FastFlowLM** (AMD's 2026 acquisition, now ROCm/FastFlowLM) is explicitly
   *built on IRON* — AMD's open research stack is the production ecosystem.

## Hardware code generated (all on disk)

| Model | Partition | RTL | HLS C++ | Stitched |
|---|---|---|---|---|
| int4 MLP | `MVAU_rtl` (MW=8 MH=2 PE=1 SIMD=1) | `_wrapper.v` + `mvu_4sx4u.sv` | — | FIFO→MVAU→FIFO |
| binary CNN | ConvInpGen_rtl + 2× MVAU_hls | `_impl.sv` (swg) + `dwc.sv` | `top_.cpp` (`Recast<XnorMul>`) | FIFO/DWC chain, 2 memstream IPs |

Analytical estimates: int4 MLP = 16 cycles, 1 BRAM + 1 DSP; binary CNN =
0 DSPs (pure LUT popcount), 4864 1-bit MACs.

## Reproduce

```bash
# venv (torch cpu, qonnx, brevitas, finn-on-path via finn_shims)
python3 -m venv demo-venv && demo-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
demo-venv/bin/pip install -e qonnx -e brevitas onnx onnxruntime pyverilator tqdm
sudo apt install verilator python3-tk bison flex autoconf help2man perl-doc libfl-dev
# (build Verilator 4.224 from source — pyverilator requires it)
# patch venv: pyverilator --std=c++14, &-strip, tclutil; install maltanar/pyverilator util/axi_utils
# see finn_shims.py + session notes for the exact patches

demo-venv/bin/python demo/demo_export.py      # Brevitas → QONNX
demo-venv/bin/python demo/finn_streamline.py  # step 1 + streamline (bit-exact)
demo-venv/bin/python demo/hw_codegen.py       # RTL/HLS codegen  (env FINN_BUILD_DIR, FINN_ROOT)
demo-venv/bin/python demo/chip_sim2.py        # chip simulation from memblock bytes
demo-venv/bin/python demo/stress_test.py      # 33 adversarial trials
demo-venv/bin/python demo/stitched_ip.py      # full FPGA block design
demo-venv/bin/python demo/finn_rtlsim.py      # Verilator RTL execution
```

## Files

- `demo/*.py` — pipeline scripts (each self-contained, reproducible)
- `demo/hw_codegen/`, `demo/chip_work/`, `demo/rtlsim_work/`, `demo/stitch_work/` — artifacts
- `demo/finn_shims.py` — the version-skew bridge (also pyverilator stub/real toggle)
- `ANALYSIS.md` — the 14-repo ecosystem analysis (Aug 2026 snapshot)
- `fastflowlm/` — ROCm/FastFlowLM clone (the IRON-based production runtime)
- `herd_placement_report.md` — how air-place-herds maps herds to physical tiles

## INT8 GEMM on the XDNA2 NPU — and the group_id root cause (2026-08-31)

### The bug that faked "wrong results at specific shapes"
- Every xclbin-path kernel (AXPY, GEMM) on this stack returned stale memory:
  ERT command reported COMPLETED, but the NPU never touched the data BOs.
- Root cause: amdxdna binds each host BO to its kernel argument slot via the
  BO **group_id** (opcode=slot0, instr=1, ninstr=2, host buffers from slot 3).
  XRTTensor creates BOs with group_id=0 → bound to the opcode slot → silent no-op.
- mlir-aie's own C++ tests (`test/npu-xrt/many_buffers/test.cpp`) create every
  BO with `kernel.group_id(3 + arg_index)` — the python runtime never did.
- Earlier "GEMM wrong at 512³/1024³, ok at 1024×512×1024" and the "2048³ crash"
  were this bug + stale BO memory, not shape-dependent kernel bugs.
- Fix: patched `iron-venv/.../hostruntime/xrtruntime/hostruntime.py`
  `XRTHostRuntime.run()` — for the xclbin path, recreate each tensor's transport
  BO in `kernel.group_id(3+i)`, copy host bytes, refresh tensor views
  (`transport._group_id` cached; pyxrt here lacks bo.group_id()/flags()).
  Also added `import numpy as np` (file had none).

### INT8 GEMM results (patched IRON GEMM, dtype_in="i8" -> int32, exact)
- Patches to `iron/operators/gemm/`: microkernel_mac_dim_map gains i8 (8,8,8)
  and i16 (4,4,8); dtype_in/out made repr=True (they were repr=False, so bf16
  and i8 builds COLLIDED on the same artifact filenames); kernel flags now pick
  -Di8_i32_ONLY / -Di8_i16_ONLY / -Di8_i8_ONLY; tile validation enforces
  m%(2r), k%s, n%(2t) per dtype.
- Results (bit-exact vs numpy int32, 0 bad elements):
  - 1024×512×1024: 0.332 ms → 3.24 TOPS
  - 2048³:           2.095 ms → 8.20 TOPS
  - 4096³:           (compile ~10+ min, run TBD)
- bf16 2048³ now also runs (was "broken"): 6.47 ms → 2.65 TFLOPS; residual
  error is bf16-accumulation noise on near-zero elements, not a bug.
- Peak is likely DMA/latency-bound; 8.2 TOPS ≈ 32 tiles × 128 MAC/cyc × 2 × 1GHz.

## INT8 GEMM correction: group_id, device wedging, and the refined fix (2026-08-31, part 2)

### The full story (corrected)
- The "stale outputs" at session start were TWO overlapping things:
  1. The 2048^3 INT8 GEMM at group-0 BOs **wedges the NPU** (IO_PAGE_FAULTs,
     aie2_set_cmd_timeout) — large-shape DMA with wrong BO groups corrupts the
     device until a PCI remove/rescan.
  2. After a wedge, every xclbin-path op returns stale/zero data (kernel
     "completes" but never touches the BOs), which faked the earlier
     "wrong at specific shapes" and "AXPY broken" reports.
- amdxdna kernel ABI (kernels.json, XML in AXLF): args opcode(0)/instr(1)/
  ninstr(2)/bo0..boN(3+). `kernel.group_id(3+i)` is the memory group each data
  BO must be allocated in. Group-0 BOs work for kernels declaring 0x10000
  (GEMM family) at small shapes, fail at large shapes (2048^3, 4096^3), and
  never work for the AXPY kernel (declares 0x20000).
- Llama prefill ops declare groups 0x90000..0xD0000 — creating BOs in those
  groups yields garbage (invalid pools), but those kernels work fine with
  group-0 BOs. A blanket "recreate BOs in kernel.group_id(3+i)" broke llama.
- REFINED FIX (final, in venv hostruntime.py XRTHostRuntime.run): rebind data
  BOs to kernel.group_id(3+i) ONLY when that group is 0x10000 or 0x20000
  (valid host groups); leave everything else at group-0. Result:
  - INT8 GEMM 2048^3: EXACT, 6.98-8.20 TOPS (npu_time), no wedging.
  - INT8 GEMM 1024x512x1024 / 4096^3: EXACT.
  - Llama 3.2 1B NPU: coherent, 7.85 tok/s, TTFT ~2.1s.
  - AXPY: 1 bf16 ulp.
- Remaining instability: running many DIFFERENT xclbins in one process
  (context-cache eviction on npu2) can degrade later contexts to garbage.
  One-op-per-process is reliable. Upstream-worthy bug reports:
  (a) INT8 large-shape GEMM at group-0 corrupts device (should EINVAL not wedge);
  (b) context eviction degrades subsequent kernels on this driver.

### INT8 GEMM final numbers (bit-exact vs numpy int32, 0 bad elements)
- 1024x512x1024: ~0.33 ms -> 3.2 TOPS
- 2048^3: 2.46 ms -> 7.0 TOPS (max seen 8.20)
- 4096^3: 17.5 ms -> 7.84 TOPS
- Kernel is pipeline-bound: objdump shows ~16 vmac vs ~100 nops + 24 vlda +
  17 vst per loop; peak ~40 TOPS needs hand-tuned double-buffered kernels.

## INT8 GEMM kernel tuning: software-pipelined k-loop (2026-08-31, part 3)

### Bottleneck analysis (2048^3 INT8, 32 cores, ~1 GHz)
- Measured ~200-220 ops/cycle/core = ~7-8.2 TOPS total. Theoretical: 8x8x8
  int8 vmac = 512 MACs/cycle/tile at peak -> ~40+ TOPS; the marketing 50 TOPS
  assumes 512 MACs/cyc x ~1.5 GHz.
- Original kernel disassembly: k-loop ~18.7 cyc per (block,k-iter) for 4 vmacs
  + 4 loads; ~30 nops; 28 vector loads vs 12 vmacs in the whole body.
- The k-loop had NO software pipelining: loads for step i+1 were issued only
  after the step-i MACs, leaving load->vmac latency exposed as nops.

### What was tried (all bit-exact, 2048^3 INT8, 5 runs/process)
- Pristine 2x2 (baseline):            mean ~7.6 TOPS (7.54/7.71)
- Manual software-pipelined k-loop:   mean ~8.0-8.3 (7.71-8.27), best 8.59  <- KEPT
  (prefetch next k-step's A0/A1/B0/B1 into ping registers while mac'ing the
  current ones; epilogue macs the last tile)
- 2x4 register-blocked variant (8 accumulators, 6 loads : 8 MACs, halved C
  traffic): mean ~7.2 - REGRESSION (register pressure + longer j-loop), reverted
- OPT_PERF_ENABLED (chess_flatten_loop on j/k loops): regression (~7.4), off
- tile_n=128 not tried: C tile 32KB + double-buffer would overflow L1

### Final state
- aie_kernels/aie2p/mm.cc: 2x2 mmul with software-pipelined k-loop (shared
  template -> all dtypes). Verified: INT8 1024x512x1024 / 2048^3 / 4096^3 all
  bit-exact (4096^3: 17.4 ms, 7.90 TOPS); bf16 GEMM rel-err ~0.02 (accumulation
  noise); Llama 3.2 1B 7.78 tok/s coherent.
- Conclusion: kernel-only tuning saturates at ~8 TOPS. The 50 TOPS-class rate
  needs a design-level rewrite (larger per-core output tiles, B/A staged in
  L1 across the whole k-block with double-buffered ObjectFifos, C accumulation
  in L1) - out of scope for a kernel edit; the IRON design is a functional
  reference, exactly as documented.
- Process-to-process variance is significant (7.7-8.6 TOPS for the same
  binary): SoC clock/thermal state; npu_time includes the ERT command window.

## Bug filing + hunt continuation (2026-08-31, part 4, post-reboot)

### Bugs filed (gh, account bong-water-water-bong)
- Xilinx/mlir-aie#3653 - "XRTHostRuntime binds host BOs with group_id=0,
  mismatching kernel-declared per-arg groups -> silent garbage, NPU wedge at
  large shapes". Repro: INT8 GEMM with group-0 BOs -> wrong results, ERT
  COMPLETED; kernel.group_id(3..5) = 0x10000 (GEMM) / 0x20000 (AXPY);
  C++ tests create BOs with kernel.group_id(3+p*2), python runtime never does.
  Large-shape variant produced IO_PAGE_FAULT + aie2_set_cmd_timeout (dmesg)
  and wedged the NPU until PCI remove/rescan (observed pre-reboot, state-dep).
  Local workaround: rebind transport BOs to kernel group for 0x10000/0x20000.
- Xilinx/mlir-aie#3654 - "3rd distinct xclbin hw_context in one process
  silently produces garbage". Repro: 3 GEMM ops with distinct shapes; ctx0/1
  exact, ctx2 1,047,753/1,048,576 wrong, deterministic (3/3) when each op's
  XRTTensors are created between kernel runs; all pass if tensors created
  upfront (1/1). Each XRTTensor opens its own pyxrt.device(0) handle, so the
  failing pattern interleaves new device handles + BOs with live multi-context
  execution. Related: #3594. CRITICAL IMPACT: corrupts the IRON Llama prefill
  -> generation is nondeterministic with a fixed seed (verified: 2 runs, same
  seed, different text; CPU ref differs too). The "Llama on NPU works"
  claim from earlier sessions is REVISED: decode (single fused ELF) executes
  correctly, but prefill (many distinct xclbins) silently corrupts, so output
  was the model free-running on garbage KV.
- amd/IRON#171 - "GEMM: no INT8 support despite aie2p mm.cc i8 kernels;
  k-loop not software-pipelined". Lists the 4 code changes for i8 support
  (mac dim map, repr=True dtype naming, kernel flags, tile validation) + the
  pipelined k-loop; verified bit-exact numbers.

### Hunt results (post-reboot, clean device)
- Pipelined 2x2 kernel (kept): 2048^3 INT8 best 8.69 / mean 8.49 TOPS
  (runs_ms 1.98-2.10), bit-exact. Alternatives all regressed and reverted:
  2x4-blocked (7.2), 2x4+pipe (7.82), OPT_PERF_ENABLED (7.4), tile_k=128
  (L1 overflow).
- Column scaling: 8 cols -> 8.71 TOPS total (272 Gops/s/core); 4 cols ->
  4.91 TOPS (307/core). Linear in cores; kernel-bound, not array-DMA-bound.
  (One anomalous 1206 Gops/s/core reading at cols=4 was a measurement
  glitch/boost-clock spike, not reproducible.)
- Kernel runs at ~250-270 ops/cycle/core vs ~1024 theoretical peak; remaining
  gap is load traffic (32 loads : 16 vmacs in body) + vmac issue rate; the
  8-accumulator register-blocked variants spill under Peano scheduling.

## ALL ISSUES FILED (2026-08-31, part 5)

| # | Repo / Issue | Title | Evidence |
|---|---|---|---|
| 1 | Xilinx/mlir-aie#3653 | Host BO group_id=0 vs kernel-declared groups -> silent garbage, NPU wedge at large shapes | INT8 GEMM wrong results; dmesg IO_PAGE_FAULT + aie2_set_cmd_timeout; C++ tests bind kernel.group_id(3+arg) |
| 2 | Xilinx/mlir-aie#3654 | 3rd distinct xclbin hw_context in one process silently garbage (ERT COMPLETED) | 3 GEMMs: ctx0/1 exact, ctx2 1,047,753 wrong, deterministic 3/3 with interleaved tensor creation; 16/16 OK if tensors upfront |
| 3 | Xilinx/mlir-aie#3655 | Llama-3.2-1B NPU prefill nondeterministic: same seed, different text every run | 3 runs, fixed seed, 3 different continuations; individual ops all deterministic; bf16 GEMM bit-identical |
| 4 | amd/IRON#171 | GEMM no INT8 support despite aie2p i8 kernels; k-loop not software-pipelined | 4 code changes enumerated; bit-exact verified; pipelined k-loop +5-8% |
| 5 | Xilinx/finn#1681 | mvu_4sx4u.sv part-select p3[52:21] out of range [47:0] for ACCU_WIDTH>27 | Source analysis: Stage-#4 loop i=0..3, lane3 LO_WIDTH=ACCU_WIDTH, OFFSETS[3]=21; SIMD=1/PE=2 on Vivado 2025.2/2026.1 |
| 6 | amd/xdna-driver#1645 | BO group mismatch silently mis-executes or wedges NPU until PCI rescan | Same dmesg evidence; ask EINVAL + fault quarantine |
| 7 | ROCm/rocBLAS#1680 | sgemm silently wrong on gfx1151 (Radeon 8060S): max rel err 1.7e4 | Fresh repro: 256^3 FP32 GEMM, C=[217.7,334.8,-1137.5,...] vs ref; library HAS gfx1151 hsaco |

Repro files: /home/bcloud/bug-reports/issue-*.md ; rocBLAS test: /tmp/rocblas_test.cpp

## Llama prefill corruption + context races (2026-08-31, part 6)

### What was proven this session (all on-device, XRT 2.21.75)

1. **#3654 root-cause narrowed.** 3 distinct i8 GEMMs with XRTTensors created
   BETWEEN runs: ctx C garbage (47.58% on 2048^3) / ctx B garbage (99.94%)
   depending on op order — the kernel whose BOs are created last on fresh
   device handles, racing resident contexts, silently mis-executes (ERT
   COMPLETED). Sharing ONE pyxrt.device(0) handle across tensors fixed the
   standard interleaved repro 3/3, but an ordering variant (3rd op's tensors
   created before 2nd op runs) still corrupts -> handle churn is a trigger,
   not the whole story; fresh BO allocation while >=3 contexts resident is
   the deeper race. Patched venv tensor.py: XRTTensor defaults to a
   process-wide shared device (`_get_shared_device()`).

2. **Llama-3.2-1B prefill is systematically corrupt (not just
   nondeterministic).** Same seed, 3 runs: different text, first-token
   logits 128k/128k elems differ (max ~16.8). NPU-vs-CPU logits: corr 0.32.
   Per-layer bisection (synced to_torch dumps): layer 0 clean; layers 1+
   corrupt ONLY token 0 (BOS), ~5-7 random dims, off by 1-7; tokens 1-6
   clean (max 0.043). Post-attention residual clean -> corruption enters in
   post-norm/FFN/residual of the first row. Dims vary per layer/run -> DMA/
   state race, not fixed arithmetic.

3. **Workarounds that reduce but don't cure:** host sync (to_torch) after
   each layer's FFN residual made 2/3 runs bit-identical; per-layer dumps
   made layers 0-3 deterministic; PCI remove/rescan clears driver error
   state (145 dmesg xdna ERROR lines pre-reset) and the persistent
   ELF-hw_context EINVAL — but the app stays nondeterministic.

4. **Flaky ELF hw_context EINVAL:** `CREATE_HWCTX err=-22` loading the fused
   decode ELF after 13 resident prefill contexts — sometimes 8/8 retries
   fail, PCI reset restores it. Patched sequence.py: plain retry with fresh
   pyxrt.elf per attempt (eviction is NOT viable — it invalidates handles
   held by live callables -> `CachedXRTKernelHandle has no attribute xclbin`).

5. **Buffers:** prefill = 13 distinct xclbin contexts (~15 ops incl. decode
   ELF), ~480 context switches per 32-layer prefill; NPU_CONTEXT_CACHE_SIZE
   npu2=32 vs driver's ~16+ slots (TDR at Context ID 15 observed).

### Repro/experiment scripts (session artifacts)

- `npu-debug-3654/repro_3654_{interleaved,upfront}_i8.py` — exact i8 repro
- `npu-debug-3654/exp1_shared_device.py` (fixed by handle sharing),
  `exp2_early_tensors.py` (still races), `exp3_forensics.py`, `exp4_debug.py`
- `npu-debug-3654/probe_contexts.py` (24 contexts OK), `probe_elf.py` (ELF ctx OK)
- `npu-debug-3654/diag_bf16_error.py` (bf16/bfp16 noise profile)
- llama_npu.py / llama_cpu.py: env-gated dumps `LLAMA_LOGITS_DUMP`,
  `LLAMA_LAYER_DUMP`, `LLAMA_ATTN_DUMP`, `LLAMA_FFN_DUMP` (all DEBUG-only)

## Llama prefill deep-dive part 2 — rebind root cause + kernel isolation (2026-08-31, part 7)

### NONDETERMINISM ROOT CAUSE FOUND AND FIXED

`XRTHostRuntime.run()` group_id rebind copies `transport.host_bytes` into a
fresh BO when a kernel declares a host group (0x10000/0x20000) different from
the tensor's current one. For tensors last WRITTEN BY AN NPU OP (x, x_norm,
attn_output, ffn_output), the host copy is STALE (never synced) — the rebind
pushed stale bytes to the device, clobbering the correct device data. x
ping-pongs 0x10000 (rms_norm) <-> 0x20000 (residual_add) EVERY LAYER, so x's
device content got reset to the stale embedding repeatedly -> run-to-run
variance (same seed, different text).

FIX (venv hostruntime.py): `transport.from_device(0, transport.nbytes)`
before the copy in the rebind, so the new BO starts from CURRENT device data.
Also added bind-once (a tensor keeps its first group forever; no ping-pong).
Result: llama runs are now BIT-IDENTICAL (same seed -> same logits, 0/128256
differ; corr A-B = 1.0). This is the real fix for the "nondeterministic"
half of mlir-aie#3655.

### KERNELS ARE NUMERICALLY SOUND (isolated tests, bf16-faithful refs)

- gate GEMM (M2048 K2048 N8192, cols8, bf16): max 0.094 vs
  bf16-inputs/fp32-accum/bf16-out reference; 0 bad >0.125.
- eltwise mul (2048x8192): -624 vs fp32-ref -622.6 is CORRECT bf16 rounding
  (ULP=4 at 622); mul is exact.
- down GEMM (M2048 K8192 N2048): 4 bad at row 0 dims 400/698/1159/2029,
  each exactly 1 ULP (216 vs 217 etc.) = fp32 summation-order difference,
  not corruption.
- CPU bf16 vs CPU fp32 logits: corr 0.9998, max 0.28 -> bf16 precision is
  NOT fragile; the NPU's corr 0.32 vs CPU is a REAL systematic error.

### REMAINING OPEN: deterministic systematic error in the app data flow

Post-fix runs are deterministic but still corr 0.32 vs CPU. The error
profile: row 0 (BOS token) of the residual stream, ~5-13 dims/layer, off by
1-7 (up to 6 bf16 ULP at outlier dims with values 200-600), entering around
the FFN stage, growing through 32 layers into logits (max diff 17.8). Since
all kernels are exact in isolation and bf16 is not fragile, the ops are
receiving subtly-wrong DEVICE data — a BO/coherence/DMA bookkeeping bug in
the runtime for buffers reused across kernels with different group
declarations. The to_torch() read path also disagrees with op-consumed data
by 1-2 ULP for rebound buffers, so intermediate dumps cannot be fully
trusted until the read path is fixed. NEXT: (a) instrument per-op BO handles
vs tensor BOs to find the first op reading a stale/aliased BO, or (b) fuse
the 13 prefill contexts into 1-2 OperatorSequences (the decode ELF path is
already clean) to eliminate the multi-context/BO-churn entirely.

### Session artifacts

- venv hostruntime.py: rebind sync-before-copy + bind-once (XRT_REBIND_LOG=1
  to trace); tensor.py shared device handle; sequence.py ELF-hw_context retry
  (re-applied after repo rebase).
- npu-debug-3654/: isolated_gemm_test.py, isolated_down_gemm.py,
  isolated_mul_test.py (kernel isolation), verify_*.py (dump truth checks),
  repro/exp/probe scripts from part 6.
- llama_npu.py/llama_cpu.py: env-gated LLAMA_LOGITS_DUMP + LLAMA_FP32 (CPU
  fp32 fragility baseline) — re-applied after the repo rebase (user committed
  the INT8 GEMM work as 6410550 "gemm: software-pipelined k-loop + validated
  INT8 support" and rebased onto devel, wiping uncommitted debug edits).

## Prefill fusion experiment (2026-08-31, part 8) — fused FFN ELF

### Built: single-ELF fused FFN (OperatorSequence)

runlist: gate GEMM -> up GEMM -> SiLU -> eltwise mul -> down GEMM ->
residual_add; SHARED weight buffers (W_ffn_gate/up/down) rewritten per layer;
sequence invoked once per layer. Scripts: npu-debug-3654/fused_ffn_test.py,
fused_dbg{,2}.py, fused_warmup.py (build_elf_ffn build dir).

### Result: correct WHEN it executes, intermittent outlier-channel miss

- With a warm-up call, runs 0 and 1: max diff 0.125 vs CPU (1 bf16 ULP,
  ZERO bad) — the fused FFN matches the CPU reference exactly.
- Run 2 of 3: the FFN contribution MISSING at the model's outlier channels
  (dims 636, 735, 762, 841, 894, 1002, 1314, 1334, 1476, 1503, 1619, 1703,
  2037 — the channels where the down-projection output is large, |~2-24|):
  fused output == x_attn (ffn_out read as 0 there).
- The SAME dims have appeared as corruption sites in every earlier
  manifestation (xnorm dump diffs, app xpost row-0 diffs, fused test) —
  the model's large-magnitude channels are where the driver/ELF execution
  intermittently loses data. Not a kernel bug (isolated kernels exact), not
  bf16 fragility (CPU bf16 vs fp32 corr 0.9998), not app logic (fused chain
  is CPU-exact when it runs).

### Implication for the fusion plan

Fusing prefill contexts into ELFs does NOT by itself remove the corruption —
the single-ELF execution has the same intermittent outlier-channel miss.
The corruption is in the driver's ELF/xclbin execution layer (data mover /
context state), triggered by the model's large-magnitude channels, at a low
but nonzero rate. Upstream: file the outlier-channel DMA drop as its own
bug with the fused-FFN repro (deterministic-ish, 1-in-3). Practical
mitigations to try next: verify-after-dispatch (read back ffn_output, retry
the call on mismatch), or quantize/clip the outlier channels, or test the
fused FFN with num_aie_columns=1 (fewer columns -> less DMA state).

## Final outcomes of the 3 follow-ups (2026-08-31, part 9)

1. **Verify-after-dispatch guard (double-run-and-compare)** — fused_guard.py:
   fresh inputs per dispatch, require two consecutive dispatches to agree
   (retry up to 5x). Result: 8/10 iterations correct, 2/10 still wrong —
   when the corruption rate spikes, pairs keep disagreeing (or consecutive
   corrupt dispatches agree on wrong data). Partial mitigation only; NOT a
   robust fix.

2. **num_aie_columns=1** — BLOCKED at compile: "Too many simultaneously
   active buffer descriptors on tile (0,0), which supports up to 16" — the
   fused 6-op FFN cannot place its DMA through one shim column. Negative
   result: the high-magnitude miss is not DMA-descriptor-count pressure.

3. **Upstream issue filed**: Xilinx/mlir-aie#3656 "Single-ELF (fused
   OperatorSequence) dispatch intermittently returns stale data at
   high-magnitude channels; ERT reports COMPLETED" — with fused_warmup.py
   repro attached in a comment, ruled-out list (kernel isolation, bf16
   baseline, rebind fixes), and cross-link to #3655. Issue body + repro at
   npu-debug-3654/issue-outlier-drop.md.

4. **Bonus finding**: a synthetic small-shape repro (M=256/K=256,1024/
   N=512,1024, 8 cols) does NOT reproduce the intermittent big-channel miss;
   instead it shows a CONSISTENT ~1% of output elements zeroed at scattered
   positions on every dispatch (repro_outlier_drop.py, repro_diag.py). Noted
   in #3656; possibly the same data-mover root cause, deterministic at small
   sizes.

Net: the driver/ELF execution layer has a data-mover reliability bug at
high-magnitude payloads (intermittent, ~1-in-3 on llama FFN; consistent at
small shapes). Kernels, bf16 precision, app logic, and runtime rebind
handling are all exonerated by isolation. Path forward is upstream
(mlir-aie/driver) or workarounds (multi-run agreement is insufficient;
try clipping/quantizing outlier channels, or a per-layer verify+recompute
against a cheap invariant).

## Magnitude-dependence discovered (2026-08-31, part 10) — the corruption is a VALUE-MAGNITUDE bug

### The break: scaling the fused FFN inputs shows the drop rate is monotonic in output magnitude

Small synthetic fused FFN (M=256/K=256,1024/N=512,1024, 8 cols, bf16, same
runlist as the llama FFN), inputs scaled uniformly:

| max |out| | elements wrong (>1.0) |
|---|---|---|
| 0.49 | 0.00% |
| 1.59 | 0.00% |
| 10.3 | 1.92% |
| 41.0 | 23.01% |
| 181 | 54.71% |
| 1112 | 81.04% |
| 3920 | 90.49% |

Larger payload values -> higher corruption probability. Below |~2| it is
clean; the llama model's FFN channels (|200-600|) sit deep in the danger zone
(hence the ~1-in-3 miss and the corr-0.32 logits).

### Localization (all read-backs bf16-faithful or self-consistent)

- ffn_up (GEMM output) exact at |107| -> GEMM compute + L3 write fine.
- ffn_hidden (mul output) wrong/zeroed at 12-47% of large positions
  (run-dependent) -> loss at/after the mul's input path inside the ELF.
- Every op isolated (standalone xclbin, L3 path) exact at the same
  magnitudes, incl. the mul at |ref|~1000+.
- emulate_bf16_mmul_with_bfp16=False: no change.
- Not bf16 fragility (CPU bf16~fp32 0.9998), not kernel arithmetic
  (isolation), not rebind/BO handling (venv fixes in place).

Conclusion: the NPU's bf16 data path (internal/on-chip; also seen in the
separate-context app at the same channels) intermittently corrupts payloads
with a probability that grows with the VALUE magnitude. ERT reports
COMPLETED. Repro scripts: mag_curve.py, localize_loss.py, hidden_selfcheck.py,
repro_outlier_drop.py (all small, no weights). Filed/updated upstream:
Xilinx/mlir-aie#3656.

## Driver follow-up (2026-08-31, part 11)

- Filed amd/xdna-driver#1646 (completed dispatches intermittently return
  wrong data; rate grows with payload magnitude) with the synthetic
  per-run-variance repro (20/20 unique outputs from identical dispatches at
  scale 2 and 5) + llama fused-FFN evidence, cross-referenced to mlir-aie#3656.
- CORRECTED #3656: the earlier fp32-reference magnitude table was partly
  rounding-noise amplification; the clean evidence is per-run variance
  (identical dispatches -> different results every time, rate scaling with
  payload magnitude). Added correction comment.
- Driver code dig (xdna-driver devel): data coherence is firmware-driven —
  aie2_sync_bo() sends MSG_OP_SYNC_BO via mailbox; completion =
  firmware AIE2_STATUS_SUCCESS -> ERT_CMD_STATE_COMPLETED -> dma_fence_signal.
  The driver has NO data barrier of its own and cannot see data correctness.
  Hypothesis filed: firmware race between command/shim-DMA commit and the
  completion/SYNC_BO response, with a data-dependent timing window (bf16/
  bfp16 conversion paths). Code-path notes posted as a comment on #1646.

## CORRECTION — the fused-FFN "corruption" was in-place buffer aliasing (2026-08-31, part 12)

User challenge ("are you positive it was a bug and not just undiscovered
code?") prompted two decisive checks:

1. **Input-stability check** (input_stability.py): device inputs verified
   bit-identical before every dispatch (x and x_norm, all runs) — yet
   outputs varied run-to-run. Rules out harness input bleed.
2. **Out-of-place residual** (no_inplace.py): same ELF, same kernels, same
   inputs, but `(residual_add, "x", "ffn_output", "x_out")` instead of
   in-place `... "x")`: **15/15 dispatches bit-identical, 0 wrong elements**.

CONCLUSION: the "magnitude-dependent corruption" was an **in-place (aliased)
buffer bug in the mlir-aie full-ELF (fused) dispatch** — a runlist op using
the same named buffer as input and output intermittently produces
stale-read errors; the absolute error scales with the payload values, so the
"magnitude curve" was the detection threshold. NOT a hardware/firmware data
path bug. The separate-xclbin path with the same in-place add is clean
(separate_inplace.py: 0 bad). The decode app's fused runlist uses the same
in-place pattern — consistent with the decode text variance.

Actions:
- mlir-aie#3656 root-cause comment posted (in-place aliasing in fused
  dispatch; recommend distinct read/write views or rejecting the pattern).
- amd/xdna-driver#1646 CLOSED (completed) with a walk-back: driver/hardware
  exonerated; completion-path analysis documented but not implicated.

STILL OPEN (not this bug, deterministic post-fix): the llama app's remaining
corr-0.32 vs CPU. Candidate now: systematic round-toward-zero bias in the
NPU's fp32->bf16 conversions at the outlier channels (observed NPU values
consistently SMALLER than CPU: 211 vs 217, -206 vs -210, ...), amplified
over 32 layers. Next experiment: check the AIE GEMM kernels' rounding mode
(`aie::set_rounding`) vs the CPU's round-to-nearest, and compare NPU vs CPU
at a single layer with rounding mode forced to conv_even everywhere.

## Rounding/precision investigation (2026-08-31, part 13) — status: app error still open

Checked the three precision suspects for the llama app's remaining
deterministic corr-0.32 vs CPU:

1. ROUNDING MODE: aie2p/mm.cc defaults to `aie::rounding_mode::floor` unless
   -DROUND_CONV_EVEN (op default True; artifacts built with conv_even).
   Empirical bit-match test inconclusive (fp32 accumulation noise vs the
   bf16 conversion; match rates 2-8% for all modes, 50/50 direction). Not
   the app's error (observed diffs are 2-7 ULP — too big for 1-ULP rounding).

2. ACCUMULATION ORDER: chunked (64/256) fp32 vs sequential fp32 differ by
   max 0.0002 at K=8192 (order is irrelevant at these magnitudes). NOT the
   error.

3. BFP16 EMULATION (default emulate_bf16_mmul_with_bfp16=True): the
   isolated down GEMM's 4x1-ULP deviations at dims 400/698/1159/2029
   DISAPPEAR with bfp16=False (max 0.25, 0 bad). REAL effect — but the app
   rebuilt with bfp16=False (cache cleared, GEMM kernels recompiled) still
   gives corr 0.32. So bfp16 is not the app's error either.

With the bfp16-off build, the app's FFN chain at layer 1 vs CPU:
- gate/up GEMM outputs: max 0.31/0.40, 0 bad (>0.5) — but 0.3-0.4 at
  |43| is ~1% (6 ULP), larger than bf16 noise — a small upstream error.
- hidden (mul output): 1 element off by 12 at row 0 (outlier product).
- output (down GEMM): 5 bad, row 0 (amplified).

OPEN: the ~1% gate/up deviations originate upstream of the GEMMs — suspect
the app's RMSNorm (post-norm) or the runtime read path (dumps vs op-consumed
data). Next experiment: isolate the app's exact RMSNorm op (weighted=True,
cols=8, size=2048*2048) with known x and W vs a bf16-faithful reference.

Also on file: llama_npu.py GEMMs now set emulate_bf16_mmul_with_bfp16=False
(validated improvement in isolation; did not move the app's corr).

## RMSNorm isolation + Newton refinement (2026-08-31, part 14) — all precision suspects exonerated

- Isolated the app-exact RMSNorm (size=2048*2048, cols=8, num_channels=1,
  tile=2048, weighted=True) vs a bf16-faithful torch reference: max rel
  error 0.98% at |ref|>=0.5 (mostly bf16 output rounding; ~2x at a few
  elements). Candidate seed, but:
- Patched aie2p/rms_norm.cc with one Newton-Raphson refinement on
  aie::invsqrt (hardware rsqrt approx) and REBUILT (kernel .o mtime
  confirmed new): isolation result byte-identical (max rel 0.9804%), and
  the app (RMSNorm artifacts + kernel .o cleared and rebuilt) still gives
  corr 0.320968 — identical to the bfp16-off run. The invsqrt is NOT the
  error. Kernel change reverted.

FINAL STATE of the llama app's remaining deterministic corr-0.32 vs CPU:
- Survives: rebind sync fix, bfp16 emulation off (rebuilt kernels), Newton
  rsqrt, rounding mode (conv_even already), accumulation order (0.0002).
- All ops isolated near-exact (RMSNorm 1%, GEMMs 0.1-0.3%, mul exact, down
  GEMM clean with bfp16 off).
- The ~1% deterministic deviation is in the APP'S DATA FLOW BETWEEN OPS
  (the runtime BO/read path feeding ops slightly-wrong device data at the
  outlier channels), compounding through the FFN over 32 layers.
- Next step (unstarted): per-op BO-handle instrumentation — verify the BO
  each kernel arg receives matches the tensor's current BO and content at
  every dispatch; that is where the ~1% per-op discrepancy must live.

## Deviation chain FULLY MAPPED (2026-08-31, part 15) — seed = post-attention x

Corrected a long-standing instrumentation bug: the app's x_norm dumps were
capturing the PRE-norm (W_norm1, attention input) instead of the POST-norm
(W_norm2, FFN input) — the same mislabel that produced the earlier "xnorm
artifact" confusion. With the correct post-norm dump, the chain is now
self-consistent at every step (bfp16-off build, rebind-per-group+sync):

1. app post-attn x vs CPU: max 0.1875 (~0.8% at |23.75| outlier dims),
   0 bad>0.5 — THE SEED (source within the attention path).
2. app post-norm vs rms_norm(app x): max 0.0409 — RMSNorm EXACT on its input.
3. isolated gate GEMM (bfp16 off) on the app's post-norm vs app's gate:
   BIT-IDENTICAL (max 0.0000) — the gate consumed the dumped data exactly;
   data flow and the GEMM are exonerated.
4. gate deviation (15.25 vs CPU) = the post-norm's ~0.2 error amplified by
   the FFN's outlier columns (W_gate/W_down large entries at dims 400/698/
   1159/2029/etc. make those outputs large and sensitive).
5. mul/down/residual: exact; the deviations compound over 32 layers to
   logits corr 0.32.

Also ruled out along the way: arg push (to("npu")) skip — no change; BO
group binding (bind-once vs rebind-per-group) — no change.

OPEN (the seed's source): the ~0.8% deviation in the post-attention x lives
inside the attention path — RoPE (trig precision), softmax (exp
approximation), the per-head score/context GEMMs, or the attention residual
ordering. Next: isolate the attention chain (RoPE + softmax + scores) with
the app-exact config vs bf16-faithful references; the K=7 score/context
GEMMs and the RoPE trig tables are the prime suspects for a >0.2% error at
values |5-24|.

## Attention-path hunt + final verified state (2026-08-31, part 16)

Isolated/verified the remaining path with clean (correctly-placed) dumps:

- The attention's softmax/context/output run on CPU torch in this app; the
  NPU side is q/k/v GEMMs + RoPE + per-head score GEMM + attn_scale mul.
- All fixes in place (rebind sync, bfp16 off, add/mul/rope/silu conv_even),
  clean post-FFN x comparison vs trusted CPU refs:
  - layer 0: max 0.125, 0 bad — CLEAN (bf16 rounding).
  - layer 1: 4 elements off by 2-3 (~1%) at dims 400/698/1159/2029 — the
    model's outlier columns; consistent with bf16 accumulation noise
    amplified by the outlier weights (K=8192 dot products at |200-600|).
  - First generated token now MATCHES CPU ("King Leontes") — was "Leir"
    variants before the fixes.
- Model fragility: single 2-ULP perturb at layer 0 -> corr 0.9998;
  13 dims x 2 ULP -> 0.9996; every-layer 2 ULP x 13 dims -> 0.9988. The
  model robustly absorbs 2-ULP-scale noise. (1%-scale and later-layer
  perturbation hooks broke on indentation/placement bugs and were removed —
  that part of the fragility curve is unmeasured, honestly noted.)

Conclusion (strong evidence, one caveat): the llama app's remaining
corr-0.32 vs CPU is the accumulation of 1-2 ULP bf16 implementation
divergence at the model's outlier channels over 32 layers — every stage is
verified within bf16 tolerance of the correct result and the data flow is
exact. The one unmeasured link is whether 1%-per-layer x perturbations
reproduce corr 0.32 (the fragility curve at the NPU's actual error scale) —
recommended as a clean follow-up with a properly-placed hook.

Kernel rounding hygiene fixes applied (aie_kernels/generic/{add,mul}.cc,
aie_kernels/generic/rope.cc, aie_kernels/aie2{,p}/silu.cc): set
aie::rounding_mode::conv_even at entry (previously inherited floor from the
prior kernel, biasing large values ~0.7%). Verified in isolation: the add's
bias disappears (max 0.25 -> 0.0000). These are legitimate improvements
regardless of the app-level outcome.

## Kernel build (2026-08-31, part 17)

Built the modified kernels explicitly with the Peano toolchain
(compile_cxx_core_function, llvm-aie clang++):
- kernel-build/{add,mul,rope,silu_aie2,silu_aie2p}.o — all OK, text symbols
  verified (add 7, mul 8, rope 1, silu 6/4).
- Llama app build dir artifacts rebuilt from the fixed sources: add/mul/
  silu/rope .o at 18:32-18:33 (post 18:29-18:31 source edits); rms_norm.o
  rebuilt from the reverted (non-Newton) source for consistency.
- Functional verification: isolated add vs exact bf16 sum max 0.0000
  (bias 0.7% -> 0); full app runs, first token matches CPU ("King Leontes"),
  logits corr 0.318 (unchanged class — the rounding fixes are correct but
  not the corr-0.32 driver, which remains bf16-accumulation divergence at
  the outlier channels).

## RESOLUTION — the corr-0.32 was a CPU-reference measurement artifact (2026-08-31, part 18)

ROOT CAUSE OF THE ENTIRE "llama app corr 0.32 vs CPU" SAGA: the CPU
reference's logits dump fired on EVERY forward pass (prefill AND each decode
token) and the DECODE pass overwrote logits_cpu.npy with decode logits.
The NPU's dump is prefill-only. With the CPU dump guarded to prefill
(logits.shape[1] > 1):

- NPU prefill logits vs TRUE CPU prefill logits: **corr 0.9974, max 0.99,
  mean 0.16** — the NPU is CORRECT (bf16 tolerance).
- The old "0.32" reference vs the true reference: corr 0.32 — it was decode
  logits all along.
- Self-consistency: full-head(NPU final x) vs NPU logits = 0.9999; the CPU
  app's own logits vs its own final x through the same head = 0.32 (the
  inconsistency that exposed the polluted reference).

The fragility-curve work (Task 1) was chasing a phantom — there was no 0.32
bug. The model's real sensitivity (2% at the 4 outlier dims -> corr 0.71) is
an interesting robustness fact but not implicated.

DECODE: with the fixed build, 2/3 runs identical ("King Leontes and
Camillo's palace."), 1/3 differs ("...and Polixene"). Prefill deterministic
and correct; decode still varies ~1-in-3 — the fused-decode in-place
aliasing family or the KV-cache read-back path (to_torch on NPU-written
buffers). THIS is what the aliasing fix (Task 2) targets.

## FIXED: in-place aliasing in the fused dispatch (2026-08-31, part 19)

SequenceFullELFCallable._sync_inputs() only pushed the INPUT consolidated
BO. An in-place runlist buffer (name in input_args AND output_args) is
allocated in the OUTPUT arena, so get_buffer("x").torch_view() marked the
OUTPUT BO dirty — never pushed to the device. x stayed stale/racy -> the
intermittent "missing FFN contribution" in fused runlists and the decode
text variance. Fix: push dirty ranges on ALL THREE consolidated buffers
(input/output/scratch) before each dispatch.

Verified:
- in-place fused FFN (was 3/10 unique outputs, bad runs): **20/20
  bit-identical, 0 wrong**.
- llama decode: **3/3 identical** ("King Leontes and Camillo's palace.").
- prefill logits vs true CPU reference: **corr 0.9974** (bf16 tolerance).
- The corr-0.32 saga: CPU reference was polluted (its logits dump fired on
  the decode pass too and overwrote the prefill logits) — measurement
  artifact, not an NPU bug.

FINAL STATE: llama-3.2-1B prefill correct (corr 0.997), decode
deterministic, all kernels verified within bf16 tolerance. Real fixes made:
rebind sync-before-copy, bfp16-off GEMMs, conv_even rounding in
add/mul/rope/silu, fused-dispatch _sync_inputs all-three-buffers.

## Validation + regression test (2026-08-31, part 20)

- 48-token generation x2: FULL outputs bit-identical; 7.86 tok/s decode,
  3.09s TTFT. End-to-end deterministic.
- Operator test suite (iron-venv + pytest installed):
  - elementwise_add/mul, silu, rms_norm: 27/27 pass.
  - rope, gemm (non-extensive): 17/17 pass.
  - NEW regression test (iron/tests/infrastructure/sequence.py:
    test_fused_inplace_buffer_bit_stable): 10/10 pass — fused in-place
    dispatch is bit-stable across repeated dispatches and matches the
    reference (guards the _sync_inputs fix).
- Changeset (iron checkout): kernels add/mul/rope/silu (conv_even),
  sequence.py (_sync_inputs all-three-buffers + ELF retry), llama app
  (bfp16-off GEMMs + env-gated debug dumps), regression test. Venv:
  tensor.py shared device, hostruntime.py rebind sync.
- Final validated state: llama-3.2-1B prefill corr 0.9974 vs true CPU
  reference (bf16 tolerance), decode fully deterministic, all touched-op
  tests green.

## Forge outcome record — operational (2026-08-31, part 21)

- forge-evidence.py: runs the verification (determinism x2, self-consistency
  via final-residual head projection, correctness vs the clean prefill-only
  CPU reference, per-op exactness cited) and emits forge-outcome-record.json
  (39s end-to-end).
- Real record: correctness corr 0.9974 / max 0.9856 (passed, bar 0.99),
  determinism bit-identical (passed), self-consistency 0.9999 (passed),
  verdict role_completed=true, confidence=high.
- Deliverables in cesarops-updates/: pitch-rewrite.md, technical-update.md,
  forge-outcome-record.md (spec), forge-evidence.py, forge-outcome-record.json.

## Cross-executor + scorecard (2026-08-31, part 22)

- forge-evidence.py --executor cpu: same role measured on the torch-CPU
  reference app; correctness = cross-executor agreement vs the NPU record
  (corr 0.9974); determinism bit-identical; self-consistency ~1.0.
- records/npu.json + records/cpu.json: two schema-v0.1 outcome records for
  the same role, different executors — comparable by construction.
- scorecard.html: self-contained role x executor scorecard (rendered from
  the records; verdict banners, evidence table, pass/fail styling).

## Line of work complete (2026-08-31, part 23)

cesarops-updates/ is the finished, self-contained deliverable set:
- README.md (index of all artifacts)
- pitch-rewrite.md (grounded post), technical-update.md (illustrated update)
- forge-outcome-record.md (v0.1 spec, now pointing at the canonical records)
- forge-evidence.py (executable, --executor npu|cpu)
- records/{npu,cpu}.json (real records, both role_completed/high)
- scorecard.html (rendered scorecard)
- scorecard-repo/ (ready-to-publish layout for cesarops-scorecard:
  README, index.html, records/, SPEC.md, forge-evidence.py)

Full evidence trail: DESCENT.md parts 6-23; upstream mlir-aie #3653-#3656;
driver #1646 closed (walked back). Final technical state: llama-3.2-1B
correct (corr 0.9974) and deterministic on the XDNA2 NPU; three real bugs
fixed with regression coverage.

## Fused prefill FFN (2026-08-31, part 24)

- LLAMA_FUSED_FFN=1: one full-ELF OperatorSequence replaces the six
  per-layer prefill FFN dispatches (gate/up GEMM -> SiLU -> mul -> down
  GEMM -> residual add). Weights + activations round-trip through shared
  fused buffers; the residual read-back reshapes the flat 1-D 'x' subview
  to (max_seq_len, emb_dim).
- Verification (7-token prompt, XDNA2):
  - prefill logits bit-identical to the separate-op path (corr 1.0);
  - run-to-run bit-identical (deterministic);
  - decode text identical ('SCENE I. The King');
  - TTFT 2.62-2.65s vs 2.72s separate (~3% prefill saving).
- Debug dump pitfall: the fused 'ffn_gate' buffer holds silu(gate) by
  dump time (SiLU is in-place in the runlist), so comparing it to the
  raw gate dump shows a fake mismatch -- compare post-silu instead.
- Committed: 76eee2a (llama prefill fused FFN); build_elf_*/ ignored.

## Second role: INT8 GEMM 2048^3 (2026-08-31, part 25)

- forge-evidence.py gained --role gemm-int8: wraps bench_int8_gemm.py,
  parses its stdout (exactness + npu_ms + TOPS), emits a schema-v0.1
  record (records/int8-gemm.json).
- Fresh run: 2048x2048x2048 i8->i32, 64x64x64 tiles, 8 cols, 5 reps:
  exact=True bad=0 max_abs=0; npu_ms=[2.6,2.25,2.22,2.11,2.1];
  TOPS best=8.26 mean=7.72 (AIE event counter, pure compute).
- Scorecard + SPEC.md + README + technical-update.md updated to carry
  role 2 (gemm.int8.2048) alongside role 1 (llama-3.2-1b.prefill).

## Prefill at real prompt length (2026-08-31, part 26)

- Root inefficiency: all prefill ops built at M=max_seq_len (2048)
  regardless of the actual prompt (7 tokens) -- the FFN/attention GEMMs
  computed ~293x more rows than used.
- Fix: AIELlamaOperators/AIELlamaBuffers take prefill_len (defaults to
  prompt_len for back-compat). Prefill ops build at
  prefill_len = ceil(seq_len/512)*512 (GEMM constraints: M%(tile_m*4)==0,
  N%(tile_n*8)==0 for bf16, 8 cols), so a 7-token prompt -> 512. Decode
  ops and KV caches keep max_seq_len (2048).
- Verified (7-token prompt, XDNA2):
  - TTFT 2.72s -> 0.945s (separate ops) / 0.921s (fused FFN) -- ~2.9x;
  - prefill logits bit-identical to the 2048-built run (corr 1.0);
  - corr 0.9963 vs fresh CPU reference at the same prompt length
    (top-1 agrees; decode text identical).
- Committed: 764e7ef.

## Branch close-out + issue sweep (2026-08-31, part 27)

- PR amd/iron#177 (feat/gemm-int8-pipelined) closed as superseded: the
  INT8/pipelined-k-loop work landed on devel in refined form
  (6410550 -> 0f5cc48 -> 5582ca1) plus the llama work built on it.
  Branch deleted locally and on the fork (1bit-MONSTER/iron); devel pushed
  to the fork (5582ca1..764e7ef).
- Issues closed with resolution comments:
  - amd/iron #171 (INT8 support, resolved by 6410550/0f5cc48/5582ca1)
  - amd/iron #173 (dtype artifact-name collision, resolved by 6410550)
  - mlir-aie #3655 (nondeterministic prefill, root cause + fix)
  - mlir-aie #3656 (fused-ELF stale data, root cause + fix in sequence.py)
- Issues filed: amd/iron #180 (kernel floor-rounding bias, fixed in 76eee2a).
- Left open (genuine upstream bugs, only local workarounds): amd/iron #174
  (group_id=0), #175 (context-cache eviction), #176 (INT8 EINVAL wedge);
  mlir-aie #3653, #3654; amd/xdna-driver #1645.
- Working tree clean; devel = 6 commits ahead of amd/iron upstream.

## Asymmetric INT8xINT4 GEMM on AIE2P (2026-09-01, part 28)

- `dtype_b="i4"` added to GEMM: B is packed `(K, N//2)` int8 (two nibbles per
  byte, low nibble first); the kernel uses the AIE2P **4x16x16 mmul** (1024
  MACs/instr vs 512 for the int8xint8 8x8x8). B bytes halved for the same
  logical width.
- The `int4_t` empty-struct `sizeof==1` trap is handled with a `B_ADV`
  pointer-advance correction in `mm.cc` (all manual B arithmetic halves the
  element count for int4).
- `design.py`: packed-B tiling + L2->L1 fifo dims (generated BD is
  `[4,4,16,8]/[512,8,32,1]`; the mac sees `B(kk`,nn)=B4[16kk+kk`,16nn+nn]`).
  `op.py`: `dtype_b` field, `i8_i4_ONLY` kernel flag, `pack_i4` static method,
  packed arg spec. `bench_int8_gemm.py --b-i4` drives it.
- Verified on NPU2 (gfx1151, 8 AIE cols, 64x64x64 tiles), bit-exact vs numpy:
  - i8xi4 2048^3: **9.52 TOPS** (vs i8xi8 8.02, +19%) — and B bytes halved
  - i8xi4 1024^3: 4.27, 512^3: 1.27 (all `exact=True bad=0 max_abs=0`)
- Committed: `0fef28b` (devel, 7 commits ahead of amd/iron upstream; pushed
  to the 1bit-MONSTER/iron fork).
- Natural next steps: symmetric INT4xINT4 (both sides i4, pack both
  operands), and threading i4 weights through the llama pipeline (the
  Q4NX work already produces 4-bit weights — the GEMM side can now consume
  them at half bandwidth).

## Symmetric INT4xINT4 — BLOCKED by the AIE API (2026-09-01, part 29)

Attempted the natural follow-up to part 28 (both operands at 4-bit, 2x MAC
density). **The toolchain has no native int4 x int4 mmul.** Compiled a real
kernel test with the Peano clang (aie2p target) through the full aie_api:

```
error: implicit instantiation of undefined template
  'aie::detail::mmul<4, 16, 16, int4, int4, 32>'
```

- Only `detail/aie2p/mmul_8_4.hpp` references int4 — the asymmetric
  int8A x int4B mode (4x16x16, 1024 MACs), already implemented in part 28.
- No int4xint4 specialization exists in ANY of the installed aie_api trees
  (iron venv, iron-venv, mlir-aie-main) nor in upstream Xilinx/aie-api
  main (checked 2026-09-01). No native i4xi4 vmac intrinsic is exposed
  either (mac_4x16_16x8 is the only 4-bit-capable intrinsic).
- Conclusion: the 4-bit path on AIE2P is **asymmetric-only** (activations
  stay 8-bit, weights go 4-bit) — which is exactly what inference needs
  (weights dominate memory traffic). Symmetric i4xi4 is not implementable
  without AMD shipping an i4xi4 mmul in the API. Documented as a
  toolchain constraint, not a bug in our code.

## W4A8 on real llama weights through i8xi4 (2026-09-01, part 30)

The valuable half of the i4 follow-up: **thread real 4-bit weights through
the NPU GEMM.** `llama_w4a8_validate.py` loads llama-3.2-1B safetensors,
quantizes a real weight to INT4 (packed two-nibbles-per-byte, per-output-
neuron scales), quantizes activations to INT8 (per-tensor scale), runs the
i8xi4 NPU GEMM, and dequantizes with (s_x * s_w).

Verified on NPU2 (M=256, 2048-deep K):

| weight | corr vs bf16 | top-5 match |
|---|---|---|
| q_proj (N=2048) | 0.9885 | 3/5 (top-2 identical) |
| k_proj (N=512)  | 0.989  | 4/5 |
| gate_proj (N=8192) | 0.9902 | 4/5 (top-4 identical) |

- **NPU int math bit-exact** vs the CPU int8 x int4 reference (`exact=True`);
  all error vs bf16 is quantization loss (W4A8 is expected to lose ~1% corr).
- This proves the W4A8 path the llama pipeline needs: weights -> i4 packed
  (the Q4NX format from ws12 is already this layout), activations -> i8
  with a per-tensor scale, one dequant multiply after each GEMM. The
  remaining llama-pipeline work is mechanical (activation quantizers on
  the bf16 tensors, scale ops after each GEMM, KV cache stays bf16).
- Committed: `e48e116` (llama_w4a8_validate.py). Run:
  `PYTHONPATH=/usr/lib/python3/dist-packages ~/amd-oss/iron-venv/bin/python
  llama_w4a8_validate.py [M] [layer] [tensor_key]`.

## Full-model W4A8 llama prefill on XDNA2 (2026-09-01, part 31)

The W4A8 path from part 30 wired end-to-end: **all 16 layers of
llama-3.2-1B prefill run on the NPU with INT4-packed weights** (asymmetric
i8xi4 GEMMs, per-output-neuron weight scales) **and INT8 activations
(per-token scales)**. The NPU does the 7 heavy GEMMs/layer (attn q/k/v/o +
ffn gate/up/down); the host does embed, rmsnorm, RoPE, attention scores/
softmax/value, SiLU, and the tied lm_head.

`llama_w4a8_npu.py`, validated vs the bf16 CPU reference (`llama_cpu.py`):

| prompt | logits corr | top1 | top5 |
|---|---|---|---|
| "The capital of France is" | 0.936 | exact (12366) | 4/5 |
| "What is 2 plus 2?" | 0.937 | exact (3639) | 5/5 |

Full prefill: ~0.8-1.0 s (GEMM-heavy path, host round-trips dominate).

### Three real bugs found and fixed on the way

1. **Shape-keyed op pool overwrote weight bindings.** Sharing one compiled
   GEMM per (K,N) shape means every op of that shape uses the LAST bound
   B weights (q and o both got o's; k and v both got v's). Fix: compile
   once per shape, but one buffer set per weight (`bind()` returns a
   per-weight callable).
2. **XRT first-dispatch readback flake** (the one documented in 5582ca1):
   the first call to a kernel after other kernels ran on the same device
   returned a stale C readback (up GEMM: run1 corr 0.065, run2 0.99).
   Fix: warmup re-run on first use + read `res.npu_time` to force the
   dispatch to complete before `to_torch()`.
3. **Naive per-tensor activation scales are too crude for 16 layers**
   (corr 0.55-0.75, prompt-dependent top1). Per-token (per-row) scales —
   Q8_0-style — recover it (corr 0.94, top1 exact on both prompts).

### Status

- devel = 9 commits ahead of amd/iron upstream (incl. e48e116 W4A8 layer
  validation, ec019eb full-model W4A8), pushed to the 1bit-MONSTER/iron
  fork.
- Remaining for production decode: the decode path uses GEMVs (M=1); an
  i4-weighted GEMV needs the same treatment (or reuse the i8xi4 GEMM with
  M padded). Also: move the host attention math onto the NPU, and
  optionally group-wise weight scales (Q4_K-style) to close the last
  corr gap toward 0.99.

## W4A8 decode with KV cache (2026-09-01, part 32)

KV-cached incremental decode added to `llama_w4a8_npu.py`: per generated
token, the 7 layer GEMMs run on the NPU as i8xi4 with `real_m=1` (row 0 =
the token, padding zeroed; weight bandwidth halved by i4, wasted M compute
sub-ms). Host does rope/attention/SiLU against the bf16 KV cache seeded by
the W4A8 prefill.

Verified vs the bf16 reference (prompt "The capital of France is"):
- first decoded token **exact** (`12366` -> " Paris")
- bf16 ref continues "Paris. Paris is the capital"; W4A8 degrades after the
  first token (" Paris\n\n stations travellers Prior") as quantization error
  compounds through the decode loop (per-token scales drift on the residual
  stream).

The decode mechanism is correct; quality is bounded by the quantization
scheme. Next levers: group-wise (Q4_K-style) weight scales, or keeping the
early layers' residual in bf16, or calibration for the activation scales.

Also fixed in the decode path: per-head attention batching shapes
(`unsqueeze(1)` -> [H, S] scores), KV append dim (`unsqueeze(1)` ->
[G, 1, hd]), and a harness-state mutation gotcha when A/B'ing against the
reference.

Committed: `597f91f` (devel, now 10 ahead of amd/iron upstream, pushed).

## Group-wise i4 scales + error isolation (2026-09-01, part 33)

Closed the W4A8 quality gap with **Q4_K-style group-wise weight scales**:
`W4A8_GROUPS=N` splits the i4 weight K-dim into N chunks, each with its own
per-column scales, run as per-group i8xi4 GEMMs and dequantized per group
(the exact int path is untouched — per-group buffers fixed a stale-C
readback race with the shared-buffer version).

**Error isolation first**: with all-i8 weights (same i8 activations) corr
jumps to **0.9965** vs 0.937 for i4 — so the WEIGHTS are the W4A8 error
bottleneck; the per-token i8 activations are nearly lossless.

Quality ladder (llama-3.2-1B prefill logits corr vs bf16):

| config | corr | notes |
|---|---|---|
| i4, G=1 (per-column) | 0.937 | baseline |
| i4, G=8 | 0.966 | |
| **i4, G=16** | **0.973** | top5 5/5, both prompts top1 exact |
| i4, G=32 | 0.972 | plateau (per-group act noise at K=64) |
| i8 weights | 0.9965 | upper bound for this act scheme |

Also: `W4A8_MIX_LEN=N` keeps the first N layers at i8 weights (cheap
partial precision; 0.937 -> 0.966 at N=8).

Committed: `2d354f7` (devel, 11 ahead of upstream, pushed). The decode
still hits early EOS after 1-2 tokens under W4A8 (EOS logit bias) — the
known remaining artifact; per-token act scales keep drifting in the
decode loop. Next: calibration or higher-precision early layers for
decode, or moving the host attention math onto the NPU.

## Stale-C readback root-caused — W4A8 decode now fluent (2026-09-01, part 34)

**The "early EOS" / gibberish decode was NOT quantization — it was the XRT
readback flake.** Bisected to: whenever the A-buffer content changes since
the previous kernel call, the first run returns STALE C (measured: all-rows
corr 0.01 vs the next call's 0.994). The generated token's row always
carries new A data, so its C read back stale/zero and the whole decode
derailed (the earlier "Paris def" / EOS behavior was this, not loss).

Fix: **always re-run each op and keep the second result** (replaces the
warmup-once heuristic; ~2x dispatches, sub-ms each).

End-to-end result — W4A8 llama-3.2-1B on XDNA2, group-wise i4 (G=16) +
per-token i8 activations:

```
"The capital of France is" ->
  "Paris is the capital of France. The city is located in the north of the
   country and is the largest city in the..."     (fluent, correct)
"What is 2 plus 2?" -> prefill corr 0.973, top1 exact, top5 5/5;
  decode echoes the question (1B-model behavior), no EOS bias
```

Both prompts: prefill top1 exact, KV-cached decode generates coherent text.
The W4A8 pipeline is **end-to-end correct** — decode quality is now bounded
by quantization, not by the runtime race.

Committed: `ab47c82` (devel, 12 ahead of upstream, pushed).

Remaining (research-grade, not correctness): moving the host attention math
onto the NPU for decode speed; calibration to push prefill corr from ~0.97
toward the i8-weight bound (0.9965); the upstream XRT readback race itself
(a fix belongs in the iron runtime — worth an issue).

## Coherence-map write trap root-caused (2026-09-01, part 35)

The real fix for the stale-C bug (part 34 worked around it with a double
call): the iron runtime's `to("npu")` upload is driven by a per-range
**coherence map**, and only host writes the runtime knows about are
transferred. `tensor.numpy()[:] = ...` is an *unmediated* write — the
runtime never marks the range dirty — so after the first dispatch the
buffer stays "on npu" and every later kernel runs on the STALE previous A
(the "stale C" was the correct computation of the wrong input).

Fix: `with tensor.overwrite() as buf: buf[:] = ...` records the write in
the coherence map; the next `to("npu")` uploads it. Verified: the
previously-failing second-call test (all-rows corr 0.01) is now 0.9931
with a SINGLE call — the double-call workaround is removed, halving
dispatches (decode ~2x faster).

Full-model: prefill corr 0.968 top1 exact; KV decode fluent
("Paris is the capital of France...").

Committed: `ce8947c` (devel, 13 ahead of upstream, pushed). Filed
upstream: amd/iron (coherence-map write trap — `.numpy()` writes silently
stale).

The W4A8 llama pipeline is now correct AND fast: the host path is the
remaining cost (attention math + per-op numpy quantize), not the NPU.

## Decode 3.5x speedup — the dequant was the hidden host cost (2026-09-01, part 36)

Profiled the W4A8 decode (was ~4.5 s/token — far too slow). Per big-GEMM
call: op() dispatch+kernel 1.7 ms, to_torch readback 0.1 ms, **dequant
~10 ms** — the `C * sx_full * s_w` multiply over the full [M_PAD, N]
buffer in float64 dominated everything (and ran xG for the group-wise
path: 16 groups x full-N float64 multiply).

Fix: dequant in **float32 on the real rows only** (padded rows are zero;
decode real_m=1 makes the dequant a 256x cut; the returned tensor stays
[M_PAD, N] so the forward's view() shapes hold).

| metric | before | after |
|---|---|---|
| big GEMM call | 12.2 ms | 2.5 ms |
| decode | ~4500 ms/token | **~1280 ms/token (3.5x)** |
| prefill corr | 0.968 | 0.9696 (top1 exact) |
| decode text | fluent | fluent ("...on the river Seine.") |

Committed: `024f27c` (devel, 14 ahead of upstream, pushed).

Remaining host cost: the per-dispatch overhead x 112 ops x G=16 groups
(~0.7 ms each — quantize + XRT + readback). Next levers: reduce G for
decode (quality/speed knob), fuse ops, or batch the group dispatches.

## Quality config ladder + recommended config (2026-09-01, part 37)

Per-group activation scales (W4A8_GROUP_ACTS=1) now work — the earlier 0.69
regression was the stale-A bug, not the math.

Quality ladder (llama-3.2-1B prefill logits corr vs bf16):

| config | corr | notes |
|---|---|---|
| i4 G=16 | 0.973 | top1 exact |
| i4 G=16 + group acts | 0.974 | |
| mix 4 i8 layers + G=16 | 0.970 | top5 5/5 |
| **mix 8 i8 layers + G=16** | **0.977–0.979** | **recommended; top1 exact** |
| all i8 weights | 0.9965 | bound — i4 noise floor ~0.97-0.98 |

Recommended: `W4A8_MIX_LEN=8 W4A8_GROUPS=16` — the first 8 layers keep i8
weights (near-exact), the last 8 use group-wise i4. Verified end-to-end:

```
"The capital of France is" ->
  "Paris is the most visited city in the world. It is a city of culture,
   history, and romance. It is..."   (corr 0.9792, top1 exact, fluent)
```

Committed: `7092037` (devel, 15 ahead of upstream, pushed).

The i4 quantization noise floor (~0.97-0.98) is the limit of this scheme;
closing the last ~0.02 to the i8 bound needs either higher-precision early
layers (already: mix 8) or a smarter weight format (e.g. 4+4-bit split,
or per-row-fp8-scale int4). Decode speed remains the G-knob (G=16: 1.28
s/token; halve G -> ~2x faster at ~0.01 corr).

## Robustness sweep + decode config (2026-09-01, part 38)

Swept the recommended config (W4A8_MIX_LEN=8 W4A8_GROUPS=16) over 4 prompts,
40 generated tokens each — all stable (no NaN, no early-EOS, no crashes):

| prompt | 40-token decode |
|---|---|
| "The capital of France is" | "Paris is the most visited city in the world. It is a city of culture, history, and romance..." |
| "What is 2 plus 2?" | echoes the question (llama-3.2-1B behavior, not W4A8) |
| "The largest planet..." | formats a quiz ("A. Mercury ... Answer: B") — 1B-level knowledge |
| "Write a short haiku..." | restates the instruction (1B behavior) |

Decode: ~890 ms/token consistent (mix8+G16); the early-EOS bug is fully
gone. Output quality tracks the model's own 1B level — W4A8 adds no visible
degradation beyond it.

Decode speed/quality knob (mix8):

| G | ms/token | quality |
|---|---|---|
| 4 | 593 | good |
| 8 | 730 | good |
| 16 | 903 | best |

Added `W4A8_LLAMA.md` (README: repro, config ladder, decode table, bug
list). Committed: `57853b8` (docs). devel now 16 commits ahead of amd/iron
upstream, all pushed.

The thread is at a clean stopping point: correct + robust + documented
end-to-end W4A8 llama on XDNA2 (prefill corr 0.98, top1 exact, fluent KV
decode), three real runtime/kernel bugs root-caused and fixed, upstream
issue filed (amd/iron#181), quality/speed/config all measured and
documented in DESCENT parts 28-38.

## Ops-mix finding + long-context robustness (2026-09-01, part 39)

### Ops-mix: the FFN weights carry the i4 error (the session's big win)

W4A8_OPS_MIX=ffn_i8|attn_i8 isolates which op family keeps i8 weights
(llama-3.2-1B prefill corr vs bf16, all layers):

```
all i4 G16       0.973
attn i8 + FFN i4  0.977   (attention i8 barely helps)
FFN i8 + attn i4  0.992   (FFN i8 is nearly the whole fix)
```

The FFN projections (gate/up/down) are the i4 precision-critical part;
attention (q/k/v/o) tolerates i4 almost free. Attention group curve with
FFN i8: G8 0.9899, G4 0.9889, G2 0.9828, G1 0.9787.

**Recommended: W4A8_OPS_MIX=ffn_i8 W4A8_GROUPS=4** — 19 dispatches/layer
(vs 112 for all-i4 G16, 6x fewer), prefill corr 0.989 top1 exact top5 5/5,
decode 559 ms/token (8x faster than the thread start), fluent:
"Paris is the capital of France. It is the most populous..." / "2 plus 2
is 4".

### Also measured (dead ends, kept as disabled knobs)

- W4A8_ZP=1 asymmetric zero-point i4: 0.971 vs 0.973 — llama weights aren't
  column-biased enough to benefit. Disabled.
- W4A8_GROUP_ACTS=1 per-group activation scales: 0.974 (marginal). Knob kept.

### Long-context robustness (182-token prompt)

- prefill 7.2s = 39 ms/token (dispatch overhead amortizes; scales well)
- decode ~590 ms/token (per-dispatch cost, context-independent)
- KV cache, causal masking, and output coherence all hold at 182 tokens
  (output: "1. The history of France is long and rich..." continuation)

Committed: `680bdb8` (ops-mix), `c520bc3` (README). devel 18 ahead of
upstream, pushed.

## Attention fine-grain isolation — q,k,v want i8, only o tolerates i4 (2026-09-01, part 40)

W4A8_ATTN_I8 knob splits the attention weights (FFN i8, G4):

```
all attn i4          0.989
q,k i8 + v,o i4      0.990
q,k,v i8 + o i4      0.992-0.993   <- final
q,k,o i8 + v i4      0.990
```

The attention splits cleanly by role: q/k/v feed the scores (precision-
critical), the output projection o tolerates i4 at G4.

**Final recommended config:**
`W4A8_OPS_MIX=ffn_i8 W4A8_ATTN_I8=q,k,v W4A8_GROUPS=4`
- prefill corr **0.992-0.993** (vs 0.9965 all-i8 — the remaining gap is
  just o at i4), top1 exact both prompts
- decode 554 ms/token
- "The capital of France is" -> "Paris is the most visited city in the
  world. It is..."

Dead end this round: `NPU_RUNTIME=hrx` — the iron HRX runtime needs a
newer libhrx than installed (`hrx_buffer_map_with_mode` missing);
abandoned, the HRX-vs-XRT dispatch-overhead question stays open.

Quality arc complete: 0.937 (per-column i4) -> 0.973 (G16) -> 0.989
(ffn_i8) -> **0.993** (q/k/v i8) — 0.9965 is the measured all-i8 bound.
Committed: `fba7943` (devel, 20 ahead of upstream, pushed).

## Final performance picture (2026-09-01, part 41)

Locked config (W4A8_OPS_MIX=ffn_i8 W4A8_ATTN_I8=q,k,v W4A8_GROUPS=4) full
measurements, llama-3.2-1B, XDNA2:

| workload | time | per-token |
|---|---|---|
| prefill, 6 tok | ~0.5-0.8 s | dispatch-bound fixed cost |
| prefill, 182 tok | 771 ms | **4 ms/tok** (amortizes) |
| decode (KV-cached) | — | **~530 ms/token** |
| first-run (per process) | +3.5 s | JIT/context init |

Quality: corr 0.992-0.993 vs bf16, top1 exact both prompts, fluent
factually-correct decode.

The W4A8 thread is complete: correct, at the quantization noise floor
(0.993 vs 0.9965 all-i8 bound), decode ~530 ms/token bounded by the XRT
per-dispatch overhead (kernel ~0.3 ms of each ~1.8 ms dispatch). All
knobs measured and documented (W4A8_GROUPS / _MIX_LEN / _GROUP_ACTS /
_OPS_MIX / _ATTN_I8 / _ZP); 20 commits ahead of upstream; DESCENT
parts 28-41.

Open threads for the next research direction:
1. 1bit-engine NPU GEMM — the engine's npu-infer path has documented bugs
   (group_id=0, missing insts; issues #2003/#2002); the i4 GEMM pattern
   proven here could inform its fix.
2. Multi-model W4A8 (qwen3 / zaya) — validates the ffn/attn finding
   generalizes.
3. Decode attention on the NPU (iron MHA op) — removes the last host math.
4. HRX runtime for the dispatch overhead — needs the matching libhrx.
5. KV-cache quantization (W8KV) for long context.
