# amd-oss knowledge index

The AMD open-source NPU chain, verified. Read `DESCENT.md` for the full
reproducible descent; `ANALYSIS.md` for the 14-repo landscape; this file is
the index.

## 1. The pipeline (top to bottom, verified bit-exact at every hop)

```
PyTorch model (Brevitas quantized)
   → brevitas.export.export_qonnx
QONNX IR (quantization-as-operator; raw weights + Quant/BipolarQuant nodes)
   → finn.transformation.qonnx.ConvertQONNXtoFINN  (FoldQuantWeights,
     ConvertQuantActToMultiThreshold)
FINN dialect (INT4 weights folded, activations → MultiThreshold comparators)
   → finn.transformation.streamline.Streamline  (absorb scale/shift → integer datapath)
Streamlined integers (INT4×INT4→INT32 MACs; binary XNOR/popcount; UINT32 accum)
   → step_convert_to_hw → partition → specialize_layers → PrepareIP
RTL / HLS code (mvu_vvu_axi.sv + mvu_4sx4u.sv / top_.cpp + thresh.h + hls_syn_.tcl)
   → memblock.dat = packed weights (the exact bytes for BRAM init)
Software chip simulation (decode memblock → run datapath in numpy)
   → reproduces PyTorch to 1e-8 (int4) / 5e-7 (binary), 33/33 stress trials
RTL simulation (Verilator 4.224 + pyverilator fork, FINN's own rtlsim harness)
   → cycle-accurate execution of the generated Verilog
Stitched FPGA design (InsertFIFO/InsertDWC/CreateStitchedIP → make_project.tcl)
   → Vivado block design; weights baked into BRAM via memstream IPs
[Vivado/Vitis HLS — the only step not run: needs Xilinx account + ~100 GB]
```

Key verified findings (DESCENT.md §1–8):

1. The whole FINN toolchain runs outside its Docker (two shims for qonnx-HEAD
   version skew).
2. `memblock.dat` layout is `[output][input]`-major.
3. qonnx Im2Col window order is `[kh][kw][c]`.
4. `RemoveCNVtoFCFlatten` pre-permutes fc weights.
5. Binary threshold resolves to `pc >= t`.
6. Real bugs found: torch.export omits `kernel_shape` on Conv; empty node
   names → `module  #(` with no name in generated RTL.
7. **The NPU stack: MLIR-AIR is NOT the shared substrate — MLIR-AIE is;
   IRON bypasses AIR entirely (zero `air.*` imports).**
8. **FastFlowLM (AMD's 2026 acquisition, now ROCm/FastFlowLM) is explicitly
   built on IRON — AMD's open research stack is the production ecosystem.**

## 2. The 14 repos (ANALYSIS.md master table)

Brevitas (quant, BSD-3, very active) · FINN (QNN→FPGA, BSD-3, maintenance)
· QONNX (quantized ONNX dialect, Apache-2.0, active) · LogicNets (dormant)
· IRON (close-to-metal NPU Python API, Apache-2.0, very active) ·
Triton-XDNA (Triton→XDNA, MIT, active) · MLIR-AIR (spatial-compile MLIR,
MIT, very active) · Omnitrace (deprecated → rocprofiler-systems) ·
Omnistat (cluster telemetry, MIT, active) · Astra-Sim (distributed sim,
MIT) · Chakra (ML trace standard, Apache-2.0) · AMD PACE (EPYC LLM
inference, MIT) · NPUEval (AIE kernel codegen eval, frozen) · AUP AI
Tutorials (29 notebooks).

## 3. FastFlowLM — the production NPU-native runtime

- NPU-first: built exclusively for AMD Ryzen AI NPUs, instant installs,
  context windows up to 256k, xclbins + XDNA driver plugin
  (`libxrt_driver_xdna.so.2`), model-specific libs (llama_npu, qwen_npu).
- HRX (amdxdna) release pinned: `jtuyls/hrx` tag
  `flm-hrx-amdxdna-v2026.07.30` (see `docs/fastflowlm/`).
- The deterministic corr=1.0 path is the fixed-point 1BP pipeline; the
  float pipeline's 32-token corr ≈ 0.99997 is f32 summation-order noise
  amplified by recurrence (~10⁵×), not quantization loss.

## 4. The HRX2 llama.cpp lane (1bit-MONSTER, research/ws12-hrx-loom)

Not in this repo — canonical copies live in the 1bit-MONSTER repo. Summary:

- Q4NX (engine tile format) runs on gfx1151 via the HRX2 llama.cpp fork.
- Round 25i (fork cdb8110): TRUE zero-DMA-copy — decode +10-49% across the
  roster, but pp32 prefill regressed 121.5 → 27.8 (3B).
- The prefill regression attribution ("CPU attention KQ^T/kqv F32 mms write
  into GTT, NPU reads slowly — cache-coherency tax") is the debated claim:
  at pp32 the attention tensors are ~300 KB/layer, the mms were already
  CPU-side at the 121.5 baseline, and round 18/19 measured broad HRX2
  claims as net-negative. The isolation experiment (minimal claims + GTT
  split) is the open question.
- Fast prefill lane is HIP (1227–1313 tok/s); warm decode is HRX
  (~80–87 tok/s). Hybrid design: docs/research/hybrid-prefill-decode.md.

## Files

- `DESCENT.md` — the full descent, reproducible, with hardware code generated
  (int4 MLP: 16 cycles, 1 BRAM + 1 DSP; binary CNN: 0 DSPs, 4864 1-bit MACs).
- `ANALYSIS.md` — master table + the Ryzen AI NPU software stack mapping.
- `structure.txt` / `gh_meta.txt` — per-repo structural and metadata snapshots.
- `docs/fastflowlm/` — FastFlowLM product/runtime/benchmarks/models docs.
- `docs/hrx-loom/` — the Loom Programming Guide (ROCm/hrx-system): the
  source-first compiler HRX uses to specialize kernels (1–15 ms JIT to
  HSACO). Relevant to the HRX2 lane: `workflows/oracles/ggml-llama-cpp.md`,
  `workflows/agent-driven-kernel-development.md`, `guide/facts-and-
  specialization.md`, `integration/jit-kernel.md`, `reference/c-api/`.
- `docs/mlir-air/` — MLIR-AIR (Xilinx/mlir-air): the spatial-compile MLIR
  framework for mapping AI workloads onto AMD NPUs and Versal AI Engine
  arrays. Position: MLIR-AIE is the shared substrate; MLIR-AIR is not it
  (IRON bypasses AIR entirely). Key: `programming_guide.md`,
  `mlir_reference.md`, `AIRComputeModel.md`.



## Xilinx doc mirrors (docs/)

One folder per repo under `docs/`, copied from the local `~/amd-oss` clones by
`sync-knowledge.sh` (selection mirrors the amd-oss MCP manifest):

| Folder | Source repo | Notes |
|---|---|---|
| `vitis-libraries` | Xilinx/Vitis_Libraries | user-guide rst (docs.amd.com Vitis Libraries) |
| `xrt` | Xilinx/XRT | runtime docs toc (xilinx.github.io/XRT) |
| `mlir-aie` | Xilinx/mlir-aie | AIE dialect/pass docs + skills (xilinx.github.io/mlir-aie) |
| `aie-rt` | Xilinx/aie-rt | AIE Run Time driver + FAL |
| `mlir-xten` | Xilinx/mlir-xten | ONNX-MLIR/TOSA streaming-dataflow extensions |
| `onnx-mlir` | Xilinx/onnx-mlir | ONNX→MLIR compiler docs |
| `rapidwright` | Xilinx/RapidWright | place-and-route framework (top md) |
| `llvm-aie` | Xilinx/llvm-aie | AIE LLVM backend incl. PERF_OPT_GUIDE (top md) |
| `mldebugger` | Xilinx/MLDebugger | NPU debug for ML designs |
| `brevitas` | Xilinx/brevitas | quantization in PyTorch (top md) |
| `xilinx-tcl-store` | Xilinx/XilinxTclStore | Vivado Tcl app store |
| `embpf-bootfw-update-tool` | Xilinx/embpf-bootfw-update-tool | embedded boot FW update |
| `xdp` | Xilinx/XDP | XRT debug/profile device platform |
| `vmr` | Xilinx/VMR | Versal Management Runtime firmware |
| `plnx-aie-examples` | Xilinx/plnx-aie-examples | AIE examples for Petalinux |
| `finn-hlslib` | Xilinx/finn-hlslib | Vitis HLS library for FINN |

## AMD research repos (docs/)

The remaining 12 repos of the 14-repo AMD research analysis set (ANALYSIS.md),
each mirrored with a full recursive md/rst walk of the repo root:

| Folder | Source repo | Notes |
|---|---|---|
| `finn` | Xilinx/finn | QONNX→FPGA dataflow compiler |
| `qonnx` | fastmachinelearning/qonnx | Quantized ONNX IR + passes |
| `iron` | Xilinx/iron | close-to-metal Python API for Ryzen AI NPUs |
| `triton-xdna` | AMD/triton-xdna | Triton → XDNA NPU flow |
| `logicnets` | Xilinx/logicnets | LUT-based NN synthesis |
| `amd-pace` | AMD/PACE | EPYC CPU LLM inference engine |
| `astra-sim` | astra-sim/astra-sim | distributed ML system simulator |
| `chakra` | mlcommons/chakra | Execution-Trace standard |
| `aup-ai-tutorials` | AMD/aup-ai-tutorials | university AI courseware |
| `npueval` | Xilinx/npueval | AIE kernel-codegen eval |
| `omnistat` | AMD/omnistat | cluster telemetry aggregation |
| `omnitrace` | AMD/omnitrace | CPU/GPU profiler (deprecated) |
