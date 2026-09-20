# AMD Research Open-Source Projects — Consolidated Technical Analysis

**Source page:** https://www.amd.com/en/corporate/research/open-source.html
**Method:** 14 repos shallow-cloned to `/home/bcloud/amd-oss/` and analyzed (structural pass + 9 deep-dive agent reports, Aug 2026 snapshot). Per-repo artifacts: `structure.txt`, `gh_meta.txt`, this report.

---

## 0. Master table

| Repo | Role | License | HEAD | Status | Stars |
|---|---|---|---|---|---|
| **Brevitas** | PyTorch quantization (PTQ/QAT) | BSD-3 | 2026-08-28 (v0.13.2) | 🔥 very active | 1567 |
| **FINN** | QNN→FPGA dataflow compiler | BSD-3 | 2026-04-14 | 🟡 maintenance-only since ~2024 | 1045 |
| **QONNX** | Quantized ONNX dialect + passes | Apache-2.0 | 2026-02-17 | 🟢 active | 191 |
| **LogicNets** | LUT-based NN methodology | Apache-2.0 | 2024-06-12 | ⚪ dormant (≈2 yrs) | 120 |
| **IRON** | Close-to-metal Python API for Ryzen AI NPUs | Apache-2.0 | 2026-08-29 | 🔥 very active | 135 |
| **Triton-XDNA** | Triton frontend → XDNA NPUs | MIT | 2026-08-29 | 🔥 active | 48 |
| **MLIR-AIR** | Spatial-compile MLIR framework (NPU backend) | MIT | 2026-08-31 | 🔥 very active | 146 |
| **Omnitrace** | CPU/GPU profiler & tracer | MIT | 2025-06-24 | ⚠️ deprecated → `rocprofiler-systems` | 357 |
| **Omnistat** | Cluster-scale job telemetry aggregation | MIT | 2026-08-26 (v1.14) | 🔥 very active | 26 |
| **Astra-Sim** | Distributed ML system simulator | MIT | 2026-03-26 | 🟢 active | 671 |
| **Chakra** | ML workload Execution-Trace standard | Apache-2.0 | 2026-07-27 | 🟢 active (MLCommons) | 192 |
| **AMD PACE** | EPYC-CPU LLM inference engine | MIT | 2026-06-15 (v1.2.0) | 🟢 active | 13 |
| **NPUEval** | LLM eval dataset for AIE kernel codegen | MIT | 2025-11-07 | ⚪ frozen (snapshot) | 32 |
| **AUP AI Tutorials** | University AI courseware (29 notebooks) | MIT | 2026-07-21 | 🔥 very active | 8 |

---

## 1. The AMD NPU software stack (Ryzen AI) — the tightest cluster

The four NPU projects form a layered compiler stack — arguably the most strategically important group on the page:

```
Triton kernels / IRON Python API        (user-facing programming models)
        │
   MLIR-AIR                             (spatial scheduling: tiling, placement,
        │                               buffering, sync — AIR dialect)
   air-to-aie pass + aircc
        │
   MLIR-AIE  ◄── IRON operators built here (separate repo)
        │
   LLVM-AIE / Peano                     (AIE ISA codegen)
        │
   XRT (libxaie) / HSA                  (hardware dispatch)
        │
   AMD Ryzen AI NPU (AIE2 / AIE2P: Phoenix/Hawk Point / Strix, Krackan)
```

### MLIR-AIR (common backend, MIT, HEAD 2026-08-31)
- Spatial-compute compiler mapping structured loop/tensor programs onto AIE arrays; ~18 AIR dialect ops + ~40 transform passes; async token model (`air.channel.put/get`, ping-pong/double-buffering, `npu_dma_packet` packet-switched routing), `air-place-herds`, `air-to-aie`, `aircc` driver, `air-runner` perf simulator.
- ~630 `.mlir` lit/hardware tests, 54 `programming_examples/`, nightly NPU2 LLM perf benchmarks, in-tree `.claude/skills/` (agent skills for LLM deployment — cited in an ISCA'26 MLArchSys paper).
- Pinned to ROCm's LLVM fork (LLVM 24, wheel `24.0.0.2026080106`); also retargets GPUs (`air-to-rocdl`, `airgpu`).

### Triton-XDNA (MIT, HEAD 2026-08-29)
- Experimental end-to-end flow: `@triton.jit` kernels → vendored Triton TTIR → triton-shared (Linalg) → MLIR-AIR Python bindings + Transform-dialect scripts → `aircc` → xclbin/elf/pdi; XRT and HSA dispatch paths; native Windows support.
- Claims compiler-generated matmul (I8/I16/BF16) at parity with handwritten NPU kernels; >90% of tested matmul configs ≥90% of baseline throughput.
- C++ side is a stub — the "backend" is Python orchestration over subprocesses + mlir-air bindings; pins mlir-air wheel via `utils/mlir-air-hash.txt` (36ce5af).

### IRON (Apache-2.0, HEAD 2026-08-29, FCCM 2025 paper arXiv:2504.18430)
- "Close-to-metal" structural Python: ObjectFIFOs, per-column Workers, TensorAccessPatterns, host Runtime sequences over MLIR-AIE bindings (does **not** use AIR). 28 operators (GEMM/MHA/RMSNorm/RoPE/softmax…; bf16), handwritten AIE C++ kernels (`aie_kernels/` aie2/aie2p), a pure-Python artifact-graph build system, ELF operator sequencing, experimental stream-DSE (KU Leuven MICAS) fused-design path, end-to-end Llama 3.2 1B app.
- Very active: daily mlir-aie wheel tracking, CI on Phoenix (NPU1) + Krackan (NPU2) self-hosted runners, tests double as benchmarks (CSVReporter).

**Verdict:** this is AMD's bet that an open, MLIR-based NPU toolchain (à la CUDA's ecosystem pull but compiler-first) plus a low-level Python API will win Ryzen AI developer mindshare. All three are genuinely active and hardware-tested.

---

## 2. The FPGA quantization stack — the mature, slower-moving cluster

```
Brevitas (train/quantize in PyTorch)
   │  QONNX export (qonnx.custom_op.general v2 ops)
   ▼
QONNX (quantized ONNX IR + transformation/analysis pass library + interpreter)
   │
   ▼
FINN (dataflow compiler: QONNX→FINN dialect→HLS/RTL HW graph→bitstream)
   │
   ▼
PYNQ-style bitfile + driver on Zynq/Alveo/Kria
```

### Brevitas (BSD-3, v0.13.2, HEAD 2026-08-28 — 3 releases in 6 weeks)
- Declarative quantizers → dependency-injected proxies → quantized layers (`brevitas.nn`) → `QuantTensor` metadata; FX-based graph PTQ (GPTQ, GPFQ, GPXQ, QRONOS, MAGR, Hadamard), QAT with STE/learned-round, binary/ternary/int/FP8 (E4M3/E5M2)/MX formats, dual ONNX export paths (TorchScript-trace + dynamo/torch.export, default from torch 2.9).
- Wide CI matrix (torch 1.13.1–2.13, py 3.10–3.14), 79 test files, committed multi-version docs HTML. `torch.compile` support is defensive, not first-class; QuantTensor-as-metadata forces dynamo workarounds.

### QONNX (Apache-2.0, HEAD 2026-02-17)
- ONNX dialect adding IntQuant/FloatQuant/BipolarQuant/Trunc (+ MultiThreshold, XnorPopcountMatMul…); arbitrary-precision ints/minifloat/fixed-point/BIPOLAR/TERNARY ride in float32 containers with `quantization_annotation` (Netron-friendly). Lazy per-op registry, ~60 transformations, BOPS cost analysis, `qonnx-exec` CLI.
- Clean layering: interchange + pass library + slow interpreter — hardware mapping deliberately left to FINN/hls4ml. Started by FINN + hls4ml communities.

### FINN (BSD-3, HEAD 2026-04-14, ~5,150 commits since 2019)
- Transformation-based compiler on an ONNX IR: 19-step pipeline from `ConvertQONNXtoFINN` (folds weights, Quant→MultiThreshold, Streamline pass eliminating float ops from the datapath) → `ConvertToHWLayers` (1.7k-line pass) → dataflow partition → folding (PE/SIMD parallelization) → HLS IPgen → FIFO sizing → stitched RTL → Verilator rtlsim → Vivado synthesis → bitfile + PYNQ driver.
- Two backends: mature HLS (Vitis HLS 2022.2 + finn-hlslib) and a hand-written SystemVerilog RTL backend (`finn-rtllib/`, 51 files) to escape HLS costs. Docker-only, Brevitas-only import path; practical precision ~4 bits.
- **Activity cliff:** 580 commits in 2023 → 479 in 2024 → **4 in 2025 → 6 in 2026** (dependabot bumps). Effectively maintenance-only — a completed research framework, not an active product line.

### LogicNets (Apache-2.0, HEAD 2024-06-12 — dormant)
- Opposite philosophy: after training a sparse quantized MLP, each neuron is *exhaustively* reduced to a truth table implemented as a LUT ROM — zero MACs, pure LUT/FF fabric, GHz-class latency (FPL'20; JSC-S: 244 LUTs @ 1353 MHz). ~860 LOC, alpha status, 2 examples (jet substructure, NIDS), no CI. Carried forward by PolyLUT (Edinburgh). No QONNX interop.

**Verdict:** Brevitas is the only still-hot member; QONNX is alive but slow; FINN is a finished body of work; LogicNets is archival. The quant-FPGA era appears to have peaked ~2023–24, with AMD's energy now on NPUs (see §1) and CPUs (see §4).

---

## 3. ROCm-adjacent observability tooling — a tale of two trajectories

### Omnitrace (MIT, v1.13.0, HEAD 2025-06-24) — **being retired**
- Comprehensive profiler (C/C++/Fortran/HIP/OpenCL/Python): Dyninst binary instrumentation, 300 Hz call-stack sampling, roctracer/legacy rocprofiler GPU counters, perfetto traces, causal profiling, hatchet-compatible JSON.
- README states it's being **rebranded to ROCm Systems Profiler** (`ROCm/rocprofiler-systems`) built on rocprofiler-sdk; this repo kept only for ROCm < 6.2. Last commit is a docs dep bump → maintenance mode confirmed.

### Omnistat (MIT, v1.14.0, HEAD 2026-08-26) — **very active**
- Cluster-scale, low-overhead Prometheus-style sampling of per-GPU Instinct telemetry (util, HBM, power, temp, clocks, RAS, XGMI, hw counters via rocprofiler-sdk nanobind extension, kernel tracing), SLURM job demarcation, per-job PDF reports, generated Grafana dashboards, systemd unit. Near-monthly releases (1.7 Aug 2025 → 1.14 Aug 2026); containerized SLURM test cluster; `skills/` playbooks for AI-agent-driven job analysis.
- Explicitly an "open research project, not part of the supported ROCm stack."

**Verdict:** pick Omnitrace only if stuck on ROCm < 6.2; otherwise use `rocprofiler-systems`. Omnistat is the live, actively-shipped one.

---

## 4. LLM inference on AMD CPUs — AMD PACE (MIT, v1.2.0, HEAD 2026-06-15)

**⚠️ Premise correction:** PACE is **CPU-only** — an EPYC (Zen 4+, AVX512-BF16) LLM inference engine, *not* an Instinct/GPU project. No HIP/ROCm/CUDA anywhere.

- PyTorch C++ extension: fused AVX512 kernels (RMSNorm/RoPE/QKV/MLP via oneDNN JIT, libXSMM TPP, AOCL-DLP, FBGEMM), SlabPool KV cache (O(1) slab alloc, L2-aware blocks, GQA decode, sliding window), paged attention port, PARD speculative decoding (up to 5× throughput claimed), BF16/FP32 (+INT8 DLRMv2); Llama/Qwen/Phi/Gemma-3/GPT-J/OPT/GPT-OSS.
- Production serving: FastAPI `pace-server` with router/engine split, NUMA multi-instance, continuous batching, Prometheus metrics, MLPerf Server-scenario harness.
- Flagship v1.2 feature: `pace-vllm` platform plugin — drop-in vLLM 0.21.x CPU replacement shipping its own native lib (`pip install pace-vllm`); force-disables prefix caching; torch.compile MLP fusion pass.
- Honest gaps in roadmap: no LLM quantization/FP8, no tensor/pipeline parallelism, no prefix caching/MLA.

**Verdict:** AMD's answer to vLLM-CPU/llama.cpp-class serving for EPYC. Young (13 stars) but engineering-heavy (56k LOC, wheels, MLPerf harness). Keep an eye on it if you serve LLMs on EPYC.

---

## 5. ML systems co-design — Astra-Sim + Chakra (AMD's role is organizational)

### Chakra (Apache-2.0, MLCommons, HEAD 2026-07-27) — the interchange standard
- Versioned protobuf Execution-Trace (ET) schema capturing compute/comm/memory/**storage/data-ops** (recent LLM-inference additions: checkpoint/KV-transfer); tool suite: converter (PyTorch/Kineto), generator, jsonizer, visualizer, timeline visualizer, trace link, **C++ ET feeder** for native-speed simulators.
- AMD's contribution is co-authorship (MLSys 2026 paper) and the Working Group — no ROCm code in-tree; MI Instinct compatibility rides on the schema being hardware-agnostic. (Doc nit: README says MIT, LICENSE.md is Apache-2.0.)

### Astra-Sim (MIT, Georgia Tech, HEAD 2026-03-26) — the simulator
- **v2.0 reorganized tree** (`common/ system/ workload/ network_frontend/` + 7 extern submodules, incl. a chakra fork with `feeder_v3`); models scheduling + collective communication + compute/memory/network end-to-end; swappable network backends (analytical congestion-aware/unaware, ns-3 packet-level, Broadcom HTSim).
- **AMD's key contribution:** the custom-collectives path replays per-rank Chakra ETs of **MSCCL++-generated** algorithms (`CustomAlgorithm`), i.e., you can simulate MI-Instinct collective schedules generated by MSCCLang before touching hardware.
- Requires `--recurse-submodules`; validated configs for H100/DGX-V100/TPUv3.

**Verdict:** mature, standard-driven co-design ecosystem; AMD is a first-class participant (MSCCL++ modeling) without owning the repos.

---

## 6. Evaluation & education — NPUEval + AUP AI Tutorials

### NPUEval (MIT, HEAD 2025-11-07 — frozen snapshot benchmark)
- 102 AIE-kernel codegen problems (element-wise/activation 46, spatial/linear-algebra 24, reductions 13, bitwise/cast/movement 17) in `dataset/npueval.jsonl`, each with prompt, canonical scalar solution, program wrapper with `event0/1` trace hooks, test vectors, tile/trace metadata.
- Scoring is **on real hardware**: compile (Peano/chess) → MLIR → xclbin → run via XRT → numeric correctness (atol/rtol) + trace-based vectorization score (vpu_cycles/total) + speedup vs scalar baseline (up to ~300×).
- `AIECoder` agent (OpenAI/Anthropic/Ollama) with compile-feedback retries, pass@k-style runs; RAG corpus of ~35 reference kernels (LlamaIndex, 70× cycle improvement shown); Streamlit demo dashboard. Requires Ubuntu 24.04 + Ryzen AI hardware (secure boot off).

### AUP AI Tutorials (MIT, HEAD 2026-07-21 — very active courseware)
- AMD University Program Jupyter-Book: 29 notebooks across get-started (HF/PyTorch inference), fundamentals, train (from scratch), specializing (fine-tuning incl. Llama3), quant (AMD Quark PTQ → ONNX), rag (LangChain/HyDE), **ai-agents (LangGraph/MCP/Pydantic AI — newest area)**, serving (prose-only chapter: Ollama/Llamafile/OpenLLM). ROCm/PyTorch 2.8 docker base; env guides for CPU/GPU/Windows-DirectML; AMD Dev Cloud auto-detection.

**Verdict:** NPUEval measures whether *models* can write NPU kernels; AUP teaches whether *people* can build on AMD hardware. Complementary, both high-quality.

---

## 7. Cross-cutting observations

1. **Strategic pivot is visible in the data:** FPGA-era projects (FINN, LogicNets, QONNX) are flatlining or frozen while NPU-stack projects (MLIR-AIR, IRON, Triton-XDNA) and CPU inference (PACE) are the actively-shipping ones — AMD is consolidating its research OSS around the Ryzen AI NPU and EPYC.
2. **Everything is MIT/BSD/Apache-2.0** and genuinely upstreamable; several projects already merged into ROCm (rocSHMEM) or standard bodies (Chakra, MSCCL++).
3. **Hardware-gated development is the norm:** IRON/Triton-XDNA/MLIR-AIR/NPUEval all require real NPUs + XRT for meaningful work; CI runs on self-hosted Phoenix/Krackan machines. Contributors without hardware are limited to compile-only flows.
4. **Compiler-first strategy:** MLIR-AIR (shared) + MLIR-AIE (per-tile) + LLVM-AIE/Peano is the common backend for *both* the DSL path (Triton-XDNA) and the structural path (IRON) — one substrate, two programming models, mirroring the CUDA/PTX ecosystem play.
5. **Maintenance signals matter:** omnitrace (deprecated), LogicNets (dormant), FINN (maintenance-only), NPUEval (frozen by design) vs. Brevitas/MLIR-AIR/IRON/Omnistat/AUP (actively shipping, weekly/monthly commits).
6. **Small repo, big claims:** AMD PACE (13★) ships a production-grade serving stack; AUP (8★) is a full university curriculum — GitHub stars badly understate AMD's engineering investment in these.

---

*Produced from shallow clones at `/home/bcloud/amd-oss/` (per-repo code inspected by 9 parallel analysis agents; GitHub metadata via API). Snapshot date ~2026-08-31.*
