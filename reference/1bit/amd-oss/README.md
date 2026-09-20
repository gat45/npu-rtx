# amd-oss-knowledge

The AMD open-source NPU chain, documented and verified — a curated knowledge
base distilled from the local research workspace `~/amd-oss` (AMD research
repos cloned and analyzed — the NPU chain + Xilinx/AMD toolchain mirrors,
Aug/Sep 2026 snapshot).

**Entry point: [INDEX.md](INDEX.md).**

## What this is

The complete, bit-exact-verified pipeline from a PyTorch model to hardware:

```
Brevitas → QONNX → FINN → RTL/HLS → chip/RTL simulation → stitched FPGA
```

plus the Ryzen AI NPU software-stack mapping (MLIR-AIE / IRON / Triton-XDNA /
FastFlowLM) and the measured receipts behind the 1bit-MONSTER HRX2 llama.cpp
work (zero-copy decode win, prefill regression debate).

## Repository layout

| Path | Contents |
|---|---|
| `INDEX.md` | Curated knowledge index (start here) |
| `DESCENT.md` | The Full Descent — the verified chain, hop by hop |
| `ANALYSIS.md` | Consolidated technical analysis (master table + NPU stack) |
| `structure.txt` | Per-repo structural pass |
| `gh_meta.txt` | Repo metadata snapshot |
| `docs/fastflowlm/` | FastFlowLM docs (AMD's NPU-native runtime, built on IRON) |
| `docs/hrx-loom/` | Loom Programming Guide (ROCm/hrx-system, mirrors rocm.github.io/hrx-system/loom/) |
| `docs/mlir-air/` | MLIR-AIR published docs (Xilinx/mlir-air, mirrors xilinx.github.io/mlir-air/dev/) |

## Canonical sources

- The local workspace `~/amd-oss/` (this repo is a published snapshot of its
  docs; `DESCENT.md`/`ANALYSIS.md` are authored there).
- The 1bit-MONSTER HRX2 research log lives in the
  [1bit-MONSTER](https://github.com/1bit-MONSTER/1bit-MONSTER) repo under
  `research/ws12-hrx-loom/` (round-by-round measured numbers) and
  `docs/research/` (hybrid-prefill-decode, IOMMU PerfOpt).
- FastFlowLM docs under `docs/fastflowlm/` are copied from
  [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) (Apache-2.0).
- Loom guide under `docs/hrx-loom/` is mirrored from
  [ROCm/hrx-system](https://github.com/ROCm/hrx-system) `loom/docs/src`
  (publishes https://rocm.github.io/hrx-system/loom/).
- MLIR-AIR docs under `docs/mlir-air/` are mirrored from
  [Xilinx/mlir-air](https://github.com/Xilinx/mlir-air) `docs/` published
  pages (publishes https://xilinx.github.io/mlir-air/dev/).

## Refresh

`DESCENT.md`/`ANALYSIS.md` are regenerated in `~/amd-oss/`; re-copy them here
and push to refresh this snapshot.
