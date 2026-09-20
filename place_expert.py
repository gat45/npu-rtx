# Étape 2 v3 — Choix placement expert (D2 Adapter extension)
# v3 : ajoute les primitives XRT prouvées par le reverse FLM/1bit (sync, group_id, MoE q41, KV segments)
# Sources : reference/1bit/FLM_SECRETS.md, NPU_GEMM_FIX.md, AMDXDNA_DRIVER_UAPI.md, 1BIT_REVERSE_XDNA2.md

BW_GTT = 56.0e9          # GB/s GTT dma-buf (1bit, Strix Halo) — à re-mesurer Strix Point
BW_RTX = 672e9           # RTX 5070 GDDR7
BW_DDR = 89.6e9          # DDR5-5600 dual
NPU_COLS_MAX = 8         # limite firmware XDNA2 (1bit RE)
NPU_TOPS = 31.0          # TFLOPS pratiques (plafond 8-col, Strix Halo ; Strix Point ~XDNA2_COLS_ACTIVE=4)

# Benchmarks calibrés (1bit performance.md, RAPPORT §3sexies) :
# MoE 35B : FLM NPU 11.66 tok/s decode vs llama.cpp-Vulkan GPU 75.65 tok/s → GPU ~6.5× le NPU
NPU_MOE_DECODE_TPS = 11.66        # FLM MoE sur NPU (@1k ctx)
GPU_MOE_DECODE_TPS = 75.65        # llama.cpp-Vulkan MoE (Strix Halo iGPU ; RTX 5070 encore plus haut)
# → ratio ~6.5 : le NPU est le tier d'overflow/éco, jamais le backend principal MoE

# Primitives XRT prouvées sur le stack FLM/XDNA2 (FLM_SECRETS.md):
# - bytes::sync_to_device()   → xrt::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0)
# - bytes::sync_from_device() → xrt::bo::sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)
# - xrt::bo group_id : opcode=0, instr=1, ninstr=2, host BOs à partir de slot 3
#   (group-0 = no-op silencieux / peut wedger le NPU — NPU_GEMM_FIX.md)
# - runlist : xrt::runlist.add(run) → execute() → wait()  (batch experts d'un token)
# - MoE : send_manual_expert_up_gate_q41() / send_manual_expert_down_gate_q41() (Qwen3.6)

XRT_SYNC_TO_DEVICE = "XCL_BO_SYNC_BO_TO_DEVICE"   # host → NPU (charger expert)
XRT_SYNC_FROM_DEVICE = "XCL_BO_SYNC_BO_FROM_DEVICE"  # NPU → host (récupérer résultat)
XRT_GROUP_INSTR = 1
XRT_GROUP_HOST_BASE = 3

def tile_util(hidden_size):
    util = ((hidden_size // 256) * 256) / hidden_size
    return max(util, 0.25)

def sync_cost(size_bytes, direction):
    """Coût d'un xrt::bo::sync() — primitive de placement expert sur NPU."""
    return size_bytes / BW_GTT  # coût DMA host↔NPU

def expert_load_cost(size_bytes):
    """Chargement expert → tile SRAM via bytes::sync_to_device + group_id."""
    return sync_cost(size_bytes, XRT_SYNC_TO_DEVICE)  # + temps DMA SRAM (runlist)

def cost_ssd_ram_xdna(size_bytes, dma_bw, npu_tops, hidden):
    t_io  = size_bytes / BW_GTT
    t_ram = size_bytes / BW_DDR
    t_dma = size_bytes / dma_bw
    t_npu = (2 * size_bytes * hidden) / (npu_tops * 1e12) / tile_util(hidden)
    return t_io + t_ram + t_dma + t_npu

def place_expert(expert_id, freq_score, size_bytes, hidden_size,
                 gpu_free_vram, npu_cols_free, thermal_state,
                 bw_used, bw_total):
    contention = max(0.0, (bw_used / bw_total) - 0.7) * 2.0

    c_rtx = (size_bytes / BW_RTX) + contention
    c_xdna = cost_ssd_ram_xdna(size_bytes, BW_GTT, NPU_TOPS, hidden_size) * (1 + contention)
    c_cpu = (size_bytes / BW_DDR) * (1 + contention)
    c_ssd = size_bytes / BW_GTT * 1.5

    if gpu_free_vram < size_bytes:
        c_rtx = float("inf")
    if npu_cols_free <= 0:
        c_xdna = float("inf")
    if thermal_state > 70:
        c_rtx = float("inf")

    costs = {"RTX": c_rtx, "XDNA2": c_xdna, "CPU": c_cpu, "SSD": c_ssd}
    return min(costs, key=costs.get)

def planner_route(experts, ctx):
    tiers = {"RTX": [], "XDNA2": [], "RAM": [], "SSD": [], "CPU": []}
    for e in experts:
        tier = place_expert(*e, **ctx)
        tiers[tier].append(e)
    return tiers

# NOTE (reverse FLM, facts durs pour le D2 Planner):
# - Le NPU calcule en INT8/BF16, PAS en Q4 → un "expert Q4" coûte en fait INT8 côté NPU.
#   La conversion Q4NX→INT8 se fait à l'init (upload BO persistant), pas par token.
# - 8 colonnes = capacité compute max. Un expert plus large que les tiles doit être
#   décomposé (col_chunk 256, row_chunk 32) → coût = n_chunks × DMA.
# - KV cache segments (k03/k47/v03/v47) via sync_to/from_device → checkpoint/restore
#   = primitive de "reprendre le contexte" quand un expert change de backend.
# - group_id > 0 OBLIGATOIRE sinon no-op silencieux (leçon NPU_GEMM_FIX.md — écho
#   de la leçon "rc=0 ≠ preuve" de AGENTS.md).