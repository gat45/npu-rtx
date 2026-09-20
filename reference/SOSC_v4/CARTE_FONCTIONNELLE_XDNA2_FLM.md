# CARTE FONCTIONNELLE COMPLÈTE — XDNA2 + FLM
## Drivers, xclbins, DLLs, APIs, mémoire, temps, flux — tout

> Version définitive — 20/07/2026  
> Sources : 116 rapports .md, 29 JSON, 44 CSV, 200+ scripts, Ghidra 14318 fns, 4 dépôts GitHub AMD

---

## SOMMAIRE

1. [STACK COMPLÈTE — 5 couches de A à Z](#1-stack-complète)
2. [DRIVERS — ipustack.sys, dxgkrnl, D3DKMT, MCDM](#2-drivers)
3. [XCLBINS — 58 overlays, 9 kernels Qwen3.5, format binaire](#3-xclbins)
4. [DLLs — Les 22 DLLs critiques](#4-dlls)
5. [FLM — Serveur HTTP, endpoints, boucle d'inférence](#5-flm)
6. [XRT — API, ABI, timings, structures](#6-xrt)
7. [Q4NX — Format de poids, structure binaire, dequant](#7-q4nx)
8. [GatedDeltaNet — Algorithme SSM Qwen3.5](#8-gateddeltanet)
9. [MÉMOIRE — Budget, KV cache, BW, contention](#9-mémoire)
10. [PERFORMANCE — TPS, TTFT, décomposition temporelle](#10-performance)
11. [OGA / RadeonML / DirectML — Stack alternative](#11-oga)
12. [GHIDRA — Résultats complets du reverse](#12-ghidra)
13. [OPTIMISATIONS — 25 techniques classées](#13-optimisations)
14. [BOTTLENECKS — Les 4 goulots avec preuves](#14-bottlenecks)
15. [DIAGRAMME DE FLUX TEMPOREL — 1 token](#15-flux-temporel)
16. [GLOSSAIRE — Tous les termes](#16-glossaire)

---

## 1. STACK COMPLÈTE (5 couches)

```
COUCHE 5 — APPLICATION (userspace Windows)
┌─────────────────────────────────────────────────────────────────────┐
│ flm.exe (6.47 MB) — Serveur HTTP port 52625                          │
│   API Ollama : /api/generate, /api/tags (PAS /api/chat)             │
│   Paramètres : --pmode performance, --port 52625, --ctx-len 16384    │
│   Sleep patché @ foff=0x389B8D (200ms → 50ms) — NEUTRE (path HTTP)  │
│   Mutex atomique, queue=6                                             │
├─────────────────────────────────────────────────────────────────────┤
│ ogad_v4.py / spec_proxy.py — Daemon / proxy spéculatif               │
│   port 55555 (spec) — draft via llama.cpp + iGPU Vulkan              │
├─────────────────────────────────────────────────────────────────────┤
│ RadeonML (rml_*.dll) — API AMD propriétaire (44 exports)            │
│   rmlCreateDefaultContext → rmlLoadGraphFromFile → rmlInfer          │
│   Utilise wchar_t* sur Windows (PAS char*)                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
COUCHE 4 — DLLs MODÈLE NPU
┌─────────────────────────────────────────────────────────────────────┐
│ qwen3_5vl_npu.dll (3.0 MB, 3526 exports, 14318 fns) — ★ CŒUR       │
│   0x310D0 = decode loop CPU (layer orchestrator, 60ms, 54%)         │
│   0x30010 = gen_layer_seq                                           │
│   0x180042e00 = update_ping_pong_flag (INEFFECTIF, body=10 insns)   │
│   0x180042e10 = wait_DMA_queue_if_full (body=133 insns)             │
│   gen_attention_seq / gen_ffn_seq / gen_ssm_seq — générateurs seq   │
├─────────────────────────────────────────────────────────────────────┤
│ Classes NPU internes (identifiées Ghidra) :                          │
│   npu_cmd (base) → npu_ddr_cmd, npu_dma_block_cmd,                  │
│     npu_issue_token_cmd, npu_preemption_cmd, npu_wait_cmd,          │
│     npu_write_cmd                                                     │
│   npu_sequence — séquence complète d'instructions NPU               │
│   npu_app_manager — gestionnaire d'application NPU                  │
│   npu_tiles — enum des tuiles AIE                                   │
│   ert_cmd_state — machine à états ERT                               │
│   conv3d_patch_embed — convolution patch embedding (AVX512)         │
│   layernorm_high_precision (body=1542 insns)                        │
├─────────────────────────────────────────────────────────────────────┤
│ Autres DLLs modèles (même pattern) :                                 │
│   llama_npu.dll (2.04 MB) — forward @ 0x1FD70                      │
│   phi4_npu.dll, qwen2_npu.dll, qwen3vl_npu.dll, qwen2vl_npu.dll,   │
│   gemma_npu.dll, nanbeige_npu.dll, lfm2_npu.dll, gpt_oss_npu.dll    │
│   whisper_npu.dll, gemma_embedding.dll                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
COUCHE 3 — DLLs GÉNÉRATEURS DE SÉQUENCES AIE
┌─────────────────────────────────────────────────────────────────────┐
│ gemm.dll (162 KB) — ★ Générateur de séquence GEMM                   │
│   ?generate_seq@Gemm@@QEAAXPEAVnpu_sequence@@IIII_NW4Activation_Ty  │
│   Arguments : (npu_sequence*, uint M, N, K, group_size,             │
│                bool is_q4, Activation_Type_t act, uint dtype)       │
│   6 types de commandes npu_cmd produits : ddr, dma_block,           │
│     issue_token, wait, write, preemption                             │
├─────────────────────────────────────────────────────────────────────┤
│ mha.dll — Générateur de séquence Multi-Head Attention               │
│   generate_mha_sequence(npu_sequence*, uint×3, bool, int)           │
├─────────────────────────────────────────────────────────────────────┤
│ dequant.dll (375 KB, 783 exports) — ★ Générateur séquence dequant   │
│   generate_dequant_q4_1_seq(npu_sequence*, uint×3, int)             │
│   Export #685 = THUNK mort (8 B) — Export #686 = vraie fct @ 0xB0F0│
│   3 pré-conditions : count≠0, count&0x1FF==0, n_blks&0x7F==0       │
│   Stack frame : 400 B                                                │
├─────────────────────────────────────────────────────────────────────┤
│ q4_npu_eXpress.dll — Loader SafeTensors + BO manager                │
│   SafeTensors::load_weights() — charge les poids .q4nx en BO XRT   │
│   bo::~bo() — destructeur BO                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Ces DLLs ne font PAS d'appels XRT directs.                          │
│ Elles produisent des npu_sequence* en mémoire (compilateur statique)│
│ Aucun Sleep, aucun XRT — 100% CPU, durée négligeable                │
│ Équivalent au backend MLIR-AIE (air.herd, air.channel, air.token)   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
COUCHE 2 — XRT (RUNTIME)
┌─────────────────────────────────────────────────────────────────────┐
│ xrt_coreutil.dll (193-272 KB selon version, 507 exports C++)        │
│   Proxy actif : v18.1 (153 KB, hook xclLoadXclBin)                 │
│   Proxy complet : v19_slab (257 KB, slab allocator)                 │
│   Proxy final : v24 (272 KB, 507 exports = 11 hooks + 496 forward) │
│   Proxy v25b (502 forwarders) : INACTIF — GetProcAddress NULL       │
│                                                                     │
│   OBJETS EXPORTÉS (comptage reverse) :                              │
│   xrt::*        = 312 objets                                        │
│   xrt_core::*   = 127 objets                                        │
│   xrt::ext::*   = 38 objets                                         │
│   xrt::aie::*   = 18 objets                                         │
│   xrt::profile::* = 12 objets                                       │
│                                                                     │
│   VTABLES : 0 exportées (Pimpl intégral) — 69 candidates dans .rdata│
│     xrt::bo vtable @ 0x20BF88 (29 méthodes)                         │
│                                                                     │
│   STRING CRITIQUE : "KDMA not supported on windows" ×2 dans .rdata  │
│     → #1431 : retourné par xrt::run::start()                       │
│     → #4133 : retourné par runlist::execute()                       │
│                                                                     │
│   DÉCOUVERTE : xrtBOAlloc IGNORE le paramètre size                  │
│     rdx jamais utilisé sur 12 appels analysés                       │
│     Le driver décide la taille d'allocation lui-même                │
├─────────────────────────────────────────────────────────────────────┤
│ xrt_core.dll (592 KB, 0 exports = STUB)                             │
│   WorkloadsSessionHost (3 processus : 1 coordinateur + 2 workers)   │
│     → IpuMcdmDriver (kernel MCDM) → dxgkrnl.sys → NPU              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
COUCHE 1 — DRIVERS + HARDWARE
┌─────────────────────────────────────────────────────────────────────┐
│ ipustack.sys (500 KB) — ★ Driver MCDM Windows                       │
│   DMA scatter-gather — 24.4% BW efficiency                          │
│   1008 instructions LFENCE dans .text                                │
│   19 × MOV ECX, 0x50 (80B) — 19 × MOV ECX, 0x98 (152B)            │
│   PAS de force_cmdlist (contrairement à Linux amdxdna.ko)           │
│   TDR timeout : 2000 ms (aie2_ctx.c)                                │
├─────────────────────────────────────────────────────────────────────┤
│ dxgkrnl.sys — DirectX Graphics Kernel                               │
│   D3DKMTQueryAdapterInfo, D3DKMTEnumAdapters3, D3DKMTCloseAdapter   │
│   Passage obligé pour tout accès NPU Windows                        │
├─────────────────────────────────────────────────────────────────────┤
│ NPU XDNA2 — AMD Ryzen AI 9 365 (Strix Point)                       │
│   PCI 0x17F0 rev 0x10 = NPU4 (XDNA2 mid-range, aie2p = 5 accum.)   │
│   8 colonnes × 4 rangées = 32 tiles AIE2P (confirmé)               │
│     4 colonnes exposées à XRT (partition 0) — 4 réservées firmware │
│     AIE header dans xclbins : 0x00000800 = 8 colonnes               │
│     column_width=8 dans TOUS les xclbins                            │
│   Horloge kernel : 1.8 GHz                                          │
│   51.3 TOPS INT8 peak — 38 TOPS eff (paper AMD) — 9-12 TOPS GEMM   │
│   Mémoire : 2 MB SRAM L1 (64 KB × 32 tiles)                        │
│             4 MB L2 (512 KB × 8 MemTiles)                           │
│             6 MB SRAM total                                         │
│   0.27% GOPS actifs : 138/51,300 — 99.73% idle                      │
│   BW DDR5 : 21.93/89.6 GB/s = 24.5% (96.7% saturé BW)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. DRIVERS

### 2.1 Windows MCDM Stack (ipustack.sys)

```
Windows MCDM (Kernel Mode Display Driver Model) :
  ipustack.sys (500 KB) — propriétaire AMD/Microsoft
    ├── DMA scatter-gather engine
    ├── SHIM interface (SHIM = Streaming Hardware Interface Module)
    ├── Sync Object management (bo::sync)
    ├── Context scheduling (hw_context)
    └── IOCTL dispatcher

COMMENT FLM PASSE PAR MCDM (vs XRT public) :
  FLM → vitis-ai-runtime2.dll → RadeonML_ipu.dll → D3DKMT → ipustack.sys
  ↑ PAS par le scheduler ERT (qui dirait "KDMA not supported on windows")

  Notre XRT public :
  xrt_coreutil → ERT_START_DPU/COPYBO → "KDMA not supported" → TIMEOUT 7s ❌

  La preuve : "KDMA not supported on windows" ×2 dans xrt_coreutil.dll
  #1431 : xrt::run::start()
  #4133 : runlist::execute()

  C'est un #ifdef _WIN32 dans le code source XRT :
    #ifdef __linux__
        kdma_submit();
    #else
        return XRT_ERROR;  // "KDMA not supported on windows"
    #endif
```

### 2.2 Linux amdxdna.ko (xdna-driver)

```
amdxdna.ko (open source, LGPL) — PAS Windows
  11 IOCTLs DRM :
    CREATE_HWCTX, DESTROY_HWCTX, CONFIG_HWCTX, CREATE_BO, GET_BO_INFO,
    SYNC_BO, EXEC_CMD, GET_INFO, SET_STATE, WAIT_CMD, GET_ARRAY

  force_cmdlist = true (DÉFAUT) — batching automatique des commandes
    Windows MCDM n'a PAS cette optimisation — chaque commande = 1 IOCTL séparé

  Opcodes ERT : ERT_START_CU(0), ERT_START_DPU(18), ERT_CMD_CHAIN(19),
    ERT_START_NPU(20), ERT_START_NPU_PREEMPT(21),
    ERT_START_NPU_PREEMPT_ELF(22)

  Format commande DPU :
    struct amdxdna_cmd_start_dpu {
        u64 dtrace_buffer;
        u64 instruction_buffer;
        u32 instruction_buffer_size;
        u16 uc_index;    // microblaze controller index
        u16 chained;     // nombre d'éléments chainés
    };
```

### 2.3 Mailbox Opcodes (30+ opcodes, driver amdxdna + shim)

```
Opcodes mailbox (SHIM DMA) :
  0x02 = CREATE_CONTEXT     → Initialisation HW context
  0x13 = SYNC_BO            → Sync buffer object (67 µs typique)
  0x18 = CHAIN_EXEC_NPU     → ★ Clé P1 : exécution chaînée
  0x20 = FORCE              → Preemption forcée
  0x21 = FINE               → Preemption fine
  0x22 = FRAME_BOUNDARY     → Preemption frame boundary
  0x40 = GET_ARRAY          → Telemetry : stats par colonne
  0x41 = GET_TILE_STATUS    → Telemetry : état tiles
  0x42 = GET_FREQ           → Telemetry : fréquence courante

DMA Protocol :
  12 mots DMA 32-bit, opcode header 0xE80
  Mot 0 : opcode (0xE80)
  Mot 8 : masque de bits = 0xFFFFFC00 (constant)
  Adresses : 48-bit compressées (low 32 + high 16 bits)
  Format TXN : opcode=1 (PMC) ou opcode=3 (TXN AIE2)
  Registres PLL : 0x4E101810 = 1000 MHz (lm_head.dll)
```

---

## 3. XCLBINS

### 3.1 Inventaire complet (58 overlays pour 40 modèles)

```
Configurations de tuilage :
  1×4  : 21 fichiers (petits modèles : LFM2, Embedding-Gemma, Whisper)
  4×4  : 25 fichiers (modèles moyens : Qwen3.5-4B, Llama-3.2)
         dont 4x4_latest = 3.99 MB
  5×4  : 3 fichiers (5x4_latest = 2.22 MB — plus petit = gemm-only)
  8×4×1 : 3 fichiers (preemption_4x8_17f0.xclbin — critique)
  N×4  : 5 fichiers (nombre de colonnes variable)
```

### 3.2 Les 9 xclbins de Qwen3.5-9B

```
| xclbin               | Taille  | Rôle                                 |
|----------------------|---------|--------------------------------------|
| layer.xclbin         | 384 KB  | Transformer layer complet            |
| mm.xclbin            | 490 KB  | Matrix Multiply (GEMM)               |
| attn.xclbin          | 304 KB  | Multi-Head Attention                 |
| dequant.xclbin       | 136 KB  | Q4NX→BF16 (kernel dpu_kernel_id=0x901)|
| lm_head.xclbin       | 262 KB  | Projection vocabulaire (logits)      |
| GateDeltaNet_prefill | 189 KB  | SSM prefill (parallel scan O(1))     |
| conv.xclbin          | 371 KB  | Convolution SSM (CausalConv)         |
| vision_attn.xclbin   | 565 KB  | Attention vision (mode VLM)          |
| vision_mm.xclbin     | 504 KB  | MatMul vision                        |

9 kernels par modèle Qwen3.5 (vs 4 pour Llama/Gemma/DeepSeek)
Kernels uniques à Qwen3.5 : GateDeltaNet_prefill + conv
```

### 3.3 Format binaire xclbin2

```
Clés observées dans les xclbins AMD :
  xilinx_v1_ipu_0_0      — identifiant IPU
  dummy_bitstream        — bitstream dummy (pas de FPGA)
  mem_topology           — topologie mémoire
  vadd.link_build        — lien build

Header AIE : @ +0x08 dans le buffer instruction = 0x00000800 (8 colonnes confirmé)
column_width = 8 dans TOUS les xclbins Qwen3.5

Contenu :
  - Descripteurs DMA SHIM (Buffer Descriptors)
  - Séquences d'instructions VLIW pour AIE vector cores
  - Lock tables inter-tiles (synchronisation)
  - Métadonnées de kernel (dpu_kernel_id, args)
```

### 3.4 Instruction buffer AIE

```
Les instructions AIE VALIDES sont produites par :
  gemm.dll::generate_seq() → npu_sequence*
  mha.dll::generate_mha_sequence() → npu_sequence*

PAS par extraction à offset 0x338 du xclbin (ce sont des métadonnées)

Flags XRT optimaux pour exécution :
  0x17 = HOST | CACHE | P2P → 7.80 decode, 45.9 prefill ★ OPTIMAL
  0x37 = +EXECBUF → +5.4% decode (défaut FLM)
  0x1F = +SVM → -81% prefill (destructif)
```

---

## 4. DLLs — Les 22 DLLs critiques

### 4.1 DLLs de l'inférence NPU

```
C:\Program Files\flm\ — 16 DLLs clés :

flm.exe              6.47 MB  Serveur HTTP, Sleep patché @ 0x389B8D
qwen3_5vl_npu.dll    3.0 MB   ★ Qwen3.5 decode loop (14318 fns, 3526 exports)
llama_npu.dll        2.04 MB  Llama decodage (forward @ 0x1FD70)
dequant.dll          375 KB   Générateur séquence dequant Q4_1
deqUant.dll          653 KB   Proxy/wrapper hook (camelCase)
dequantback.dll      ~400 KB  Backup FLM
gemm.dll             162 KB   Générateur séquence GEMM
mha.dll              ~200 KB  Générateur séquence MHA
q4_npu_eXpress.dll   ~300 KB  Loader SafeTensors + BO manager
xrt_coreutil.dll     193-272 KB  ★ Runtime XRT (507 exports C++)
xrt_core.dll         592 KB   STUB (0 exports) → WorkloadsSessionHost
lm_head.dll          ~300 KB  Projection tête de génération
gemma_embedding.dll  ~200 KB  Embeddings Gemma
whisper_npu.dll      ~300 KB  Whisper ASR NPU
```

### 4.2 DLLs de support

```
abseil_dll.dll       Concurrency/timing (Sleep, SleepConditionVariableSRW)
libcurl.dll          Timeouts HTTP client (Sleep, SleepEx)
libfftw3-3.dll       FFT double précision (Sleep, WaitForSingleObject)
libprotobuf.dll      Sérialisation protobuf
avcodec-61.dll       FFmpeg codecs audio (Whisper)
```

### 4.3 DLLs système AMD

```
RadeonML_ipu.dll      3.7 MB, 44 exports — API officielle Windows pour NPU
                       (rmlCreateDefaultContext, rmlLoadGraphFromFile, rmlInfer...)
vitis-ai-runtime2.dll — Runtime Vitis AI
                       (utilisé par FLM pour contourner l'absence de KDMA)
onnxruntime_providers_vitisai.dll — EP VitisAI pour ONNX Runtime
                       (problème : ABI mismatch avec PyPI ORT 1.22.0)
onnxruntime_providers_ryzenai.dll — EP RyzenAI (CastAvx bug: WinError 1114)
```

### 4.4 DLL loading order critique (pour OGA)

```python
import os, ctypes
# 1. AMD 1.6.1b deployment d'abord (ORT 1.23.2.dev matching)
os.add_dll_directory(r'C:\Program Files\RyzenAI\1.6.1b\deployment')
os.add_dll_directory(r'C:\Program Files\RyzenAI\1.6.1b\onnxruntime\bin')
# 2. amdihk64.dll depuis DriverStore
os.add_dll_directory(r'C:\Windows\System32\DriverStore\FileRepository\...')
# 3. Charger DLLs AMD dans le bon ordre
for dll in ['amdihk64.dll', 'xrt_coreutil.dll', 'dyn_dispatch_core.dll',
            'onnxruntime_vitisai_ep.dll']:
    ctypes.CDLL(dll)
# 4. Import OGA (charge son ORT)
import onnxruntime_genai as og
```

---

## 5. FLM — Serveur, endpoints, boucle d'inférence

### 5.1 Endpoints API

```
Port 52625 — API Ollama (PAS OpenAI)
  ✅ /api/generate       — POST, body JSON {model, prompt, options{num_predict, num_ctx, temperature, stop}}
  ✅ /api/tags           — GET, liste des modèles disponibles
  ❌ /api/chat           — Connection refused
  ❌ /v1/chat/completions — OpenAI vintage (port 52630, obsolète)

Réponse /api/generate inclut dans usage :
  decoding_speed_tps    — TPS decode
  prefill_speed_tps     — TPS prefill
  prefill_duration_ttft — durée prefill (attention: pas le TTFT wall-clock!)
```

### 5.2 Pipeline de requête FLM

```
HTTP Request
    ↓
flm.exe (orchestrator):
  1. Queue mutex (atomique, queue=6)
  2. hw_context::hw_context()    →  charge métadonnées xclbin depuis driver (I/O disque)
  3. kernel::kernel()            →  configure AIE tiles
  4. bo::bo() × N                →  mappe tous les Buffer Objects
  5. runlist::runlist()          →  initialise DMA SHIM locks
                                  ←  overhead cumulé = ~2362ms (H6)
    ↓
qwen3_5vl_npu.dll (0x310D0 decode loop):
  6. Pour chaque layer (32) :
     a. gen_attention_seq / gen_ffn_seq / gen_ssm_seq
        → produit npu_sequence*
     b. bo::sync() WRITE         → poids (~300 KB/layer) → SRAM AIE
     c. runlist::execute()       → dispatch vers AIE cores
     d. runlist::wait()          → bloque jusqu'à fin compute (96.5% temps)
     e. bo::sync() READ          → activations → DRAM
                                 ← ~1.1 ms/layer × 32 = ~35ms théorique
                                  ← mesuré : 127ms/token (overhead ×3.7)
    ↓
7. Wakeup/sampling (next-token) → ~26ms
8. HTTP Response
```

### 5.3 TTFT wall-clock décomposition

```
TTFT wall = 4024ms (prompt 42 tokens)
├── prefill_duration (API FLM) : 1420ms (35%)
├── XRT init overhead (H6) :     2362ms (59%) ← RECONSTRUCTION HW_CONTEXT
│     ├── hw_context::hw_context()   → ~800ms
│     ├── kernel::kernel()           → ~500ms
│     ├── bo::bo() × N              → ~600ms
│     └── runlist::runlist()        → ~462ms
├── Sleep patch (HTTP path) :       0ms (H1 réfuté, patch neutre)
└── Réseau/margin :                 242ms (6%)
```

### 5.4 Modèles enregistrés dans FLM

```
| Modèle              | Tag              | version | ctx_defaut | vlm | Statut  |
|---------------------|------------------|---------|------------|-----|---------|
| Qwen3.5-9B-NPU2     | qwen3.5:9b       | 0.9.43  | 32768      | Oui | ✅      |
| Qwen3.5-9B-cust-NPU2| qwen3.5:9b-cust  | 0.9.43  | 262144     | Non | ❌ absent model_list |
| Qwen3.5-4B-NPU2     | qwen3.5:4b       | 0.9.43  | 32768      | Oui | ✅      |
| Qwen3.5-0.8B-NPU2   | qwen3.5:0.8b     | 0.9.38  | 16384      | Oui | ⚠️ version |
| Llama-3.2-1B-NPU2   | llama3.2:1b      | 0.9.43  | 8192       | Non | ✅      |
| LFM2-1.2B-NPU2      | lfm2:1.2b        | 0.9.43  | 8192       | Non | ✅      |
| DeepSeek-R1-8B-NPU2 | deepseek-r1:8b   | 0.9.42  | 16384      | Non | ⚠️ version |
| Phi4-mini-IT-4B-NPU2| phi4-mini-it:4b  | 0.9.43  | 8192       | Non | ✅      |

Fichier : C:\Program Files\flm\model_list (compilé dans flm.exe)
Patch manquant : model_list_patch.json pour qwen3.5:9b-cust
```

---

## 6. XRT — API, ABI, Timings

### 6.1 ABI réelle (tailles mesurées par Ghidra)

```
Structure                Taille publique documentée    Taille réelle (RVA)    Delta
────────────────────────────────────────────────────────────────────────────────
xrt::bo                  48 B                         152 B (0x98)         +104 B
xrt::hw_context          ~48 B                        136 B (0x88)         +88 B
xrt::device              ~56 B                        208 B (0xD0)         +152 B
xrt::run                 ~48 B                        80 B  (0x50)         +32 B
xrt::xclbin              ?                            64 B  (0x40)
xrt::ip                  ?                            56 B  (0x38)

Preuve Ghidra : instructions MOV ECX, size dans le .text
  0x50 (80 B)  = run size × 19 occurrences
  0x98 (152 B) = bo size × 19 occurrences
  0x88 (136 B) = hw_context size
  0xD0 (208 B) = device size

DÉCOUVERTE CRITIQUE : xrtBOAlloc IGNORE le paramètre size
  → rdx (second argument = size) jamais utilisé sur 12 appels analysés
  → Le driver décide la taille d'allocation lui-même
  → Le proxy marche par TIMING/ÉTAT INITIAL, pas par correction de size
```

### 6.2 Pipeline XRT par token

```
API C++ observée (mangled names):

Construction (une fois par requête) :
  runlist::runlist(hw_context const&)             — crée la runlist
  kernel::kernel(hw_context const&, module, name) — ouvre le kernel

Par sous-tenseur (3064 appels/token) :
  bo::bo(device const&, size_t)                   — alloue BO
  run::set_arg_at_index(int, bo const&)           — bind argument
  runlist::add(run const&)                        — ajoute à la runlist
  bo::sync(xclBOSyncDirection, offset, size)      — sync DMA (35-400 µs)

Par layer (32 layers × ~1.1 ms) :
  runlist::execute()                              — soumet au NPU
  runlist::wait(chrono::duration)                 — bloque (81.32ms/token cumulé)

Par token (1×) :
  runlist::~runlist()                              — détruit
  kernel::~kernel()                                — ferme kernel
  bo::~bo() × N                                    — libère BOs

Constructeur watch (H6) :
  ?wait@runlist@xrt@@QEBA?AW4cv_status@std@@AEBV$duration@_JU$ratio@$00$0DOI@@std@@chrono@4@@Z
```

### 6.3 Timing XRT (xrt_trace.csv, 6144 lignes)

```
| Opération          | Nb appels | % total | Durée moy | Durée cumulée |
|--------------------|-----------|---------|-----------|---------------|
| bo::sync           | 3 865     | 62.9%   | 35-400 µs | ~8.4s (67%)   |
| run::wait          | 1 105     | 18.0%   | ~3000 µs  | ~3.3s (26%)   |
| runlist::wait      | 1 104     | 18.0%   | ~3000 µs  | ~3.3s (26%)   |
| bo::~bo            | 68        | 1.1%    | ~100 µs   | ~0.01s (0.1%) |
| **Total**          | 6 144     | 100%    |           | 12.6s         |

Détail bo::sync :
  3 MB  → 35-50 µs
  128 MB → 300-400 µs

run::wait = 81.32 ms/token = 96.5% du temps NPU
Overhead gap (non-XRT) : 467ms = ~0.57ms/call = overhead OS/scheduling

IOCTL dispatch : 67 µs × 3064 = 205ms cumulé / token
```

---

## 7. Q4NX — Format de poids

### 7.1 Structure binaire

```
format safetensors-like :

Bytes 0-7       : uint64 LE = longueur header JSON (55496 pour Qwen3.5-9B)
Bytes 8-55503   : JSON dict UTF-8
  { "nom_tenseur": {
      "dtype": "BF16"|"I8"|"F32",
      "shape": [p, q, 5120],
      "data_offsets": [start, end]
    }, ... }
Bytes 55504+    : données binaires brutes
  offset = 8 + hdr_len + data_offsets[0]

Lecture Python :
  with open('model.q4nx', 'rb') as f:
      hdr_len = struct.unpack('<Q', f.read(8))[0]
      meta = json.loads(f.read(hdr_len))
      data_start = 8 + hdr_len
      mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
      tenseur = mm[data_start + off0 : data_start + off1]
```

### 7.2 Contenu Qwen3.5-9B (7.4 GB, 475 tenseurs)

```
Dtype | Nb   | Rôle
------|------|--------------------------------------------------
BF16  | 178  | Embeddings + copies BF16 des SSM critiques
I8    | 249  | Poids Q4_1 packed (nibbles + scales BF16) ← format RÉEL
F32   | 48   | Scalaires SSM (ssm_a, ssm_dt.bias)
Total | 475  |

DÉCOUVERTE : Q4NX = I8 (INT8), PAS Q4 (INT4) !
  Taille réelle ~7.5 GB au lieu de ~4.5 GB attendus pour du INT4
  Gain = bandwidth reduction uniquement (dequant → BF16 avant compute)
  Pas de compute INT4 natif
```

### 7.3 Format Q4_1 packed

```
Config : row_block=32, col_block=256, parallel_size=16, keep_block_in_2D=true

Shape logique → Shape Q4NX :
  W[rows, cols] → [rows/32, cols/256, 5120]

Structure d'un bloc [p, q] (32 lignes × 256 colonnes = 5120 bytes) :
  Bytes 0-511     : 256 BF16 = scales (d)  — colonne-major (group, row_local)
  Bytes 512-1023  : 256 BF16 = mins (m)   — même layout
  Bytes 1024-5119 : 4096 bytes = Q4 nibbles repackés (2 poids/byte)

Dequantization :
  W[row,col] = q_val * d[group*32 + row_loc] + m[group*32 + row_loc]
  où group = col % 256 // 32
  et q_val = nibble extrait du byte paqué (4-bit: bas/haut)
```

### 7.4 Shapes réelles des tenseurs Qwen3.5-9B

```
| Nom Q4NX                 | Shape Q4NX          | Shape réelle          |
|--------------------------|---------------------|-----------------------|
| qkv_proj.weight          | [256, 16, 5120]     | [8192, 4096]          |
| mlp.gate_proj.weight     | [384, 16, 5120]     | [12288, 4096]         |
| mlp.up_proj.weight       | [384, 16, 5120]     | [12288, 4096]         |
| mlp.down_proj.weight     | [128, 48, 5120]     | [4096, 12288]         |
| self_attn.gate_proj.w    | [128, 16, 5120]     | [4096, 4096]          |
| ssm_out_proj.weight      | [128, 32, 5120]     | [4096, 8192]          |
| lm_head.weight           | [7760, 32, 5120]    | [248320, 4096]        |

Formule : shape_réelle = [p * 32, q * 256]
```

### 7.5 Dequant DLL

```
dequant.dll (375 KB, 783 exports, pattern pImpl)
  Export #685 = THUNK mort (8 B, forwarder)
  Export #686 = vraie fonction @ RVA 0xB0F0 (400 B stack frame)
  3 pré-conditions : count ≠ 0, count & 0x1FF == 0, n_blks & 0x7F == 0

dequant.xclbin = 141 KB
  kernel : dpu_kernel_id = 0x901
  AIE header @ +0x08 = 0x00000800 (8 colonnes)

FLM_Q4NX_Converter : 15 architectures supportées, ~30 min par modèle
```

---

## 8. GATED DELTA NET — Algorithme SSM Qwen3.5

### 8.1 Architecture du modèle

```
Qwen3.5-9B — GateDeltaNet (hybride SSM/Transformer)

32 layers :
  24 linear_attention (SSM) + 8 full_attention (pattern 3:1)

Paramètres :
  hidden_size      = 4096 (9B) / 2560 (4B) / 3584 (9B-cust)
  q_num_heads      = 16
  kv_num_heads     = 32  ← GQA INVERSE (Q < K/V, inhabituel)
  head_dim         = 128
  intermediate     = 12288
  full_attention_interval = 4
  num_key_value_heads = 4 (GQA standard pour full-attention)
  vocab_size       = 248320 (VLM) / 151936 (text-cust)
  flm_version      = 0.9.43
```

### 8.2 Algorithme SSM (1 token, decode)

```
Pour chaque couche SSM :

1. QKV = input @ W_qkv                    [4096] → [8192+...]
2. q, k, v = split(QKV)                   → q:[2560], k:[2560], v:[3072]
3. reshape q → [16, 128], k → [32, 128], v → [32, 128]
4. q = q / sqrt(128)                       — normalisation
5. k = k / ||k||                           — normalisation L2

6. conv_input = [q, k, v]                  [8192]
7. conv_state = shift(conv_state, conv_input)  — shift register (kernel=4)
8. conv_out = depthwise_conv1d(conv_state, W_conv)  — groups=8192
9. conv = silu(conv_out)                   — activation SiLU

10. a = ssm_alpha(conv)                    [32] — gate exponentielle
11. b = ssm_beta(conv)                     [32] — gate de mise à jour
12. decay = exp(a)                         — taux d'oubli

13. S = decay * S + k_T @ v_T              — mise à jour SSM [32,128,128]
14. Y = q @ S                              — lecture SSM
15. Y = Y * b                              — gate beta

16. out = Y @ W_out + residual             — projection sortie

Dimensions internes :
  state : [batch, 32, 128, 128]            — état récurrent matriciel
  Chaque tête q (16) lit depuis 2 têtes kv (GQA inverse)
  gate a et beta : [batch, 16] → broadcast sur 32 têtes kv
```

### 8.3 Prefill parallèle (SSM)

```
Utilise GateDeltaNet_prefill.xclbin (189 KB)
  Principe : scan parallèle O(1) en tokens
  chunk_size = 64 tokens par chunk
  Au lieu de boucler token par token :
    1. Calculer toutes les mises à jour d'état en parallèle
    2. Appliquer les décroissances cumulatives
    3. Lire les sorties
  Complexité : O(chunk_size² × head_dim) par chunk

CONSÉQUENCE : TTFT prefill CONSTANT quel que soit le nb de tokens SSM
  Prefill 9 tokens = 941ms | 33 tokens = 936ms
  Batch fixe indépendant du nb de tokens !
```

### 8.4 Couches full_attention (8/32)

```
  GQA standard (num_key_value_heads=4)
  KV cache : BF16
  q_proj / k_proj / v_proj / o_proj : Q4_1 (α_w=0.381-0.424)
  Attention non fusionnée → overhead par layer

  KV bytes/token : 16,384 B (full-attention) / 30,000 B (total)
```

---

## 9. MÉMOIRE

### 9.1 Budget mémoire

```
| Composant             | Taille        | Localisation          |
|-----------------------|---------------|-----------------------|
| Qwen3.5-9B Q4NX       | 7.90 GB       | DRAM (fichier)        |
| Qwen3.5-9B-cust       | 7.90 GB       | DRAM (fichier)        |
| W_eff (poids decode)  | 2.87-3.83 GB  | chargement layer par layer |
| KV cache (ctx=8192)   | 245.8 MB      | DRAM (allocation XRT) |
| SRAM L1 NPU           | 2 MB          | NPU on-chip           |
| SRAM L2 NPU (MemTile) | 4 MB          | NPU on-chip           |
| SRAM totale NPU       | 6 MB          | NPU on-chip           |
| RAM système           | 24.3 GB libre | DRAM système          |
| RAM système (occupé)  | 31.1/33.6 GB  | 92.4% utilisé         |
| Page file             | 21.8/29.7 GB  | SSD                   |

Modèles installés (C:\Users\videl\flm\models\) :
  10 modèles, total brut ~37 GB
```

### 9.2 Bande passante DDR5

```
BW nominale LPDDR5X          : 89.6 GB/s (peak) / ~128 GB/s (paper TileFuse)
BW effective decode           : 21.93 GB/s (24.5% du peak)
BW effective prefill          : ~30 GB/s
η (efficacité DDR)            : 0.724 (perte DM: 27.6%)

Causes des 24.5% :
  - Scheduling XRT idle pendant compute : ×3.7
  - iGPU 880M partage le bus : -3.5 GB/s
  - iGPU framebuffer 2560×1600@180Hz
  - Page file actif (92.4% RAM utilisée) : -10%
  - Column mapping idle tiles : -40%

Contention RAM détectée :
  Qwen3.5-9B standard en contention (18-19 juin) :
    W_eff virtuel = 4.59-4.93 GB (+60%)
    TPS tombé de 7.43 → 4.27-4.65 t/s
  En état propre (21 juin) :
    W_eff = 2.86-2.92 GB (-2%)
    TPS = 7.06-7.43 t/s

  Diagnostique : inflation W_eff due à la contention LPDDR5X
  (navigateur, GPU, OS en arrière-plan)
```

### 9.3 KV Cache par modèle et par contexte

```
| ctx    | llama3.2:1b | qwen3.5:4b | qwen3.5:9b | deepseek-r1:8b |
|--------|-------------|------------|------------|----------------|
| 1024   | 16.8 MB     | 30.7 MB    | 30.7 MB    | 67.1 MB        |
| 2048   | 33.6        | 61.4       | 61.4       | 134.2          |
| 4096   | 67.1        | 122.9      | 122.9      | 268.4          |
| 8192   | 134.2       | 245.8      | 245.8      | 536.9 ⚠️       |
| 16384  | 268.4       | 491.5      | 491.5      | 1073.7 🔴       |
| 32768  | 536.9       | 983.0      | 983.0      | 2147.5 🔴       |
```

---

## 10. PERFORMANCE

### 10.1 Tous les benchmarks (TPS decode)

```
| Modèle                | Backend    | TPS    | Prefill | W_eff  | Notes                  |
|-----------------------|------------|--------|---------|--------|------------------------|
| TinyLlama 1.1B        | OGA NPU    | 35.7   | —       | ~0.5   | INT4, modèle test      |
| TinyLlama 1.1B        | CPU INT4   | 71.8   | —       | —      | Baseline CPU           |
| lfm2:1.2b             | FLM Q4NX   | 46-57  | 109.5   | 0.60   | Modèle le + rapide     |
| qwen3:1.7b            | FLM        | 34.8   | 104.8   | ~0.83  | Code/traduction        |
| qwen3.5:0.8b          | FLM        | 39.0   | —       | 0.48   | NPU-saturé TTFT        |
| qwen3.5:4b            | FLM        | 12-16  | —       | 1.75   | Plafond BW 30 GB/s     |
| **qwen3.5:9b**        | **FLM**    | **7.2-7.8** | **46.0** | **3.80** | **★ Plafond HW**   |
| qwen3.5:9b-cust       | FLM        | 7.42   | —       | 2.87   | Text-only, 97% plafond |
| qwen3.5:9b-cust ctx8k | FLM        | 7.09   | —       | 2.86   | Dérive KV -7.7%        |
| deepseek-r1:8b        | FLM        | 9.5-10 | —       | 5.74   | MoE efficient          |
| Qwen 2.5 7B           | VitisAI    | 9.75   | —       | —      | NPU pur                |
| Qwen 2.5 7B           | Hybride    | 5.93   | —       | —      | Perte -40% MCDM        |
| Phi-3.5-mini          | VitisAI    | 11.96  | —       | —      | 3.8B model             |
| **Qwen 3.5 9B**       | **RTX 5070** | **53.83** | **2029** | —      | **GPU CUDA**       |
| llama3.2:1b           | FLM        | 19.7   | 1.6s    | ~0.5   | SRAM cached            |
```

### 10.2 Roofline calibré (Qwen3.5-9B XDNA2)

```
Formule :
  TPS = BW_eff × η / (W_eff + c × KV)
  TPS = 21.93 × 0.724 / (3.80 + 0.28 × KV)

Paramètres calibrés :
  BW_eff         = 21.927 GB/s
  η              = 0.7309 (efficacité DDR)
  W_eff          = 3.83 GB (9B standard) / 2.87 GB (cust)
  c              = 0.28 (moyen, varie 0.25-0.33)
  KV_bytes/token = 30,000 B (total) / 16,384 B (full-attn)
  PLAGE_VALIDEE  = 1024-8192 ctx, R² = 0.96

Temps par token :
  step_ms(ctx) = 82.3 + 0.00175 × ctx  (ms)
  A = 82.3 ms (temps fixe : poids + dispatch + wakeup)
  B = 0.00175 ms/tok (KV scaling)
  R² = 0.96
```

### 10.3 Décomposition temporelle par token (mesuré)

```
| Phase                    | Durée   | % temps | Détail                         |
|--------------------------|---------|---------|--------------------------------|
| 0x310D0 decode loop CPU  | 60 ms   | 54%     | 32 layers × orchestration      |
|  ├─ gen_*_seq (32 layers) | ~15 ms  | 13%     | prod. npu_sequence* CPU        |
|  └─ dispatches XRT       | ~45 ms  | 41%     | 3064 appels × 67µs             |
| bo::sync + DMA            | 20 ms   | 18%     | sync WRITE/READ poids          |
| NPU compute (matmul)      | 5 ms    | 4.5%    | 639 instructions GEMM/chaine   |
| Wakeup/sampling           | 26 ms   | 23%     | next-token decoding            |
| **Total token**           | **111ms**| **100%** | **9 TPS théorique, 7.6 mesuré**|

CPU total : 86 ms (86.8%)
NPU idle  : 99.73% (138/51,300 GOPS actifs)
DMA idle  : pendant compute (pas de prefetch layer suivant)
```

---

## 11. OGA / RadeonML / DirectML — Stack alternative

### 11.1 Stack actuelle OGA

```
OGA 0.14+ → DirectML.dll → D3D12.dll → dxgkrnl.sys → NPU XDNA2

Pipeline OGA :
  Python C++ OGA API
    ↓
  onnxruntime-genai
    ↓
  onnxruntime (DirectMLExecutionProvider)
    ↓
  DirectML.dll → D3D12.dll → D3DKMT → dxgkrnl.sys → NPU

Modèles supportés OGA + DirectML 1.24.4 :
  TinyLlama 1.1B INT4     → 74.8 TPS (CPU) / 35.7 TPS (NPU)
  Qwen2-1.5B INT4         → ~50 TPS (NPU estimé)
  Qwen3.5-4B/9B INT4      → ❌ Bloqué (LinearAttention pas dans ORT 1.24.4)
```

### 11.2 Pipeline RadeonML

```
Pipeline FLM (qui marche) :
  FLM → vitis-ai-runtime2.dll → RadeonML_ipu.dll → D3DKMT → ipustack.sys → NPU

API RadeonML (C, 44 exports, wchar_t* sur Windows) :
  rmlCreateDefaultContext()       → Contexte NPU
  rmlLoadGraphFromFile(path)      → Charge modèle (path wchar_t* !)
  rmlCreateModelFromGraph()       → Modèle exécutable
  rmlSetModelInput/model/output   → Attache tenseurs
  rmlInfer()                      → Inférence (D3DKMTSubmitCommand en interne)
  rmlWaitDevice()                 → Attend résultat
  rmlMapTensor/UnmapTensor        → Lit/écrit tenseurs

Erreur 0xFFFFFF7E = RML_ERROR_FILE_NOT_FOUND (-130)
  Cause : path passé en char* au lieu de wchar_t*
```

### 11.3 Limitations OGA Qwen3.5

```
2 ops com.microsoft non supportés par DirectML 1.24.4 :
  1. CausalConvWithState (24×) — registre à décalage + Conv1D depthwise
     ✅ PATCHÉ : cosine_sim=1.0, max_error=1.9e-6 (24/24 patches)
  2. LinearAttention / Gated DeltaNet (24×) — ⛔ NON RÉSOLU
     Mise à jour récurrente matricielle [32,128,128]
     Kernel AIE dédié nécessaire (FLM le fournit via GateDeltaNet_prefill.xclbin)

Bug CastAvx (RÉSOLU 2026-06-24) :
  Cause : ABI mismatch entre onnxruntime PyPI 1.22.0 et build AMD 1.23.3.dev
  Fix : Réinstaller ryzenai-lt-1.6.1.exe (contient ORT matching)
  Symptôme : WinError 1114 (DLL init failed) = onnxruntime_providers_ryzenai.dll
```

---

## 12. GHIDRA — Résultats complets du reverse

### 12.1 qwen3_5vl_npu.dll (3.0 MB)

```
Exports : 3 526
Fonctions : 14 318
RVA clés :
  0x310D0  = decode loop CPU (layer orchestrator, 60ms/token)
  0x30010  = gen_layer_seq
  0x180042e00 = update_ping_pong_flag (body=10 insns, INEFFECTIF)
  0x180042e10 = wait_DMA_queue_if_full (body=133 insns)
  0xC340   = dispatch_loop (boucle dispatch IOCTL NPU)
  0x13490  = to_npu (transfert commandes NPU)
  0x14C12  = allocator (allocation buffer DMA)
  0x2B60   = enqueue (file commandes XRT)

Structures FLM identifiées :
  qwen3_5vl_npu_sequence  — séquence d'instructions NPU (destructeur exporté)
  npu_app_manager         — gestionnaire d'application NPU
  npu_tiles               — enum des tuiles AIE
  ert_cmd_state           — machine à états ERT
  xrt::xclbin::kernel     — référence kernel xclbin
  xrt::bo                 — Buffer Object XRT
  xrt::hw_context         — Contexte HW (smart pointer)
  buffer<bfloat16_t>      — Buffer BF16 (poids)
  conv3d_patch_embed      — Conv patch embed (4 var. AVX512)
  layernorm_high_precision — Layer norm (body=1542 insns)

50 IOCTLs/token : DMA_SUBMIT(12.6×), DMA_WAIT(12.9×),
                  SUSPEND(12.6×), RESUME(12.6×)
                  ~78ms overhead pour 5ms NPU (94% perdu en marshaling)
```

### 12.2 ipustack.sys (500 KB) — C:\tmp\ghidra_npu_full\

```
Fonctions : insuffisamment analysé (1.2 GB de projet, pas de place C:)
Découvertes :
  1008 LFENCE dans .text (barrières de mémoire, pas 13 comme estimé)
  19 × MOV ECX, 0x50 (80B) — run size
  19 × MOV ECX, 0x98 (152B) — bo size
  DMA scatter-gather : 24.4% BW efficiency
  8 AIE columns, 32 tiles
  6 MB SRAM total
  TDR timeout : 2000 ms (aie2_ctx.c)
```

### 12.3 dequant.dll (375 KB)

```
783 exports, pattern pImpl
Export #685 = THUNK mort (8 B, simple forwarder)
Export #686 = vraie fonction @ RVA 0xB0F0 (400 B stack frame)
3 pré-conditions : count≠0, count&0x1FF==0, n_blks&0x7F==0
dequant.xclbin = 141 KB, kernel dpu_kernel_id=0x901
  AIE header @ +0x08 = 0x00000800 (8 colonnes)
```

### 12.4 xrt_coreutil.dll (versions)

```
Proxy v18.1  (153 KB)  — hook xclLoadXclBin, actif
Proxy v19_slab (257 KB) — slab allocator
Proxy v24    (272 KB)  — 507 exports C++ : 11 hooks + 496 forwarders
Proxy v25b   (inconnu) — 502 forwarders, INACTIF
                          GetProcAddress retourne NULL pour xrtBOAlloc
                          8 hypothèses écartées (CFG, SHA256, permissions, .dll vs .DLL...)
```

---

## 13. OPTIMISATIONS — 25 techniques classées

### 13.1 Bandwidth (gain immédiat, 0 effort)

```
| # | Technique                    | Gain BW     | Gain TPS | Fichier             |
|---|------------------------------|-------------|----------|---------------------|
| O2| Écran 60 Hz                  | +2 GB/s     | +10%     | bus_optimizer.py    |
| O4| Tuer Edge (msedge.exe)       | +3-8 GB/s   | +15-35%  | bus_optimizer.py    |
| O3| WSL2 --shutdown              | +0.5-1 GB/s | +3-5%    | bus_optimizer.py    |
| O5| Priority AboveNormal (OGAD)  | —           | +5-10%   | bus_optimizer.py    |
| O7| Pre-warm NPU (1 token dummy) | —           | -2092ms  | bus_optimizer.py    |
|   | Désactiver iGPU framebuffer  | +2-3 GB/s   | +10-12%  | —                   |
|   | **Total bus**                | +7-14 GB/s  | **+30-60%** | —                 |
```

### 13.2 KV Cache

```
| # | Technique                    | Gain TPS    | Fichier               |
|---|------------------------------|-------------|-----------------------|
| K1| KV Adaptive Throttle         | +3-55%      | kv_controller.py      |
| K2| KV INT4 quantization         | ×4 capacité | kv_controller.py      |
| K3| StreamingLLM Sink (4 tok.)   | -97% KV     | kv_controller.py      |
| K4| PagedAttention (blocs 16)    | Allocation  | kv_controller.py      |
```

### 13.3 Architecture (batching + spéculation)

```
| # | Technique                    | Gain TPS    | Fichier               |
|---|------------------------------|-------------|-----------------------|
| B1| Continuous Batching          | ×2-5        | batching_scheduler.py |
| B2| Speculative Proxy (iGPU)     | ×1.6-2.8    | spec_proxy.py         |
| B3| Async Pipeline (token N+1)   | +30-50%     | —                     |
```

### 13.4 Modèle (compilation, quantification)

```
| # | Technique                    | Gain TPS    | Fichier               |
|---|------------------------------|-------------|-----------------------|
| M1| Mixed Precision D2 (ILP)     | ×2          | d2_xdna2_unified.py   |
| M2| Phase 3 Dispatch NPU/iGPU    | +15-35%     | phase3_v3_dispatch.py |
| M3| Dimension Padding (512 mult) | +50% tuiles | patch_compiler.py     |
```

### 13.5 Bypass MCDM (driver)

```
| # | Technique                    | Gain TPS    | Effort  | Risque |
|---|------------------------------|-------------|---------|--------|
| P1| Proxy IOCTL Interceptor      | +393%       | 30j.h   | 6/10   |
| P2| Direct PCIe NPU Driver       | +500%       | 90j.h   | 8/10   |
| P3| amdxdna.sys Patching         | +300%       | 15j.h   | 4/10   |
```

### 13.6 Roofline cible INT4 custom

```
Permet Qwen3.5-9B Q4NX :
  TPS = 0.87 × 64 / (4.5 + 0.75) = 10.6 t/s théorique
  Actuel : 7.42 t/s (custom, ctx=1024) = 70% du peak INT4
  Gap : 3.2 t/s = overhead XRT scheduling non résolu
```

---

## 14. BOTTLENECKS — Les 4 goulots (avec preuves)

### Goulot #1 — MCDM SHIM dispatch (96.5% du temps NPU)

```
run::wait = 81.32 ms/token = 96.5% du temps NPU
├── 537 gaps > 111ms = 80% du temps token
├── 3064 dispatch × 67µs = 205ms cumulé / token (scheduling pur)
├── IOCTL overhead = 1658 × 67µs = 111ms par session
├── Scheduler Python = +2000ms fixe par requête
└── 50 IOCTLs/token : DMA_SUBMIT, DMA_WAIT, SUSPEND, RESUME

PREUVE : xrt_trace.csv (6144 lignes, 8 captures)
  run::wait = 81.32 ms/token
  138/51,300 GOPS actifs = 0.27%

CAUSE RACINE : Windows MCDM n'a PAS force_cmdlist (contrairement à Linux)
  Chaque commande = 1 IOCTL D3DKMT séparé
  Pas de batching automatique

STRATÉGIES DE BYPASS :
  P1 : Proxy Driver IOCTL Interceptor → +393% (7.5→37 TPS)
  P2 : Direct PCIe NPU Driver → +500% (7.5→45 TPS)
  P3 : amdxdna.sys Patching → +300% (7.5→30 TPS)
```

### Goulot #2 — BW DDR5 saturée (96.7% utilisée)

```
BW effective : 21.93 / 89.6 GB/s = 24.5% du peak
├── iGPU 880M : -3.5 GB/s (framebuffer 2560×1600@180Hz)
├── Column mapping idle tiles : -40%
├── Paging RAM (92.4% utilisée) : -10%
├── Pas de ping-pong DMA → idle SHIM pendant compute
└── Overhead XRT scheduling ×3.7

PREUVE : 21.93 GB/s calibré par régression OLS sur 3 runs live
  R² = 0.96, step_ms(ctx) = 82.3 + 0.00175 × ctx
  BW nominale ~128 GB/s (TileFuse paper)

SOLUTIONS :
  ROCmFP4 (-25% BW) → +25% TPS
  XQuant (7.9→~4 GB) → +100% TPS
  Désactiver iGPU → +12% TPS
```

### Goulot #3 — H6 : XRT init overhead fixe (2362ms)

```
2362ms = reconstruction hw_context + remapping BO à chaque requête HTTP
  ZÉRO compute dans ces 2362ms
  37% du temps TTFT wall à se "réchauffer"

PREUVE : h6_validation.py (2026-06-21)
  7 délais inter-requête (0 à 5000ms)
  Ratio R2/R1 = 0.924-0.988 pour TOUS les délais
  AUCUNE fenêtre de contexte chaud détectée
  Même à 0ms entre R1 et R2 : pas de warm contexte

Décomposition :
  hw_context::hw_context()     → ~800 ms (métadonnées xclbin)
  kernel::kernel()             → ~500 ms (configuration AIE tiles)
  bo::bo() × N                 → ~600 ms (BO mapping)
  runlist::runlist()           → ~462 ms (DMA SHIM locks)

SOLUTION : Persistent hw_context (mode "Stateful")
  TTFT estimé sans overhead : ~1420ms (prefill pur)
  Gain : -65%
  Blocage : AMD/FLM doit réécrire le serveur
```

### Goulot #4 — Proxy v25b inactif

```
v25b (502 forwarders) ne fonctionne PAS
  GetProcAddress retourne NULL pour xrtBOAlloc
  8 hypothèses écartées :
    CFG (Control Flow Guard) ❌
    SHA256 signature ❌
    Permissions fichier ❌
    DLL search order ❌
    .dll vs .DLL casse ❌
    Dépendances manquantes ❌
    Version Windows ❌
    Antivirus ❌
  Cause probable : conflit loader Windows (non résolu)
```

---

## 15. DIAGRAMME DE FLUX TEMPOREL — 1 TOKEN

```
ÉCHELLE : 111ms pour 1 token (9 TPS brut, 7.6 mesuré)

TEMPS (ms) :
0                   60           80        85             111
├────────────────────┼─────────────┼─────────┼──────────────┤
│ 0x310D0            │ bo::sync   │ NPU     │ Wakeup/      │
│ CPU decode loop    │ + DMA      │ MatMul  │ Sampling     │
│ (orchestration)    │             │         │ (next-token) │
│ 60ms               │ 20ms        │ 5ms     │ 26ms         │
│ (54%)              │ (18%)       │ (4.5%)  │ (23%)        │
└────────────────────┴─────────────┴─────────┴──────────────┘
←── 86 ms CPU (86.8%) ──→← 20ms DMA →← 5ms NPU →

DÉTAIL DES 60ms CPU (0x310D0) :
├── 32 layers × génération de séquence
│   ├── gen_attention_seq  (8 layers full-attention)
│   ├── gen_ffn_seq        (32 layers FFN)
│   └── gen_ssm_seq        (24 layers SSM)
├── 32 layers × ~47 sous-tenseurs = 1504 dispatches
├── 3064 appels XRT × 67µs = 205ms dispatch cumulé
└── Scheduler Python = +2000ms fixe/requête

DÉTAIL DES 20ms DMA :
├── bo::sync WRITE : poids W_n (~300 KB/layer) → AIE SRAM : 35-50 µs
├── bo::sync READ : activations → DRAM : 5 µs
├── Pas de prefetch du layer suivant (SHIM idle pendant compute)
└── 3865 bo::sync total = 62.9% des opérations XRT

DÉTAIL DES 5ms NPU :
├── 639 instructions GEMM schedulées 1 par 1 (monochain)
├── Scheduler monochain (pas de force_cmdlist sur Windows)
├── GOPS actifs : 138/51,300 = 0.27%
└── ×2-4 TPS potentiel si cmd-chain activée

SÉRIALISATIONS AVANT NPU (4 couches) :
  1. FLM mutex (queue=6, atomique)
  2. WorkloadsSessionHost (3 processus : 1+2 workers)
  3. xrt_coreutil DLL STUB (forwarder monochain, 507 exports)
  4. DPU Scheduler Monochain (639 instructions × 1/1)
```

---

## 16. GLOSSAIRE

```
| Terme              | Définition                                                |
|--------------------|-----------------------------------------------------------|
| MCDM               | Microsoft Compute Driver Model — kernel driver DirectX    |
| D3DKMT             | DirectX Graphics Kernel Mode — API d'accès GPU/NPU kernel |
| SHIM               | Streaming Hardware Interface Module — DMA engine du NPU   |
| XRT                | Xilinx Runtime — framework de gestion kernels FPGA/NPU    |
| ERT                | Embedded Runtime — protocole de commande NPU/XRT          |
| BO                 | Buffer Object — allocation mémoire pour le NPU            |
| HW context         | Contexte matériel NPU (xclbin + état AIE tiles)           |
| TDR                | Timeout Detection Recovery — 2s avant reset driver        |
| KDMA               | Kernel DMA — mécanisme de DMA direct (pas sur Windows)    |
| RadeonML IPU       | API AMD propriétaire pour NPU (44 exports C)              |
| OGA                | ONNX Runtime Gen AI — framework AMD pour LLM sur NPU      |
| VitisAI EP         | ONNX Execution Provider pour NPU AMD                      |
| AIE2P              | Architecture NPU XDNA2 (5 accumulateurs par tile)         |
| Overlay            | Fichier xclbin contenant la configuration des AIE tiles   |
| IOCTL              | Device I/O Control — appel système pour driver            |
| PLL                | Phase-Locked Loop — registre d'horloge (ex: 1000 MHz)     |
| SRAM L1            | 64 KB par tile AIE — mémoire locale ultra-rapide          |
| MemTile            | Tile mémoire (512 KB) entre les colonnes AIE — cache L2   |
| LFENCE             | Instruction barrière mémoire x64 — 1008 dans ipustack.sys |
| VLIW               | Very Long Instruction Word — format instr. NPU            |
| PMC                | Performance Monitor Counter — opcode DMA = 1              |
| TXN AIE2           | Transaction NPU AIE2 — opcode DMA = 3                     |
```

---

*Carte fonctionnelle générée le 20/07/2026 — 116 rapports .md, 29 JSON, 44 CSV, 200+ scripts Python, Ghidra 14318 fonctions, 4 dépôts GitHub AMD reverse-engineered, 55 découvertes fondamentales intégrées.*
