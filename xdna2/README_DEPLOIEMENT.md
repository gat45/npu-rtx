# DÉPLOIEMENT — machine cible (Ryzen AI 9 365 + RTX 5070 + XDNA2)
# Établi 2026-09-21 (Phase 1.4) — à exécuter UNIQUEMENT sur la machine cible.
# Machine dev (GTX 1080, pas de NPU) : tout ce qui est testable ici l'est via
# xdna2/planner_core.py + xdna2/benchmark_sim.py (simulation, ancres calibrées).

## 0. Préambule provenance

Toute valeur mesurée sur la machine cible remplace une constante ASSUMED/UNKNOWN
de xdna2/planner_core.py et npu-rtx/cost_contention_patch.py. Règle d'or :
ne JAMAIS marquer MEASURED sans trace horodatée dans runs/<ts>/.

## 1. Découverte hardware (remplace les UNKNOWN du HardwareProfile)

```bash
nvidia-smi --query-gpu=name,memory.total,clocks.sm,temperature.gpu,power.draw --format=csv
nvidia-smi -q -d MEMORY,POWER,CLOCK
xrt-smi examine                      # NPU XDNA2 sain ? (attendu : NPU4, 4 colonnes)
xrt-smi validate                     # auto-test firmware
xrt-smi configure --pmode performance
```

## 2. Microbenchs bus (P0.2) — PCIe / H2D / D2H

```bash
python - <<'PY'
import torch, time
a = torch.empty(64*1024*1024, dtype=torch.uint8, device="cuda")  # 64 MiB
b = torch.empty_like(a, device="cpu", pin_memory=True)
for _ in range(5): b.to("cuda")   # warmup
t0=time.perf_counter()
for _ in range(20): b.to("cuda", non_blocking=True)
torch.cuda.synchronize()
print("H2D GB/s:", 64e6*20/(time.perf_counter()-t0)/1e9)
t0=time.perf_counter()
for _ in range(20): a.to("cpu", non_blocking=True)
torch.cuda.synchronize()
print("D2H GB/s:", 64e6*20/(time.perf_counter()-t0)/1e9)
PY
```

## 3. NPU XDNA2 — DMA/submit/completion (P0.6, remplit BW host_coherent)

Reprendre la sonde 1bit : xrt::bo alloc + sync TO/FROM device, tailles
256 KiB → 64 MiB (les tailles experts ≈ 1.69 MiB Q4 / 3.35 MiB Q8).
Mesurer : alloc / map / sync / run / wait séparément → remplir
`BUS_PATHS["host_coherent"]["bw"]` dans cost_contention_patch.py.

## 4. Ancres débit (re-calibrer sur CETTE machine)

```bash
# GPU (attendu ~53.83 tok/s sur 9B, mesuré 06/2026) :
llama-bench -m <qwen35-9b>.gguf -p 0 -n 256 -r 5 -fa 1
# NPU via FLM (attendu 7.06-7.43) : flm.exe run <model> ... (voir SOSC_v4)
```

## 5. Voie A — double-flux indépendant (2 process, mmap partagé)

```bash
# terminal 1 :
llama-server -m <model>.gguf --port 8081 -ngl 99 --mlock
# terminal 2 (backend XDNA quand le backend ggml-xdna est buildé) :
<xdna-runner> -m <model>.gguf
# charge simultanée sur les deux ports → débit agrégat vs somme isolée
```

## 6. Voie B — co-exécution mono-flux (chantier, base self-build-jz)

Base : branche `self-build-jz` (PR #26501 async intégrée), worktree
`E:\oneplus\ab-wt`. Partition MUL_MAT par LIGNES CUDA↔XDNA2 + sync légère
(flag mémoire partagée + poll) isolée du scheduler générique.
Validation correction AVANT perf : diff token-par-token, seed fixe, temp=0.

## 7. Ce que la machine dev (GTX 1080) peut déjà valider

- planner_core.py : placement + coûts + contraintes (logique pure)
- benchmark_sim.py : sensibilité aux distributions Pareto, bornes hit-rate
- adapter_bridge.py : conformité au contrat governor (execute/measure/rollback)
- tests : py -m unittest (dossier xdna2)

## 8. Checklist P0 (RAPPORT_SYNTHESE_CORRECTIONS §5)

| Test | Commande | Remplit |
|---|---|---|
| P0.1 | nvidia-smi (ci-dessus) | HardwareProfile GPU |
| P0.2 | microbench torch (ci-dessus) | PCIe H2D/D2H |
| P0.3 | bench kernel NVFP4 SM120 | kernel_registry |
| P0.4 | bytes/token dense backbone | static/bytes_per_token (+dense) |
| P0.6 | sonde xrt::bo (ci-dessus) | BW host_coherent, scaling cols |
| P0.10 | prefill/decode 4K/32K/128K | full_matrix |

---

## 9. PLAN 5070 FINAL — NVFP4 + KV turbo4 + Voie A (2026-09-21, profiler_tiers)

Décision de configuration, ancrée sur les runs `profiler_tiers/SORTIE_5070*.txt`
(GGUF réel Marco-Nano 8B-A0.6B + modèle 35B-A3B, VRAM 8 Go garde 8 %) :

### 9.1 Configuration retenue

| Paramètre | Valeur | Justification |
|---|---|---|
| Experts | **NVFP4 (1.25 MB/exp)** | résidence 143/256 vs 73/256 en Q4 → hit GPU 0.56 vs 0.25 |
| KV | **turbo4 (K et V)** | ×0.258 vs f16 (0.16 GiB @8k), texte ≈ f16 validé Phase 2 |
| K=turbo3 | **INTERDIT** | bug upstream : garbage CPU ET CUDA (Phase 2) — V=turbo3 OK |
| Résidence cible | **128–143 experts/couche** | frontière VRAM exacte = 143 (sweep) ; 128 = marge KV long ctx |
| Overflow restant | 4.4 MB/tok → **0.14 ms/tok PCIe** | gen5 x8 = 31.5 GB/s ; goulot dur seulement si > 5 ms |
| Contexte long | 64k possible (KV turbo4 1.29 GiB → 119/256, hit 0.46) | le turbo4 est ce qui rend 64k tenable sur 8 Go |
| Decode attendu | ~110 t/s (BW-bound, dense backbone ~1.5 GB/tok) | NVFP4 n'accélère PAS le decode, il augmente le hit |

### 9.2 Voie A — répartition GPU / NPU

- GPU (5070) : flux principal — dense backbone + experts résidents (hit ~0.56).
- NPU XDNA2 (sur-package Strix, zéro PCIe) : **overflow experts** (0.44) au coût
  ×16/hit GPU — jamais backend principal (bench.json Phase 1).
- Agrégat simulé Voie A : 61.5 t/s — à confirmer machine cible (§5 ci-dessus,
  2 process + mmap partagé).
- Alternative mono-flux : Voie B (§6) si le planner partitionne MUL_MAT.

### 9.3 Séquence de validation sur la machine cible

```bash
# 1) vérifier les hypothèses du plan (remplacer ASSUMED par MEASURED) :
py gpu_tier_profiler.py raw   --machine 5070 --gguf <model>.gguf --ctx 8192
py gpu_tier_profiler.py sweep --machine 5070 --model 35b --expert-fmt nvfp4 --kv turbo4 --ctx 8192
llama-bench -m <model>.gguf -p 512 -n 128 -fa on   # recalibre BW_eff (hyp. 165 GB/s)
# 2) microbenchs P0.2/P0.6 (sections 2-3) puis Voie A (section 5)
```

Hypothèses à confirmer en premier : 1.25 MB/expert NVFP4 réel (requant),
BW_eff 165 GB/s, coût overflow NPU ×16.

### 9.4 Mapping repos (rôles) — vérifié 2026-09-21

| Repo | Rôle | Preuves / points d'entrée |
|---|---|---|
| **GaTmanes/xdna2-** (attention : tiret final) | **Runtime d'inférence complet (GGUF → génération) — le plus avancé / principal** | cloné `E:/oneplus/xdna2-runtime` ; fork llama.cpp avec **`bin/ggml-xdna.dll`** (backend XDNA2 natif) + kernels `.xclbin/.insts` pré-buildés (`kernels/decode_layer_f3best_*`, `decode_front_attn_*`) ; Qwen3.5-9B 32 couches validé : CPU-Q4 == NPU-Q4 (8/8 tokens + golden 64/64), batched GEMV ×2.46, 1.86 → 0.667 s/token ; scripts `run_9B_npu_conservateur.sh` |
| **Tagman45/adaptive-xdna-runtime** | **Compilateur adaptatif de kernels XDNA2 : IRON, DMA/TAP, oracle, cache SHA** — couche supérieure, active | cloné `E:/oneplus/adaptive-xdna-runtime` (`54272f2`, MLIR-AIE 1.4) ; **78/78 tests OK** en local ; entrées : `adaptive_xdna_runtime/planner.py`, `compiler.py`, `registry.py` (artefacts validés SHA + ExecutionManifest), `observation.py` (oracles), `mcp_server.py` |

Flux entre les deux couches :

```
requête (op, K, N, quant)
  → adaptive-xdna-runtime : planner variantes → génération IRON → oracle NPU
    → registry (jamais d'artefact non validé ; SHA xclbin + plan DMA liés)
  → xdna2- (runtime principal) : exécution XRT résidente — GGUF → génération
    (ggml-xdna.dll + xclbin ; c'est lui qui porte la Voie A côté NPU :
    l'overflow experts 0.44 du plan §9.1 s'exécute via ce runtime)
  → observation : re-mesure → recalibrage planner (ASSUMED → MEASURED, §0)
```

Note d'usage : le runtime `xdna2-` embarque ses kernels pré-buildés pour
Qwen3.5-9B (géométries figées dans les noms de fichiers) ; pour les NOUVELLES
géométries/experts (35B-A3B), c'est `adaptive-xdna-runtime` qui génère, valide
(oracle) et registre les kernels que `xdna2-` consomme ensuite.

Côté npu-rtx (ce repo) : `xdna2/planner_core.py` + `adapter_bridge.py` et
`profiler_tiers/` produisent placements/sweeps/calibrage consommés par la couche
planner ; l'état du smoked test matériel (GEMV INT8 64×64 validé NPU, aucun
kernel promu auto) est documenté dans
`INT8_GEMV_NPU_BASELINE_2026-09-19.md` du repo cloné.
