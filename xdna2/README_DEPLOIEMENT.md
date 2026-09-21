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
