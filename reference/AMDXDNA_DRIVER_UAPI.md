# Référence — UAPI driver amdxdna (Linux) — source amd/xdna-driver amdxdna_accel.h
# Fetched 2026-09-20 · header officiel main

## IOCTL principaux (interface DRM)
- DRM_AMDXDNA_CREATE_HWCTX / DESTROY / CONFIG — contexte matériel + QoS
- DRM_AMDXDNA_CREATE_BO / GET_BO_INFO / SYNC_BO
- DRM_AMDXDNA_EXEC_CMD / WAIT_CMD (seq numbers, timeout ms)
- DRM_AMDXDNA_GET_INFO / GET_ARRAY (query hardware + telemetry)
- DRM_AMDXDNA_SET_STATE (power mode, preempt, clock freq, coredump)

## BO types (critique pour le planner)
| Type | Meaning |
|------|---------|
| AMDXDNA_BO_SHARE (1) | BO partagé user ↔ device (mémoire normale) |
| AMDXDNA_BO_DEV_HEAP (2) | mémoire hôte partagée utilisée comme heap device |
| AMDXDNA_BO_DEV (3) | alloué DEPUIS BO_DEV_HEAP |
| AMDXDNA_BO_CMD (4) | BO commande interne XRT |

→ PAS de limite de taille BO imposée par le driver (limite réelle = ressources Linux / memlock).
→ BO_DEV_HEAP = base pour persister un pool de buffers (éviter malloc/copy/free par expert).
→ GET_BO_INFO retourne vaddr (user VA) + xdna_addr (device VA) → mapping host↔device.

## Sync explicite (base du prefetch planner)
- SYNC_DIRECT_TO_DEVICE (0) : host → NPU
- SYNC_DIRECT_FROM_DEVICE (1) : NPU → host
→ pilier de la primitive "charger expert → XDNA2" / "reprendre résultat".

## QoS hints exposés par le driver (réutilisables par le planner)
struct amdxdna_qos_info : gops, fps, dma_bandwidth, latency, frame_exec_time, priority,
user_start_col, reserved.
→ Le planner peut INDIQUER au driver la bande passante DMA attendue et la latence cible.

## Télémétrie / mesures disponibles (alimentent le cost model)
- QUERY_AIE_METADATA : col_size, cols, rows, tile metadata (dma_channel_count, lock_count, event_reg_count)
- QUERY_SENSORS : power, column_utilization, temperature (input/max/average/highest)
- QUERY_RESOURCE_INFO : npu_clk_max, npu_tops_max, npu_task_max, npu_tops_curr, npu_task_curr
- QUERY_AIE_LOAD : load_percent, operations_per_second, activity_counters
- HWCTX entry : command_submissions, command_completions, migrations, preemptions, errors,
  dma_bandwidth, latency, frame_exec_time, heap_usage, uc_info (health)
- BO usage : total_usage / internal_usage / heap_usage par PID
- Power modes : DEFAULT/LOW/MEDIUM/HIGH/TURBO (+ xrt-smi --pmode performance côté Windows)

## Ce que ça change pour le D2 Planner
1. On peut mesurer en TEMPS RÉEL : utilisation colonnes AIE, température, power, DMA BW,
   TOPS courants → le cost model n'est plus à l'aveugle (retour des faits "9.8 GB/s etc").
2. user_start_col + num_col (QoS / CREATE_HWCTX) → le planner peut réserver une partition
   de colonnes à un contexte (ex. experts MoE) sans que le driver le fasse pour nous.
3. Migrations/preemptions exposées → détecter si un contexte NPU est chassé (contention
   avec un autre processus utilisant le NPU).
4. BO_DEV_HEAP + SYNC_DIRECT_* + BO persistence → primitives "pool d'experts persistant"
   (pas de malloc/copy/free par token).

## Limites connues (à garder pour MANQUANT)
- Header Linux (UAPI). Notre machine tourne Windows : API XRT Windows (xrt-smi, DX/ryzenai)
  différente. À vérifier quel sous-ensemble est exposé sur Windows.
- Le driver n'expose pas de compteur "GB/s effectif DDR" direct → toujours dériver des
  timings réels de kernels (leçon du repo xdna2-).