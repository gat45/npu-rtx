# profiler-v3 — POSITIONNEMENT vs OUTILS AMD + ARCHITECTURE ADAPTATEURS
# Établi 2026-09-20 · Dossier npu-rtx/ · Sources : AMD AI Analyzer, npu_perf_trace.sh, amdxdna
# telemetry UAPI, xdna-top, ryzenai-lab, xdna-engine

## 1. CONCLUSION DE LA CARTOGRAPHIE

**Personne ne reproduit exactement profiler-v3** (profilage temporel fin + CPU/NPU + DMA/gaps/
sync + modèle de débit + oracle/planner hétérogène). Les projets publics font des morceaux :

| Outil | Couche couverte | Ne fait PAS |
|---|---|---|
| **AMD AI Analyzer** (Ryzen AI 1.8) | modèle → partition CPU/NPU → timeline layer/op → stats (JSON) | ❌ **BF16 uniquement** (INT8 non supporté en 1.8) ; pas expert/paging/RTX |
| **amdxdna + npu_perf_trace.sh** | events XRT/driver/perf → trace temporelle | pas kernel/expert/SSD/PCIe |
| **AMD telemetry UAPI** | hardware counters/sensors (clock, power, col_util) | pas de DAG ni oracle |
| **xdna-top** | monitoring NPU+iGPU, record/compare/baseline | moniteur, pas profiler de kernels ni planner |
| **ryzenai-lab** | benchmarks système (NPU/CPU/power, prefill/decode) | résultat global, pas DAG temporel |
| **xdna-engine** | kernels AIE Rust/XRT, latence int8 | pas la profondeur d'instrumentation |
| **profiler-v3 (nous)** | corrélation temporelle + DMA + compute + gaps + sync + débit + contention + multi-device + oracle + D2 Planner | **position unique — non doublonné** |

## 2. AMD A FOURNI LES PRIMITIVES BAS NIVEAU (à brancher, pas réinventer)

### 2.1 npu_perf_trace.sh (script officiel AMD) — la cible de dissection n°1
```
perf
 ├── amdxdna_trace:*
 └── sdt_xrt:*
       ↓
application → perf record → perf script → trace temporelle
```
Événements XRT dans le driver :
- `XRT_PROFILING_TRACE_PARTITION_INIT` / `_PARTITION_DONE`
- `XRT_PROFILING_TRACE_ENTER` / `_EXIT`
→ permet de reconstruire les intervalles scheduler/contexte.

### 2.2 Telemetry UAPI (amdxdna)
- `DRM_AMDXDNA_QUERY_TELEMETRY`, `_QUERY_SENSORS`, `_QUERY_HW_CONTEXTS`, `_QUERY_CLOCK_METADATA`
- compteurs : NPU clock, H clock, power, column utilization (`npu_busy[i]` par colonne),
  L1 interrupt counter, DMA counter, Deep Sleep counter (MERT)
→ **NPU busy/clock/power SANS instrumentation intrusive** (pas de timers par kernel).

### 2.3 Attention : le NPU peut être idle alors que le chemin complet est long
```
NPU kernel = 30 µs  mais  host_issue + DMA + sync + descriptor_setup = 140 µs
```
→ séparer device_compute / transfer / host_submit / wait (leçon déjà validée côté Hexagon).

## 3. RÈGLE D'OR (le point que xdna-top documente et qu'on adopte)

> **Les compteurs hardware AMD deviennent des OBSERVATIONS du profiler. Le profiler ne doit
> jamais fabriquer une estimation quand une valeur hardware existe.**

xdna-top insiste : pas de "pourcentage d'utilisation générique inventé" — il exploite les
compteurs réels de soumissions/completions + interfaces AMDXDNA. Même philosophie que
profiler-v3 : mesurer d'abord, dériver ensuite.

## 4. ARCHITECTURE À AJOUTER À PROFILER-V3 (4 adaptateurs, pas de réécriture)

```
profiler-v3/
├── collectors/
│   ├── xrt_sdt          # événements XRT SDT (submit/complete/context/partition)
│   ├── amdxdna_trace    # tracepoints amdxdna_trace:*
│   ├── amdxdna_telemetry# QUERY_TELEMETRY/SENSORS/HW_CONTEXTS/CLOCK_METADATA
│   └── perf             # perf record/script → timestamps
│
├── correlation/
│   ├── timestamps       # alignement multi-source
│   ├── submit_complete  # couple submit→complete
│   ├── dma_kernel       # DMA → kernel → completion
│   └── context          # partition/context mapping
│
├── hardware/
│   ├── npu  cpu  ddr  pcie  nv_gpu  ssd
│
└── oracle/
    └── d2               # D2 Adaptive Precision & Residency Planner
```

### Pipeline de corrélation (le critical-path timeline du D2)
```
CPU
   │
   ├──────────────┐
   ▼              │
XRT submit        │
   │              │
   ▼              │
NPU queue         │
   │              │
   ▼              │
DMA ─────────────►│
   │              │
   ▼              │
kernel            │
   │              │
   ▼              │
completion ───────┘
```
→ produit l'event stream pour le D2 :
```
event { timestamp, duration, pid, context, partition, command, source }
```

## 5. ANGLE MORT — VERSIONING DE LA PILE (à enregistrer dans CHAQUE trace)

Le couple kernel/driver/XRT/firmware change les ioctls de télémétrie disponibles (issue AMD :
driver mainline sans ioctls attendus par un SHIM XRT plus récent). Chaque profil DOIT porter :
```
kernel_version · amdxdna_version · firmware_version · XRT_version · driver_git
```
Sinon on compare deux profils sur des chemins hardware/software différents (même piège que
le protocole canonique AGENTS.md : cold/warm, thermal, caps).

## 6. POSITIONNEMENT AI ANALYZER (source de validation NPU, PAS l'architecture)

- AMD AI Analyzer 1.8 : BF16 uniquement → utile pour valider le profil NPU BF16 de référence,
  pas pour étudier Q2/Q3/Q4/Q5/Q6/Q8/IQ/FP8/FP4/NVFP4 ni le paging/PCIe/RTX.
- → garder en référence, ne pas mettre au centre.

## 7. SOURCES (URLs — à ajouter au URLS_REGISTRY.md)
| Source | URL |
|---|---|
| AMD AI Analyzer 1.8 | https://ryzenai.docs.amd.com/en/main/ai_analyzer.html (rechercher exacte) |
| AMD XDNA Driver | https://github.com/amd/xdna-driver (npu_perf_trace.sh, telemetry) |
| amdxdna telemetry UAPI | https://github.com/amd/xdna-driver/blob/main/src/include/uapi/drm_local/amdxdna_accel.h |
| xdna-top | https://github.com/Glabby000/xdna-top (rechercher exacte) |
| ryzenai-lab | https://github.com/ryzenai-lab (rechercher exacte) |
| xdna-engine | https://github.com/ryanhnr/xdna-engine (rechercher exacte) |
| AIE/XDNA doc (MERT counters) | https://www.kernel.org/doc/html/latest/accel/amdxdna/amdnpu.html |

## 8. LIEN AVEC LES AUTRES DOCS
- MATRICE_EXPERIMENTALE_P0_P5.md : P0.2 (hardware throughput matrix) consomme ces compteurs
- BLINDSPOTS_REPONSES.md : #38-43 (XDNA DMA/sync, chemin complet), #72-73 (BW temporelle,
  température/fréquence) — alimentés par collectors/telemetry
- URLS_REGISTRY.md : ajouter la section AMD profiling (ce doc)