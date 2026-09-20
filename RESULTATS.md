place_expert importé OK → E43 / 0.7 fréq / 100MB → tier XDNA2 (conforme plan).
Contention patch prêt (overlay cost_model).
Validation et benchmark : scripts créés, blocage = absence build XDNA2 + device dans env actuel.
Prochain : sur machine cible, exécuter RUN_PLAN.md + build xdna2-forensics/llama-upstream (NDK r27c, -GGML_HEXAGON=OFF).

2026-09-20 — sources AMD vérifiées (fetch) :
- OGA 1.8.0 : mode Hybrid NPU+iGPU OFFICIEL (Strix/Krackan Point) → RAPPORT corrigé (§3bis)
- UAPI amdxdna_accel.h : BO types (SHARE/DEV_HEAP/DEV/CMD), SYNC_DIRECT_TO/FROM_DEVICE, QoS hints, telemetry (power/temp/col_util) → reference/AMD_OGA_HYBRID_OFFICIEL.md + AMDXDNA_DRIVER_UAPI.md
- 1bit-MONSTER : engine C++26 model-agnostic NPU/GPU/CPU, format 1BP → reference/1BIT_MONSTER.md
- MANQUANT complété (points 11-12 : UAPI Windows à vérifier, mapping API→code absent)
