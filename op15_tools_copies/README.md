# op15_tools_copies/ — copies OP15, MORTES pour le flux 5070/XDNA2

⚠️ **Ces fichiers proviennent de `geniex_harness/profiler_v4/` (projet OnePlus 15,
Snapdragon/Hexagon/HTP/Adreno). Ils ne servent PAS le pipeline 5070/XDNA2** —
copiés ici le 2026-09-21 à la demande « tu ne modifies pas les sources, tu copies »,
déplacés ensuite pour étiqueter clairement leur périmètre (matériels distincts).

| Fichier | Origine OP15 | Pourquoi mort ici |
|---|---|---|
| `parse_hexagon_profile.py` | parser traces Hexagon/HTP (DSP) | le HX 365 n'a pas de Hexagon |
| `parse_opencl_profile.py` | parser profiling OpenCL **Adreno** | GPU cible = CUDA (5070), pas Adreno |
| `android_telemetry.py` | télémétrie adb Android | la cible est un PC x86 Windows |
| `live_monitor.py` | monitoring live device OP15 | idem — pas d'appareil Android dans le flux 5070 |
| `telemetry_dashboard.py` | dashboard télémétrie OP15 | idem |
| `adaptive_lever.py` | leviers adaptatifs runtime OP15 | logique couplée aux compteurs OP15 |

`capability_db.py` a été **remis dans `profiler_tiers/`** : `predict_from_hf*.py`
et `profiler.py` l'importent (prédiction de risque par GPU, partie réutilisable).

Le flux 5070 actif vit dans `profiler_tiers/` : `gpu_tier_profiler.py`,
`moe_axis_profiler.py`, `critical_path_5070.py`, `microbench_h2d.py`,
`profile_model.py`, `predictor*.py`, `capability_db.py`, `predict_from_hf*.py`.
