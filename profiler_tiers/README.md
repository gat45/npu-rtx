# profiler_tiers/ — Profiler GPU-tier, par COPIE (sources existantes intactes)

**Cible = 5070 laptop 8 Go** (machine cible du plan 5070+XDNA2). La GTX 1080 8 Go
(machine dev) sert uniquement de banc de validation de logique — ses t/s ne
prédisent pas la 5070 (Blackwell/GDDR7).

> Périmètre : tier GPU x86 **exclusivement**. Le OnePlus 15 (HTP/GenieX) n'est
> pas concerné par ce bloc.

## Fichiers

| Fichier | Rôle | Origine |
|---|---|---|
| `gpu_tier_profiler.py` | Moteur : profils machine 5070/1080, goulots PCIe + bus mémoire, modes `raw`/`plan`/`bench` | **nouveau** (calibré sur ancres mesurées) |
| `profile_model.py` | Lecture header GGUF + parsing tenseurs (retour `(meta, tensors)`) | **copie** de `profiler_v4/` (inchangée) |
| `SORTIE_5070.txt` | Profil 5070 : raw GGUF réel + plan 35B-A3B | généré |
| `SORTIE_1080.txt` | Profil 1080 : mêmes modes (banc de validation) | généré |

## Usage

```bash
cd geniex_harness/npu-rtx/profiler_tiers

# Profilage RAW d'un GGUF local (fit VRAM, KV f16 estimé, métadonnées MoE)
py gpu_tier_profiler.py raw --machine 5070 --gguf <chemin.gguf> [--ctx 8192]

# PLAN : placement MoE + goulots (KV f16|turbo4|turbo3|q8_0)
py gpu_tier_profiler.py plan --machine 5070 --model 35b --kv turbo4 --experts 64 --ctx 8192
py gpu_tier_profiler.py plan --machine 5070 --model marco8b --kv q8_0 --experts 0

# Bench réel (si llama.cpp buildé) : pp512/tg128 + seuil t/s
py gpu_tier_profiler.py bench --machine 5070 --bin <llama-bench> --model <f.gguf>
```

## Profils machine

| Paramètre | **5070 laptop 8 Go** (cible) | GTX 1080 8 Go (dev) |
|---|---|---|
| Bus mémoire théo | 384 GB/s (GDDR7 128-bit) | 320 GB/s (GDDR5X 256-bit) |
| BW effective (decode LLM) | **~165 GB/s** (51 % — roofline FLM 9B recalé 5070, Phase 1) | 205 GB/s (×0.64, ancre Qwen9B IQ4NL = 32.9 t/s) |
| PCIe | **gen5 x8 = 31.5 GB/s** (goulot D2/FATE streaming experts) | gen3 x16 = 15.75 GB/s |
| Budget VRAM utile | 7.36 GiB (garde 8 %) | 7.36 GiB |

## Goulots modélisés (dans l'ordre d'impact)

1. **PCIe (streaming experts, MoE offload)** : octets d'experts manquants / token ÷
   15.75 GB/s. Règle de garde : > 5 ms/token ⇒ placement injustifiable, réduire le
   working set d'experts ou monter la résidence.
2. **Bus mémoire (decode BW-bound)** : poids actifs + KV lus / token ÷ BW_eff.
   C'est le goulot #1 en decode full-résident.
3. **KV** : f16 vs turbo4 (×0.258 validé Phase 2, texte ≈ f16) — mais **K=turbo3
   est INTERDIT** (bug upstream Phase 2 : garbage CPU ET CUDA, set_rows sain).
   V=turbo3/turbo4 et K=turbo4 OK.

## Ancres (provenance)

- 1080 : Qwen3.5-9B IQ4NL = **32.9 t/s** decode (mesuré, sessions précédentes).
- 5070 : BW 384 GB/s et PCIe x8 G4 d'après `CORRECTIONS_MATERIEL (5070 Laptop 384 GB/s, Strix Point 50 TOPS)` (npu-rtx, commit `47458ae`).
- Overflow NPU = ×16 hit GPU (Phase 1, `benchmark_sim.py`) — NPU = tier d'overflow, jamais backend principal.
- KV turbo4 ×0.258 vs f16 + validation texte ≈ f16 (Phase 2, `RAPPORT_PHASE2_TURBO_1080_20260921.md`).
