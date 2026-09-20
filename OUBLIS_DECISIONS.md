# OUBLIS & DÉCISIONS — mapping expert→tiles, conversions, shared expert, imatrix, MTP, énergie
# Établi 2026-09-20 · Dossier npu-rtx/ · Complète VERIFICATION_NPU_PREFERENCES + STATIC_ORACLE

---

## 1. MAPPING D'UN EXPERT 2560×640 VERS LES GÉOMÉTRIES XDNA2 (le point à trouver)

### Le format Q4NX / reorder (FLM_SECRETS) — la brique existante
```
row_chunk_size = 32 · col_chunk_size = 256 · block = 5120 B (32×256 tile)
RTP registers : 0x1000 = tile M · 0x1004 = tile K · 0x1008 = tile N
shim_tiles[8] = IT0..IT7 (une colonne = un shim DMA)
```
→ Le layout Q4NX est DÉJÀ un tiling 32×256 pour le DMA NPU. C'est le point d'ancrage.

### Mapping d'un expert Qwen (gate [2560,640,512], up, down [640,2560,512])
Un expert = [2560, 640] (gate/up) ou [640, 2560] (down).

| Tenseur | Shape | Rows | Cols | Tiles 32×256 (Q4NX-style) |
|---|---|---|---|---|
| gate | 2560×640 | 2560 | 640 | 80 × 2.5 → 80 × 3 (padding 128 cols) |
| up | 2560×640 | 2560 | 640 | 80 × 3 |
| down | 640×2560 | 640 | 2560 | 20 × 10 |

→ **coût DMA par expert** : (80×3 + 80×3 + 20×10) tiles × 5120 B ≈ (240+240+200)×5120 ≈ 3.48 MB
en Q4NX-layout (contre 2.64 MiB théorique — l'overhead de padding + scales explique l'écart).

### Contrainte INT8 (DESCENT) : 64 KB core-local → tuiles recommandées
Pour INT8, le microkernel `mmul` 8x8x8 (512 MACs/cyc/tile) :
- tile M/N/K à choisir pour tenir dans 64 KB L1 par core (A+B+C)
- example IRON : 2x2 (M,N tuiles par core), tile_k=128 limite
- 2560 = 320 × 8 → aligné sur 8 (bon pour 8x8x8) ; 640 = 80 × 8 → OK ; **640 et 2560 sont
  multiples de 8** → pas de pénalité d'alignement (contrairement à 3584 du Qwen dense).

### Décision P0 : mapping à implémenter
```
static/expert_mapper.py :
  expert (2560×640×3) → liste de (tile_M, tile_K, tile_N, offset_GGUF, kernel mmul mode)
  contrainte : L1 64KB, 8x8x8 int8, row_chunk 32 / col_chunk 256 (Q4NX legacy)
  sortie : table par expert → le cost model utilise ces tiles, pas l'expert entier
```

## 2. CONVERSION NVFP4→INT8 (les 4 chemins à benchmarker — oublié du plan)

Le stockage est NVFP4 (ou Q4), mais XDNA2 exécute INT8. Il faut comparer :
```
A: NVFP4 → RTX (pas de conversion, GEMM NVFP4)
B: NVFP4 → INT8 → XDNA2   (conversion + GEMM INT8)
C: BF16 → INT8 → XDNA2    (conversion depuis maître)
D: BF16 → BFP16 → XDNA2   (conversion + GEMM BFP16)
```
Coût à mesurer (par tensor/expert) : conversion_time, DDR/PCIe extra, buffer temporaire,
quality_delta. **NVFP4 (4-bit + scales) ≠ INT8 (8-bit + scales)** → la conversion n'est pas
gratuite (point 17 de l'analyse D2). C'est le **premier vrai test D2** à faire.

## 3. SHARED EXPERT (oublié dans le static oracle)

config.json : `shared_expert_intermediate_size = 640` → shared = même taille qu'un routed
(4 915 200 params/couche).
```
48 layers × 4 915 200 = 235 929 600 params (0.44 GiB BF16 / 0.15 GiB Q4)
```
→ **le shared expert = dense-hot** : toujours actif → résident permanent (VRAM), jamais évincé.
À traiter comme un tensor dense, PAS comme un expert routé (point 51 BLINDSPOTS).

## 4. IMPORTANCE MATRIX (oublié : imatrix par expert)

- Une imatrix globale ne couvre pas 512 experts × 48 couches.
- À tester : global / layer / expert / tensor / workload-specific (point 25 MATRICE).
- ⚠️ collecte CPU-side coûteuse à grande échelle.
- **D2 : la sensibilité par expert doit venir de la calibration, pas de l'imatrix seule**
  (hotness ≠ sensitivity).

## 5. MTP ROLLBACK (oublié dans le modèle de coût)

Qwen3.8 = MTP natif (4B). Le pipeline spec : draft → verify → **rollback**.
```
MTP affecte : expert cache, GDN state, QSA KV, workspace
rollback : cache pollution si le draft a préchargé des experts non utilisés
```
→ D2 doit comparer MTP off/1/2/3 APRÈS cache stable (point 62 BLINDSPOTS). Les expériences
publiques (Weschera) : MTP = 2× code, 0% prose, négatif si draft long.

## 6. ÉNERGIE (oublié — mesurer si possible)

- J/token, J/request, J/GB transféré quand mesurable (corpus FLM : NPU 2W vs GPU, 46 tok/s/W).
- Sur la machine cible : NPU power, GPU power, CPU power (telemetry AMD + nvidia-smi).
- L'optimum énergie peut DIFFÉRER de l'optimum latence (le NPU est éco, le RTX est rapide).

## 7. GDN STATE / QSA KV — dimensions réelles (complément STATIC_ORACLE)

```
GDN state : 48 V-heads × 16 QK-heads × 128 head_dim → matrice [.,.,128,128] par couche
  (corpus FLM : état SSM [32,128,128] par couche GDN)
QSA KV : 12 couches QSA, 2 KV heads, head_dim 256 → KV bytes/token = 12 × 2 × 256 × 2 (BF16)
  = ~12 288 B/token (QSA seul) ; total avec GDN compressé ≈ 30 000 B/token (mesuré FLM)
```
→ à injecter dans le budget VRAM 6 composants (dense/GDN_state/QSA_KV/workspace/PLE/cache).

## 8. LISTE DES OUBLIS → STATUT

| Oubli | Statut | Doc |
|---|---|---|
| Mapping expert→tiles XDNA2 | 🔴 à implémenter (P0) | ce doc §1 |
| Conversion NVFP4→INT8 (A/B/C/D) | 🔴 premier test D2 | ce doc §2 |
| Shared expert = dense-hot | 🔴 à traiter séparé | ce doc §3 |
| imatrix par expert | 🔴 à tester | ce doc §4 |
| MTP rollback | 🔴 après cache stable | ce doc §5 |
| Énergie J/token | 🔴 si telemetry | ce doc §6 |
| GDN state / QSA KV dims | 🟢 injectées | ce doc §7 |
| V5 = Snapdragon (pas XDNA2) | ✅ | PLAN §16 |
| INT8-XDNA2 = P0 | ✅ | VERIFICATION_NPU_PREFERENCES |

## 9. MISE À JOUR DES AUTRES DOCS (faites)
- **VERIFICATION_NPU_PREFERENCES.md** : INT8 6.65-8.69 TOPS, BFP16 4.64, BF16 émulé ¼
- **STATIC_ORACLE_QWEN38.md** : tailles expert/bytes-token/PLE/cache/lower bounds
- **PLAN_CAMPAGNE_PARETO.md** : V5=Snapdragon, plan reproduction R1-R3
- **REFERENCE_RESULTATS_PUBLICS_QWEN38.md** : carte résultats publics
- **URLS_REGISTRY.md** : sources Qwen + AMD + profiling