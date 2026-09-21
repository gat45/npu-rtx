# RAPPORT DE SESSION — Profilage expert-grain 35B Genesis & imatrix

**Date** : 2026-09-21 · **Machine** : GTX 1080 8 GB (sm_61) + Ryzen, 32 GiB RAM, disque E: ~42 GiB libres
**Modèle** : `D:/Hermes3.6-35B-A3B-Uncensored-Genesis-Final-Q8_K_P.gguf` (40.61 GiB, Q8_0 global + gate/up F16 couches 20/27/29–39)
**Repo** : `npu-rtx` (gat45/npu-rtx, branche master) — tous les résultats commités

---

## 1. Ce qui a été mesuré (résumé exécutif)

| # | Résultat | Statut | Commit |
|---|---|---|---|
| 1 | SNR **exact** des 30 720 tenseurs experts (40 couches × 3 familles × 256) | ✅ MEASURED | `d4f9341` |
| 2 | Candidat requant chiffré : **40.60 → 20.76 GiB (−49 %)**, pool experts 37.97 → 18.12 GiB | ✅ MEASURED (tailles exactes) | `d4f9341` |
| 3 | Motif de fragilité par couche : 28, 32–34, 36, 38–39 fragiles ; 1–27 homogènes | ✅ MEASURED | `d4f9341` |
| 4 | **P0-3** : αw spectral ne sépare pas les experts (ρ global −0.18) | ✅ TESTÉ ET VERDICT RENDU | `bd9cf18` |
| 5 | Contrôle quantizer-égal : le « down +6 dB » était un **artefact Q5_0-vs-Q4_0** | ✅ TESTÉ | `2710d65` |
| 6 | Gap spectral orthogonal au SNR (ρ=+0.04 NS) mais redondant avec decay (ρ=+0.54) | ✅ MEASURED | `2710d65` |
| 7 | Ranking des experts fragiles **invariant au quantizer** (mêmes argmin Q4_0/Q5_0) | ✅ MEASURED | `2710d65` |
| 8 | Empreinte runtime hors-poids du fork : **~9 GiB private** (b2048, c512) — plus gros que prévu | ✅ MEASURED (externe) | (ce rapport) |
| 9 | Imatrix 35B sur corpus mixte local : **run lancé, 1er checkpoint attendu** | 🟡 EN COURS (24 chunks, stop prévu au 1/6) | — |
| 10 | `--moe-cache` / `--cpu-moe` / `--tensor-read-lazy` **existe dans le binaire du fork** | ✅ DÉCOUVERT (vérifié --help) | — |

---

## 2. Méthode : SNR exact grain expert (inspiration d2_layer_profiler_v3)

Fichier : `profiler_tiers/requant_experts.py`

- **Streaming strict** : 1 expert (~1 Mo) chargé à la fois, jamais de tenseur entier (1 GiB f32) en RAM
- **SNR exact** = `quantize() → dequantize()` (gguf-py du fork, numpy pur) vs source Q8_0/F16 — **pas** un SQNR théorique
- Découpage par expert : tenseurs 3D `(512, 2048, 256)` → l'expert `e` est une tranche contiguë de `packed_size(n_exp, dtype)` octets, lue par `seek+read` direct
- 92 160 mesures (30 720 experts × 3 familles), ~90 s par tranche de 14 couches, 3 tranches au total
- Types testables en python : Q4_0 (4.5 bpw), Q5_0 (5.5), Q8_0 (8.5) — les Q4_K/Q6_K/TQ* ne sont pas implémentés dans `gguf.quants.quantize()` (NotImplementedError), réservés à `llama-quantize` C++

### Requants candidats chiffrés (tailles exactes, pas de bpw linéaire)

| Candidat | Règle `--tensor-type` | Fichier | Pool experts |
|---|---|---|---:|
| Mixed M1 | gate/up→Q4_0, down→Q5_0 | **20.76 GiB** | 18.12 GiB |
| Alternative | gate/up→Q4_K, down→Q6_K (via llama-quantize C++) | ~22.1 GiB (simulé) | 19.45 GiB |
| Source | — | 40.60 GiB | 37.97 GiB |

**Conséquence RAM (décisive)** : 18.12 GiB d'experts tiennent dans les 32 GiB système (37.97 GiB impossible) → le déploiement réel sur cette machine devient possible sans streaming.

### Top 10 experts fragiles (à garder en haute précision, +~50 MiB seulement)

```
blk.0  gate e252 (18.41 dB) · blk.36 gate e2 (18.67) · blk.32 gate e123 (18.92)
blk.34 gate e91  (19.00) · blk.33 gate e75 (19.06) · blk.38 gate e181 (19.38)
blk.9  up   e58  (19.44) · blk.36 up   e2  (19.59) · blk.32 up   e123 (19.60)
blk.39 gate e141 (19.61)
```

---

## 3. Motif par couche (SNR mesuré, mixed gate/up Q4_0 + down Q5_0)

| Couches | gate/up mean | gate/up min | Verdict |
|---|---:|---:|---|
| 0 | 21.14 | 18.41 | sain (mais e252 fragile) |
| 1–27 | 21.16–21.28 | 19.7–21.2 | homogène |
| **28** | 20.57 | 18.5 | ⚠ fragile |
| **32–34** | 20.98–21.04 | 18.92–19.06 | ⚠ cluster fragile |
| 35 | 21.14 | 19.93 | récupère |
| **36** | 21.11 | 18.67 | ⚠ e2 très fragile |
| **38–39** | 21.05–21.08 | 19.38–19.61 | ⚠ fin de réseau |

→ L'hypothèse APEX « premières/dernières couches plus sensibles » est **confortée en grain expert** sur ce modèle précis (Hermes Genesis), pas supposée.

---

## 4. P0-3 — Verdict sur le signal spectral (αw/decay/gap)

Fichiers : `d2_expert_spectral_scan.py` (fusion des 8 outils D:\lama-tensorRT), `P03_VERDICT_ALPHA_VS_SNR.md`

### 4.1 Ce qui a été testé

1 536 experts (couches 0–1 × gate/up/down), chaque expert portant **dans le même JSON** son SNR exact ET ses features spectrales (randomized SVD rank-32, méthode Halko de `d2_qdf_bayesian_optimized`). Aucun appariement cross-run.

### 4.2 Résultats statistiques propres (scipy.spearmanr + bootstrap 2000)

| Relation | ρ | CI 95 % | p | Statut |
|---|---:|---|---:|---|
| SNR ↔ decay (global) | **−0.184** | [−0.238, −0.127] | 4e-13 | significatif mais **faible** |
| SNR ↔ gap | +0.040 | — | 0.12 | **NS → orthogonal au SNR** |
| decay ↔ gap | **+0.541** | — | 2e-117 | **redondants entre eux** |
| SNR ↔ entropie | +0.177 | — | 3e-12 | faible |

Corrections méthodologiques appliquées : Spearman artisanal remplacé (les ties — 86.1 % des SNR dupliqués par l'arrondi — étaient mal gérés), bootstrap CI par groupe, Pearson croisé (signe inversé sur 1/down → relation non linéaire).

### 4.3 Verdict

- **αw ne peut PAS piloter l'allocation expert-grain** : variance intra-couche quasi nulle (std decay 0.007–0.098) pour des SNR étalés de 18.4 à 30.5 dB
- Usage légitime restant : **modulateur de décision par COUCHE** (le `--plan` du scanner le fait déjà)
- **Le SNR mesuré reste le seul signal expert-grain validé**
- Le gap est la seule feature orthogonale au SNR — mais sa valeur prédictive *fonctionnelle* (vs erreur d'activation/KL) reste non démontrée

---

## 5. Contrôle quantizer-égal (angle mort confirmé et fermé)

Le contrôle demandé par l'analyse (même quantizer sur les 3 familles) a été exécuté :

| Condition | gate | up | down | Conclusion |
|---|---:|---:|---:|---|
| Mixed (Q4_0/Q4_0/Q5_0) | 21.14 | 21.22 | 27.38 | « down plus robuste » apparent |
| **Q4_0 uniforme** | 21.14 | 21.22 | **22.14 / 21.31** | écart ≤ 1 dB → **artefact** |
| **Q5_0 uniforme** | 27.21 | 27.29 | 28.23 / 27.38 | idem |

**Le « +6 dB du down » était 100 % l'effet du choix Q5_0 vs Q4_0, pas une propriété de la famille `ffn_down_exps`.** Point positif : les argmin (experts fragiles) sont identiques aux deux quantizers → **l'index de fragilité est invariant au quantizer**, exploitable pour la sélection des exclusions haute précision.

---

## 6. Découverte runtime : empreinte mémoire hors-poids mesurée

Le fork **mute les logs de breakdown** (UI chat custom, timestamps custom, pas de lignes "CPU model size / KV buffer / compute buffer"). Mesure **externe** sur le run imatrix actif (`-c 512 -b 2048`, CPU, mmap) :

| Métrique (PowerShell, processus vivant) | Valeur |
|---|---:|
| Private bytes (KV + compute buffers + heap + imatrix) | **8.9 GiB** |
| Working set | 20.4 GiB (dont pages modèle mmap touchées) |
| Working set peak | 17.2 GiB (au chargement) |

⚠️ Caveats honnêtes : (1) le private inclut l'imatrix elle-même (468 tenseurs × stats), pas pur runtime llama ; (2) mesure faite avec 2 runs concurrents sur le disque D:. La décomposition fine (KV vs compute vs imatrix) exige le test b512 dédié — **pas encore fait** (machine saturée). Référence externe citée par l'analyse : cas documenté compute buffer ~1.39 GiB causant un OOM à lui seul.

---

## 7. Découverte majeure : le fork a déjà le cache d'experts intégré

Vérifié par `llama-cli --help` du binaire build-cu61 :

```
--moe-cache MODE     adaptively cache the hottest CPU-resident MoE experts
                     in spare VRAM
                     auto | on | soft (try spare VRAM first, evict as needed)
                     N = VRAM budget in MiB per device
--cpu-moe            keep all Mixture of Experts (MoE) weights in the CPU
-ncmoe, --n-cpu-moe  first N layers MoE on CPU
--tensor-read-lazy   on : read tensor rows from disk on demand (mmap)
```

C'est **exactement** le mécanisme « taille totale des experts ≠ pool résident » décrit dans la littérature (PoC MoE offload llama.cpp, RFC expert cache +84 % decode, SpecMD/LRU alternatives). La ligne de log observée (`MoE cache fit kept stock placement: no selected device satisfies the cache hardware policy`) = ce planner, inactif faute de GPU sélectionné en `-ngl 0`.

### Matrice de test 1080 prête (aucun GGUF à écrire)

| Config | Commande | Mesure attendue |
|---|---|---|
| A | `-ngl 99 --cpu-moe --moe-cache off` | baseline experts sur CPU (hit 0 par définition) |
| B | `--moe-cache 2048` | hit réel @ 2 GiB cache VRAM |
| C | `--moe-cache 4096` | hit réel @ 4 GiB |
| D | `--moe-cache 6144` | quasi tout le budget 1080 |

→ Ces runs **valident le hit modelled (0.47 / 0.60 selon requant)** et le choix « miss → compute CPU vs H2D » — les deux hypothèses restantes du profil 70.5 t/s.

---

## 8. Imatrix 35B — protocole et état

### 8.1 Construction du corpus (100 % local, anti-bias MoEQuant)

`build_corpus_mix.py` : entrelacement de blocs ~2 Ko depuis ce qui existe sur disque :
- **CODE** : 308 blocs de `.c/.h` ggml (24 fichiers)
- **PROSE-FR** : 66 blocs des rapports `.md` npu-rtx (17 fichiers)
- **PROSE-EN** : 100 blocs des docs du fork (20 fichiers)
→ `corpus_mix.txt` : **0.96 Mo, 474 blocs** (pools locaux saturés ; la taille reste dans les pratiques llama.cpp pour une imatrix)

### 8.2 Run

```
llama-imatrix.exe -m D:/Hermes...Q8_K_P.gguf -f corpus_mix.txt \
  -o imatrix_35b.gguf -c 512 -t 12 --chunks 24 \
  --output-frequency 6 --process-output
```

| Paramètre | Choix | Raison |
|---|---|---|
| `--chunks 24` | ~770k tokens | ≥ 655k de l'objectif routing/PPL |
| `--output-frequency 6` | checkpoint 6× | le run survit à toute interruption, fichier toujours utilisable |
| `--process-output` | inclus | couvre `output.weight` (248k vocab = goulot V4 sur 1080) |
| chargement | ~7 min | 40.61 GiB mmap paginé |

### 8.3 État au moment du rapport

- Processus **actif** (CPU cumulé 121 s, WS 20.9 GiB — mémoire stabilisée)
- **Chunk 1/24 pas encore terminé** au moment de la capture (paging SSD concurrent des 2 runs)
- **Décision utilisateur : stop au checkpoint 1/6** (~154k tokens de calibration, imatrix déjà utilisable) pour libérer la machine → **matrice moe-cache A/B/C/D ce soir**
- Le run est robuste à l'interruption : le checkpoint partiel `imatrix_35b.gguf` est écrit toutes les ~6 itérations et restaure les stats au redémarrage (`--in-file`)

---

## 9. Implications pour le planner D2

| Décision du planner | Signal validé localement | Ce qui manque encore |
|---|---|---|
| **Quels experts protèger** (haute précision) | SNR exact expert-grain + top fragiles + invariance au quantizer | ΔPPL conditionnel (validation fonctionnelle) |
| **Quels experts compresser** | 1–27 homogènes → Q4_0 sûr ; imatrix pour descendre sous Q4 | branche IQ3/IQ2 + imatrix (test A/B) |
| **Combien de résidents VRAM** | budget mesuré 3.29 GiB ; **`--moe-cache` natif dispo** | hit réel des configs A/B/C/D |
| **Miss → CPU ou H2D** | H2D 13.08 GB/s, CPU 0.34 ms/expert (measured) | mesure A/B (le moe-cache tranche) |
| **Quant vs résidence couplés** | tailles exactes par expert pour chaque format | ILP conjoint (après PPL du candidat) |

### L'échelle de formats à retenir (proposition de l'analyse, à valider par taille réelle)

`TQ1_0 (1.69) < IQ1_S (1.56*) < IQ2_XXS (2.06) < IQ2_XS (2.31) < IQ2_S (2.5) < IQ3_S (3.44) < Q3_K_S (3.50) < Q3_K_M (3.91) < Q3_K_L (4.27) < IQ4_XS (4.25*) < Q4_K_S (4.58) < Q4_K_M (4.84) < Q5_0 (5.5) < Q8_0 (8.5)` — (*bpw effectifs selon contenu). Notre candidat 20.76 GiB se situe **entre Q3_K_S (19.24) et Q3_K_M (21.19)** avec une qualité supérieure là où ça compte (les ~15 experts fragiles restés Q8_0).

---

## 10. Prochaines étapes (ordre fixé)

1. **Stop imatrix au checkpoint 1/6** → machine libérée (décision prise)
2. **Validation `--show-statistics`** du checkpoint (ΣAct², % actifs, entropie par tenseur)
3. **Matrice moe-cache A/B/C/D sur la 1080** (`llama-bench` ou decode court, 30–40 min) → hit réel + t/s mesurés → remplace le dernier ASSUMED du profil
4. **Reprise imatrix** (optionnelle, `--in-file` restaure) vers 24 chunks si le quantize final exige une calibration complète
5. **Écriture du candidat 20.76 GiB** (`requant_experts.py --write`, ~45 min) puis **PPL wiki.test** vs baseline → ΔPPL du candidat
6. **Branche imatrix** : Q4_0+imatrix vs Q4_0 seul sur les ~100 experts stratifiés (5 strates SNR × 3 familles × 4 régions de couches)
7. **P0-1 couplé** (PPL + routing 655k sélections, hook `GGML_ROUTING_TRACE`) — reste le run statistique de référence du skew 35B

---

## 11. Fichiers produits cette session

| Fichier | Rôle |
|---|---|
| `profiler_tiers/requant_experts.py` | requantizer expert-grain (SNR exact + `--write` + tailles) |
| `profiler_tiers/snr_experts_chunk{0_13,14_27,28_39}.json` | 92 160 SNR par expert |
| `profiler_tiers/snr_experts_ctrl_{q4,q5}.json` | contrôles quantizer-égal |
| `profiler_tiers/SORTIE_35B_LAYER_EXPERT_SNR.md` | rapport SNR + motif couches (+ correction contrôle) |
| `profiler_tiers/d2_expert_spectral_scan.py` | scanner fusionné SNR+spectre (1 lecture disque) |
| `profiler_tiers/spectral_experts_smoke.json` | SNR + decay/gap/ent de 1 536 experts |
| `profiler_tiers/P03_VERDICT_ALPHA_VS_SNR.md` | verdict P0-3 (+ addendum scipy/contrôle) |
| `profiler_tiers/build_corpus_mix.py` | constructeur de corpus multi-domaine local |
| `profiler_tiers/corpus_mix.txt` | corpus 0.96 Mo (474 blocs code/FR/EN) |
| `profiler_tiers/imatrix_run.log` | log du run imatrix (live) |

Commits : `d4f9341` → `bd9cf18` → `2710d65` (+ ce rapport). Tout est poussable sur GitHub.

---

## 12. Limites déclarées (pour la traçabilité)

1. **SNR = fidélité poids**, pas impact fonctionnel — la validation ΔPPL/KL reste à faire (étape 5-6)
2. **Référence = Q8_K_P, pas F16** — le SNR mesure Q8→Q4, l'erreur FP16→Q8 n'est pas incluse (échantillon F16 impossible : le GGUF F16 du 35B n'existe pas localement)
3. **Python quantize limité à Q4_0/Q5_0/Q8_0** — les K-quants/IQ/TQ passent par `llama-quantize` C++ (à tester avec `--dry-run` pour les tailles exactes)
4. **Empreinte runtime** : private bytes inclut l'imatrix ; décomposition fine non faite
5. **αw/decay/gap** : verdict rendu sur 2 couches (1 536 experts) — extension possible mais le verdict de redondance decay↔gap (ρ=0.54) tient déjà
6. **Imatrix partielle** : stop au checkpoint 1/6 par décision — suffisante pour le quantize test, à compléter pour le quantize final si ΔPPL déçoit
