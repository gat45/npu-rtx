# Phase 2 — Validation kernels turbo3/turbo4 (GTX 1080, sm_61) — 2026-09-21

## Verdict en une ligne
Le fork `llama-cpp-turboquant` (build 4deec55) **compile et tourne en sm_61** après correctifs
de shared-memory ; **turbo4 est validé end-to-end**, mais un **bug upstream majeur** est trouvé :
**toute lecture de K en turbo3 produit du garbage** (CPU et CUDA, FA et MUL_MAT) — preuves ci-dessous.

## 0. Correction de périmètre (pivot)
`apply_turbo_cuda_v2.py` (patch fossile de la session 20/09) est **obsolète** : il cible un layout
turbo4 legacy (3-bit+QJL, 68 B) que le fork a abandonné (`TURBO4_USE_4BIT=1` → nibble 4-bit, 66 B),
et crée des doublons de symboles avec la stack CUDA **canonique** du fork (`turbo-quant.cuh`,
set-rows/getrows/fattn intégrés, "Metal + CUDA validated"). Il est conservé comme doc d'intention
uniquement. La validation porte sur l'implémentation canonique du fork.

Différence clé vs patch : le fork applique une **rotation WHT** (signes seed=42, butterfly,
×1/√128) avant quantization — côté set_rows CUDA ET côté quantizer CPU ref — avec pré-rotation
de Q dans le graphe (`ggml_turbo_wht`, gate sur `k->type` turbo2/3/4). Le dot K·Q reste valide
car WHT ⊗ WHT = I.

## 1. N0 — compile standalone (sm_61) : ✅
`turbo-quant.cuh` + `common.cuh` compilent à 0 erreur :
`nvcc -std=c++17 -arch=sm_61 -ccbin <cl 14.44 BuildTools> -Iggml/include -Iggml/src -Iggml/src/ggml-cuda`
(NB : `-std=c++17` requis — fold expressions ; MSVC = BuildTools 14.44, pas Community).

## 2. Build complet sm_61 : ✅ après correctifs shared-memory
- **Bug de portabilité réel** : les instances FA vec turbo en **D=512** dépassent la smem Pascal
  (0x10100 = 65 792 B > 0xc000 = 48 000 B). Le LUT turbo (8/4 centroïdes × D en half) est le
  facteur additionnel vs f16. Personne n'avait compilé D512+turbo pour sm_61 (Metal + GPU récents OK).
- **Correctif minimal** : retrait des instances `DECL_FATTN_VEC_CASE_D512` turbo (21 fichiers
  template-instances) + des 3 lignes de dispatch `FATTN_VEC_CASE_D512(...TURBO...)` dans `fattn.cu`.
  Les head-dims réels (64/128/256 : Qwen, Marco, gpt-oss…) ne sont pas touchés.
- Diff capturé : `npu-rtx/reference/patches/turboquant-sm61-build.patch` (43 lignes).
- Binaires produits : `build-cu61/bin/{llama-cli,llama-bench,test-backend-ops,test-turbo-quant}.exe`.
- Script reproductible : `npu-rtx/reference/build_cu61.bat` (vcvars64 + Ninja, GGML_CUDA=ON, arch 61).

## 3. N1 — validation numérique : ✅ (avec angle mort documenté)
- **`test-backend-ops` FLASH_ATTN_EXT `-p turbo`** : **117/117 OK, 0 FAIL** — toutes combos
  K/V (turbo2/3/4 × f16/q8_0/turbo*, mixtes), D=64/128/256, prec f32/def, softcap on/off.
- **`test-turbo-quant`** (round-trip CPU du fork) : turbo3 cos 0.986 / turbo4 cos 0.996 ;
  chunked==whole 25/25 (tq3_1s, tq4_1s, turbo2/3/4, k=128..4096).
- **Angle mort du harnais** : SET_ROWS/GET_ROWS ne génèrent **aucun** cas turbo (799 cas, tous
  f32/q8_0/…), et les cas FA turbo ont `Q->ne[1]=4` → `ncols>1` → le fast-path LUT (decode,
  ncols=1) n'est **jamais** testé. Couverture à compléter côté upstream.

## 4. N2 — end-to-end Marco-Nano-Instruct 8B-A0.6B Q4_0 (1080, -ngl 99, -fa on)
| KV | tg (t/s) | pp512 (t/s) | Sortie (temp 0) |
|---|---|---|---|
| f16/f16 | 102.7 | 3357 | ✅ correcte |
| q8_0/q8_0 | 94.3 | — | ✅ |
| **turbo4/turbo4** | 79.1 | 3232 | ✅ sémantiquement ≈ f16 |
| **turbo3/turbo3** | 78.3 | — | ❌ **garbage** |
| turbo3 K + q8_0 V | 84.6 | — | ❌ garbage (K seul en cause) |
| q8_0 K + turbo3 V | — | — | ✅ correcte |
| turbo4 K + q8_0 V | 83.6 | — | ✅ |
| q8_0 K + turbo4 V | — | — | ✅ |

Perf à contexte court : f16 reste le plus rapide sur 1080 (le gain BW de turbo ne rattrape pas
le coût de dequant générique sm_61). La valeur turbo = **capacité KV** (×4.9 / ×3.8 compression)
→ contexte long / VRAM contrainte, à rejouer sur 5070.

## 5. 🐛 FINDING MAJEUR — K=turbo3 cassé sur les DEUX backends (upstream)
Matrice de bissection (Marco-Nano, temp 0, -n 30..40) :
| Config | Résultat |
|---|---|
| CUDA, -fa on, K=turbo3 | garbage (`//1011111122…`) |
| CUDA, **-fa off** (chemin MUL_MAT), K=turbo3 | garbage |
| **CPU pur (-ngl 0)**, -fa on, K=turbo3 | garbage |
| CPU pur, K=q8_0 **V=turbo3** | ✅ texte correct |
| CPU pur, K=turbo4 V=turbo4 | ✅ texte correct |

Conclusions :
1. Le **set_rows turbo3 est sain** (V=turbo3 lit le même packing → texte correct).
2. La **lecture de K turbo3 est cassée de façon identique CPU et CUDA** — suspect : le chemin
   de déquant K turbo3 (MUL_MAT CPU + FA CUDA) diverge du packing set_rows (ex. inversion
   qs/signs, mapping WHT, ou offset `j+16` hérité de q5_0 ; le vec_dot CUDA FA a été vérifié
   cohérent octet par octet avec le packer → le bug serait partagé amont).
3. **Le LUT FA n'est pas la cause racine** (garbage identique LUT actif/désactivé ;
   garbages différents mais tous invalides → au moins 2 lecteurs K turbo3 faux, ou 1 faux + 1
   conséquence).
4. turbo4 n'est **pas** touché (chemin nibble distinct).

Expériences menées puis annulées (fattn-vec.cuh restauré à l'upstream) : fix hypothétique du
remplissage LUT (lecture Q half vs float — sans effet, branche non compilée en sm_61 :
`V_DOT2_F32_F16_AVAILABLE` indéfini sur Pascal), désactivation forcée du LUT (change le garbage,
ne le corrige pas).

Repro :
```bash
cd npu-rtx/reference/llama-cpp-turboquant/build-cu61/bin
./llama-cli -m Marco-Nano-Instruct.Q4_0.gguf -ngl 0 -fa on -ctk turbo3 -ctv q8_0 \
  -p "Explain in one sentence what photosynthesis is." -n 30 --temp 0 --single-turn
# -> garbage ; même commande avec -ctk q8_0 -ctv turbo3 -> texte correct
```

À faire : bissection ciblée du lecteur K turbo3 (CPU `ggml_compute_forward_mul_mat` +
déquant turbo3 ; comparer au lecteur V et au ref `dequantize_row_turbo3_0`) puis issue upstream.

## 6. Limites
- sm_61 valide la **logique**, pas la perf Blackwell (5070) — les t/s mesurés ne prédisent rien
  pour la machine cible.
- Le bug K=turbo3 est **indépendant de l'archi GPU** (présent en CPU pur) → il concerne aussi
  la 5070 ; turbo4 reste utilisable dès maintenant.
- Marco-Nano = 8B-A0.6B, contexte court ; mesure KV longue capacité non faite (1080 = 8 Go).

## 7. Artefacts
- `patches/turboquant-sm61-build.patch` — diff sm_61 minimal (D512 smem)
- `build_cu61.bat` — build reproductible
- `llama-cpp-turboquant/build-cu61/` — build complet (non commité)
- Logs N1 : `/tmp/n1_fa_full.log`, `/tmp/n1_set.log`, `/tmp/n1_gr.log`
