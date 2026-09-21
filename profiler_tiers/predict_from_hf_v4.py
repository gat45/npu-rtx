#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_from_hf_v4.py — comparaison de formats de quantification REELS
(pas hypothetiques) pour un modele HuggingFace GGUF, sans telechargement.

Corrige deux limites reelles de predict_from_hf.py (v3), auditees le 2026-09-13 :
  1. v3 ignore le format REEL du fichier distant : il calcule toujours une
     taille hypothetique "si le modele etait en Q4_0", quel que soit le
     fichier demande (--file Nanbeige-Q6_K.gguf donnait le meme resultat
     que --file Nanbeige-Q4_0.gguf). Corrige ici en lisant le champ
     "ttype" reel de chaque tenseur dans l'en-tete GGUF distant.
  2. v3 n'a aucune notion de couverture HTP par type ni de QAIRT (W4A16/
     W8A16). Ajoute ici la table de couverture ggml-hexagon confirmee la
     session du 2026-09-13 (Marco-Nano SOFT_MAX, Nanbeige4.2-3B MUL_MAT) :
     seuls Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 (+F16/F32) passent MUL_MAT sur HTP ;
     tout K-quant (Q2_K..Q8_K) et Q1_0/Q2_0 tombent en repli CPU total.
     Les representations QAIRT (W4A16/W8A16) sont un chemin d'execution
     SEPARE (QNN, pas GGML/ggml-hexagon) : leur taille est estimee ici a
     titre de comparaison memoire uniquement, PAS de debit, car ce
     runtime ne les execute pas directement.

Usage :
  py predict_from_hf_v4.py <repo> --file <fichier.gguf>   # analyse un fichier reel
  py predict_from_hf_v4.py <repo> --grid                  # grille tous formats dispo
"""
import argparse
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import profile_model as pm  # noqa: E402
from predict_from_hf import fetch_gguf_header_remote, list_hf_gguf_files  # noqa: E402

# ---------------------------------------------------------------------------
# Couverture HTP confirmee experimentalement le 2026-09-13
# (Marco-Nano : SOFT_MAX ne0=232 rejete ; Nanbeige4.2-3B : MUL_MAT Q4_K/Q6_K
#  rejete a 100%, Q4_0 accepte a 93.2%). Source : lecture directe de
#  ggml_hexagon_supported_mul_mat() dans le fork self-build-jz, commit 6be3d3dd.
HTP_MUL_MAT_COVERAGE = {
    "F32": True, "F16": True,
    "Q4_0": True, "Q4_1": True, "Q8_0": True, "IQ4_NL": True, "MXFP4": True,
    "Q2_K": False, "Q3_K": False, "Q4_K": False, "Q5_K": False,
    "Q6_K": False, "Q8_K": False,
    "Q1_0": False, "Q2_0": False,
    "IQ1_S": False, "IQ2_XXS": False, "IQ3_XXS": False,
}

# Bits par poids reels (pas l'approximation BPW de profile_model, qui sert
# au dimensionnement memoire hypothetique) — pour affichage informatif.
REAL_WBITS = {
    "F32": 32, "F16": 16, "Q8_0": 8.5, "Q4_1": 4.5, "Q4_0": 4.25,
    "IQ4_NL": 4.25, "MXFP4": 4.0,
    "Q8_K": 8.0, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5, "Q3_K": 3.4375,
    "Q2_K": 2.625,
    "Q1_0": 1.0, "Q2_0": 2.0,
}

# QAIRT (Qualcomm QNN) — chemin d'execution distinct, PAS ggml-hexagon.
# Estimation memoire seulement ; aucun modele de debit ici (architecture
# d'execution differente : pas de MUL_MAT ggml, pas de scheduler CPU/HTP
# ggml_backend_sched, packing/scales proprietaires Qualcomm).
QAIRT_FORMATS = {
    "QAIRT-W8A16": {"weight_bits": 8, "act_bits": 16, "note": "poids INT8, activations INT16"},
    "QAIRT-W4A16": {"weight_bits": 4, "act_bits": 16, "note": "poids INT4, activations INT16"},
}


def real_size_from_tensor_list(tlist):
    """Taille reelle (octets) du fichier GGUF distant, par type de tenseur
    REELLEMENT present (pas une hypothese). tlist : liste de (name, dims,
    ttype_code, nb) telle que retournee par fetch_gguf_header_remote.
    Retourne (total_bytes, {nom_type_str: bytes})."""
    total = 0
    by_type = {}
    for _name, _dims, ttype_code, nb in tlist:
        type_name = pm.V_TYPES.get(ttype_code, f"UNKNOWN_{ttype_code}")
        total += nb
        by_type[type_name] = by_type.get(type_name, 0) + nb
    return total, by_type


# ---------------------------------------------------------------------------
# Calibration empirique 2026-09-13 (deux mesures reelles independantes,
# PAS un modele invente) :
#   Nanbeige4.2-3B : Q6_K sur attn_q/output (tenseur RECURRENT, 1x/couche) ->
#                    CPU/HTP scheduler time ratio mesure = 50.8x
#   Marco-Nano     : SOFT_MAX ne0=232 (op RECURRENTE, 1x/couche/token) ->
#                    +31.7% tg32 en supprimant le fallback CPU correspondant
# Familles de tenseurs GGUF connues comme recurrentes (1x/couche, donc
# executees a CHAQUE token en decode) vs isolees (1x par forward pass, cout
# amorti sur tout le contexte) :
RECURRENT_TENSOR_PATTERNS = ["attn_q", "attn_k", "attn_v", "attn_output",
                            "ffn_gate", "ffn_up", "ffn_down", "ffn_norm",
                            "attn_norm", "ffn_moe", "attn_qkv"]
ISOLATED_TENSOR_PATTERNS = ["output.weight", "token_embd", "output_norm"]


def classify_tensor_name(name):
    low = name.lower()
    for pat in ISOLATED_TENSOR_PATTERNS:
        if pat in low:
            return "isolated"
    for pat in RECURRENT_TENSOR_PATTERNS:
        if pat in low:
            return "recurrent"
    return "unknown"


def compute_htp_risk(tlist, n_layer):
    """Score de risque HTP multi-niveaux, calibre sur les ratios CPU/HTP
    mesures reellement le 2026-09-13 (pas une formule arbitraire) :
      - un tenseur non-supporte RECURRENT (attn_*/ffn_*, 1x/couche) a
        historiquement produit un ratio CPU/HTP de 50.8x (Nanbeige) et un
        gain de +31.7% tg32 en le corrigeant (Marco) -> penalite VERY_HIGH
      - un tenseur non-supporte ISOLATED (output.weight, 1x/forward pass)
        n'a pas ete mesure comme dominant le temps -> penalite MODERATE
    """
    unsupported_recurrent = []
    unsupported_isolated = []
    unsupported_unknown = []
    total_bytes = 0
    unsupported_bytes = 0

    for name, _dims, ttype_code, nb in tlist:
        total_bytes += nb
        type_name = pm.V_TYPES.get(ttype_code, f"UNKNOWN_{ttype_code}")
        if HTP_MUL_MAT_COVERAGE.get(type_name, False):
            continue
        unsupported_bytes += nb
        cls = classify_tensor_name(name)
        if cls == "recurrent":
            unsupported_recurrent.append((name, type_name, nb))
        elif cls == "isolated":
            unsupported_isolated.append((name, type_name, nb))
        else:
            unsupported_unknown.append((name, type_name, nb))

    c_bytes = 1.0 - (unsupported_bytes / total_bytes if total_bytes else 0.0)

    if unsupported_recurrent:
        verdict = "REQUANTIFIER (risque tres eleve, calibre sur Nanbeige/Marco 2026-09-13)"
        risk = "VERY_HIGH"
        rationale = (f"{len(unsupported_recurrent)} tenseur(s) recurrent(s) non supporte(s) "
                    f"detecte(s) (ex: {unsupported_recurrent[0][0]}, type {unsupported_recurrent[0][1]}). "
                    f"Historique mesure : ce motif exact a produit un ratio CPU/HTP de 50.8x "
                    f"sur Nanbeige4.2-3B et +31.7% tg32 gagnable en le corrigeant sur Marco-Nano.")
    elif unsupported_isolated:
        verdict = "SURVEILLER (risque modere, cout non amorti par token mais existant)"
        risk = "MODERATE"
        rationale = (f"{len(unsupported_isolated)} tenseur(s) isole(s) non supporte(s) "
                    f"(ex: {unsupported_isolated[0][0]}) — cout probablement acceptable "
                    f"car execute une seule fois par forward pass, pas par token.")
    elif unsupported_unknown:
        risk = "UNKNOWN"
        verdict = "A VERIFIER MANUELLEMENT (famille de tenseur non classifiee)"
        rationale = f"{len(unsupported_unknown)} tenseur(s) non supporte(s) de famille non reconnue."
    else:
        risk = "LOW"
        verdict = "OK — aucun tenseur non supporte detecte"
        rationale = "Couverture HTP complete sur les types de tenseurs presents."

    return {
        "c_bytes_pct": c_bytes * 100.0,
        "unsupported_recurrent": unsupported_recurrent,
        "unsupported_isolated": unsupported_isolated,
        "unsupported_unknown": unsupported_unknown,
        "risk": risk, "verdict": verdict, "rationale": rationale,
    }


def check_softmax_risk(n_experts):
    """SOFT_MAX rejete par ggml_hexagon_supported_softmax() si ne0 > 32 et
    non multiple de 32 — ne0 du routeur MoE = n_experts. Verifie sur Marco-
    Nano (ne0=232, 232%32=8) le 2026-09-13, cf. capability_db.CAPABILITY_ENTRIES.
    Retourne None si le modele n'est pas MoE (pas de SOFT_MAX de routing)."""
    if not n_experts or n_experts <= 1:
        return None
    if n_experts <= 32:
        return {"at_risk": False, "n_experts": n_experts,
               "reason": "ne0 <= 32 : la voie HVX tail-only fonctionne (pas de rejet)"}
    aligned = (n_experts % 32) == 0
    return {
        "at_risk": not aligned, "n_experts": n_experts,
        "reason": (f"{n_experts} % 32 = {n_experts % 32} -> "
                  f"{'aligne, SOFT_MAX devrait passer HTP' if aligned else 'NON aligne, SOFT_MAX du routeur MoE rejete -> CPU a chaque couche/token (motif identique a Marco-Nano ne0=232, +31.7% tg32 mesure en le corrigeant, non deploye sans validation numerique)'}"),
    }


def analyze_real_file(repo_id, filename):
    meta, tlist = fetch_gguf_header_remote(repo_id, filename)
    tensors = {}
    for name, dims, ttype_code, nb in tlist:
        ne = 1
        for d in dims:
            ne *= d
        tensors[name] = {"dtype": pm.V_TYPES.get(ttype_code, "?"), "shape": dims,
                         "bytes": nb, "elems": ne, "ttype": ttype_code}
    a = pm.analyze({}, tensors, gguf_meta=meta, is_gguf=True, ctx=4096)
    total_bytes, by_type = real_size_from_tensor_list(tlist)

    dominant_type = max(by_type.items(), key=lambda kv: kv[1])[0] if by_type else "?"
    htp_ok_bytes = sum(b for t, b in by_type.items() if HTP_MUL_MAT_COVERAGE.get(t, False))
    htp_ok_pct = 100.0 * htp_ok_bytes / total_bytes if total_bytes else 0.0
    risk = compute_htp_risk(tlist, a["arch"]["n_layer"])
    softmax_risk = check_softmax_risk(a["arch"].get("n_experts", 0))

    return {
        "repo": repo_id, "file": filename, "arch": a["arch"],
        "total_gib": total_bytes / 2**30,
        "by_type_gib": {t: b / 2**30 for t, b in by_type.items()},
        "dominant_type": dominant_type,
        "htp_mul_mat_covered_pct": htp_ok_pct,
        "n_tensor_types": len(by_type),
        "mixed": len(by_type) > 1,
        "risk": risk,
        "softmax_risk": softmax_risk,
    }


def render_real_file(r):
    L = [f"# Analyse REELLE (pas hypothetique) — {r['repo']} / {r['file']}", ""]
    L.append(f"- architecture : {r['arch']['n_layer']} couches, hidden {r['arch']['hidden']}")
    L.append(f"- taille reelle totale : {r['total_gib']:.2f} GiB")
    L.append(f"- type dominant : {r['dominant_type']}")
    L.append("")
    L.append("## Repartition reelle par type de tenseur")
    for t, gib in sorted(r["by_type_gib"].items(), key=lambda kv: -kv[1]):
        htp = "HTP MUL_MAT OK" if HTP_MUL_MAT_COVERAGE.get(t) else "REPLI CPU (non supporte MUL_MAT ggml-hexagon)"
        L.append(f"  - {t:10s} : {gib:6.3f} GiB  [{htp}]")
    L.append("")
    if r["mixed"]:
        L.append(f"[MIXTE] Ce fichier contient {r['n_tensor_types']} types de tenseurs differents. "
                 f"Un fichier nomme d'apres son type dominant ({r['dominant_type']}) peut donc "
                 f"contenir des tenseurs isoles dans un autre format (typiquement attn_q/output "
                 f"en K-quant plus precis) — c'est exactement la cause racine trouvee sur "
                 f"Nanbeige4.2-3B le 2026-09-13 (347/5099 MUL_MAT rejetes, tous en Q6_K).")
    L.append("")
    L.append(f"- Couverture HTP MUL_MAT (par octets de poids) : {r['htp_mul_mat_covered_pct']:.1f}% "
             "(indicatif seul, voir verdict pondere ci-dessous)")
    L.append("")
    L.append("## Verdict HTP-aware (pondere par frequence d'execution, pas seulement par octets)")
    risk = r["risk"]
    L.append(f"- **Risque : {risk['risk']}**")
    L.append(f"- **Verdict : {risk['verdict']}**")
    L.append(f"  {risk['rationale']}")
    if risk["unsupported_recurrent"]:
        L.append("")
        L.append("  Tenseurs RECURRENTS non supportes (1x/couche, executes a CHAQUE token — "
                 "cout le plus critique, calibre sur Nanbeige/Marco 2026-09-13) :")
        for name, ttype, nb in risk["unsupported_recurrent"][:10]:
            L.append(f"    - {name} ({ttype}, {nb/2**20:.1f} MiB)")
        if len(risk["unsupported_recurrent"]) > 10:
            L.append(f"    ... et {len(risk['unsupported_recurrent'])-10} de plus")
    if risk["unsupported_isolated"]:
        L.append("")
        L.append("  Tenseurs ISOLES non supportes (1x/forward pass, cout amorti) :")
        for name, ttype, nb in risk["unsupported_isolated"][:5]:
            L.append(f"    - {name} ({ttype}, {nb/2**20:.1f} MiB)")
    sm = r.get("softmax_risk")
    if sm:
        L.append("")
        L.append("## Risque SOFT_MAX routeur MoE (ggml_hexagon_supported_softmax)")
        L.append(f"- n_experts detecte : {sm['n_experts']}")
        L.append(f"- {'RISQUE : ' if sm['at_risk'] else 'OK : '}{sm['reason']}")
    L.append("")
    L.append("[LIMITE] Ce verdict s'appuie sur DEUX mesures reelles (Nanbeige4.2-3B, Marco-Nano, "
             "2026-09-13), pas une calibration statistique large. Le classement recurrent/isole "
             "des familles de tenseurs est heuristique (nom du tenseur), pas trace depuis un "
             "vrai graphe d'execution — a verifier par mesure directe (split-timing/mulmat-reject) "
             "avant toute decision de deploiement definitive.")
    return "\n".join(L)


def render_grid(repo_id, files):
    """Grille GGUF (types reels agreges par fichier) + ligne QAIRT informative."""
    L = [f"# Grille de comparaison de formats — {repo_id}", ""]
    L.append("| Fichier | Taille reelle (GiB) | Type dominant | Couverture HTP MUL_MAT |")
    L.append("|---|---:|---|---:|")
    for fn in files:
        try:
            r = analyze_real_file(repo_id, fn)
            L.append(f"| {fn} | {r['total_gib']:.2f} | {r['dominant_type']} | {r['htp_mul_mat_covered_pct']:.1f}% |")
        except Exception as e:  # noqa: BLE001 - diagnostic tool, on veut voir l'echec par fichier
            L.append(f"| {fn} | ERREUR: {e} | - | - |")
    L.append("")
    L.append("## Pour comparaison memoire — QAIRT (chemin d'execution SEPARE, non GGML)")
    L.append("[NON EXECUTABLE PAR CE RUNTIME] Ces lignes sont une estimation memoire "
             "informative uniquement. QAIRT (QNN) n'utilise pas ggml_hexagon_supported_mul_mat, "
             "pas le scheduler ggml_backend_sched, ni le meme packing de poids — ce n'est PAS "
             "un format que ggml-hexagon peut charger directement, et cet outil ne modelise "
             "aucun debit pour cette voie.")
    for name, spec in QAIRT_FORMATS.items():
        L.append(f"  - {name} ({spec['note']}) : {spec['weight_bits']} bits/poids nominal, "
                 f"activations {spec['act_bits']} bits — taille memoire du meme ordre qu'un "
                 f"GGUF {'Q4_K/Q4_0' if spec['weight_bits'] == 4 else 'Q8_0/Q8_K'} equivalent, "
                 f"mais **kernel, scheduler et cout des frontieres CPU/HTP totalement differents "
                 f"et non mesures ici**.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--file", help="analyse reelle d'un seul fichier")
    ap.add_argument("--grid", action="store_true", help="grille sur tous les .gguf du repo")
    args = ap.parse_args()

    if args.grid or not args.file:
        files = list_hf_gguf_files(args.repo)
        if not files:
            print("Aucun fichier .gguf trouve.")
            return
        print(render_grid(args.repo, files))
        return

    r = analyze_real_file(args.repo, args.file)
    print(render_real_file(r))


if __name__ == "__main__":
    main()
