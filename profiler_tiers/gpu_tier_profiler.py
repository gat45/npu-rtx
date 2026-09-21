#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_tier_profiler.py — Profiler GPU-tier par COPIE (zéro modif des sources profiler_v4)
Le dossier profiler_tiers/ contient une copie de profiler_v4/*.py ; ce moteur est autonome.

Deux profils matériels avec TOUS les goulots (PCIe D2/DMA + bus mémoire GDDR) :
  5070 : laptop 8 Go — 384 GB/s GDDR7 théo, ~165 GB/s eff (51%, roofline recalé)
         PCIe 5.0 x8 ~31.5 GB/s ; + NPU XDNA2 sur-package (Voie A double-flux)
  1080 : dev        — 320 GB/s GDDR5X théo, ~205 GB/s eff (64% Pascal)
         PCIe 3.0 x16 ~15.75 GB/s ; banc de validation LOGIQUE uniquement

Usage :
  py gpu_tier_profiler.py plan  --machine 5070 --model 35b [--kv turbo4] [--experts 64] [--ctx 8192]
  py gpu_tier_profiler.py raw   --machine 5070 --gguf <fichier.gguf>
  py gpu_tier_profiler.py bench --machine 5070 --kind gpu --q 4bit --ctx 8192
  py gpu_tier_profiler.py compare
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# TABLES MATÉRIEL — ancres mesurées quand elles existent, sinon datasheet × eta
# ============================================================================
MACHINES = {
    "5070": {
        "label": "RTX 5070 Laptop 8 GB (machine cible, CUDA sm_120)",
        "sm": "sm_120",
        "vram_gb": 8.0,
        "bw_theo_gbs": 384.0,
        "bw_eff_gbs": 165.0,           # 51% — roofline FLM 9B recalé 5070 (Phase 1)
        "bw_eff_source": "measured-roofline-FLM-9B-rewritten-5070",
        "pcie_gen": 5.0,
        "pcie_lanes": 8,
        "pcie_gbs_theo": 31.5,         # sens unique x8 gen5 (~3.94 GB/s/lane)
        "kv_types_avail": ["f16", "q8_0", "turbo4"],
        "kv_forbidden": ["turbo3-as-K"],
        "experts_per_layer": 256,
        "expert_bytes_q4": 2_450_000,    # 2.45 MB/expert Q4_0 (35B-A3B)
        "expert_bytes_nvfp4": 1_250_000, # 1.25 MB/expert NVFP4 (FP4 natif Blackwell sm_120)
        "n_layers": 40,
        "backend": "CUDA sm_120",
        "npu": True,                   # XDNA2 sur-package Strix (pas de PCIe NPU<->GPU)
        "notes": "NPU = tier overflow/double-flux Voie A (61.5 t/s agrégat simulé), jamais backend principal (bench.json Phase 1)",
    },
    "1080": {
        "label": "GTX 1080 8 GB (dev, Pascal sm_61)",
        "sm": "sm_61",
        "vram_gb": 8.0,
        "bw_theo_gbs": 320.0,
        "bw_eff_gbs": 205.0,           # ~64% effectif GDDR5X (Pascal typique)
        "bw_eff_source": "Pascal-typical-64pct",
        "pcie_gen": 3.0,
        "pcie_lanes": 16,
        "pcie_gbs_theo": 15.75,        # gen3 x16 ≈ 985 MB/s/lane
        "kv_types_avail": ["f16", "q8_0", "turbo4"],
        "kv_forbidden": ["turbo3-as-K"],
        "experts_per_layer": 256,
        "expert_bytes_q4": 2_450_000,
        "expert_bytes_nvfp4": 1_250_000, # théorique sur Pascal (pas de FP4 natif — validation logique seule)
        "n_layers": 40,
        "backend": "CUDA sm_61 (build-cu61, patch D512 smem)",
        "npu": False,
        "notes": "banc de validation LOGIQUE uniquement — perf non représentative Blackwell",
    },
}

# Ancres débit mesurées / calibrées (bench.json Phase 1 + rapports)
# NPU overflow ≈ 16× un hit GPU ; Voie A double-flux = 61.5 t/s agrégat simulé.
NPU_OVERFLOW_COST = 16.0
VOIE_A_AGREGAT_TPS = 61.5
ANCHOR_1080_QWEN9B_IQ4NL_TPS = 32.9   # mesure 1080 déjà consolidée
DENSE_BACKBONE_BYTES = 1.5e9          # stand-in calibré W_eff (85.2% trafic, vLLM #51197) — à recalibrer par llama-bench

# Modèles de référence (archi MoE)
MODELS = {
    "35b": {"label": "Qwen3.5-35B-A3B", "n_layers": 40, "experts": 256, "top_k": 8,
            "active_params_b": 3.1, "kv_head_dim": 128, "n_kv_heads": 4},
    "9b":  {"label": "Qwen3.5-9B dense", "n_layers": 40, "experts": 0, "top_k": 0,
            "active_params_b": 9.0, "kv_head_dim": 128, "n_kv_heads": 8},
    # marco8b : GGUF RÉEL mesuré (models_marco_v2, arch qwen3moe, 4.27 GiB Q4_0)
    "marco8b": {"label": "Marco-Nano 8B-A0.6B (qwen3moe, Q4_0 réel)", "n_layers": 36, "experts": 232, "top_k": 8,
                "active_params_b": 0.6, "kv_head_dim": 128, "n_kv_heads": 4},
}


# ============================================================================
# GOULOTS — PCIe (D2 expert streaming / FATE) + bus mémoire (decode BW-bound)
# ============================================================================
def pcie_stream_time_s(machine, bytes_to_move):
    """Temps de streaming d'experts (D2 style) sur PCIe = goulot #1 en MoE offload."""
    return bytes_to_move / (machine["pcie_gbs_theo"] * 1e9)


def kv_cache_bytes(model, ctx_tokens, ktype="f16"):
    """Octets KV totaux : 2 (K+V) × layers × heads × head_dim × 2 B (f16) × ctx.
    Compression turbo4 = 4.125 bpw vs 16 bpw f16 → ×0.2578."""
    per_tok_layer = model["n_kv_heads"] * model["kv_head_dim"] * 2 * 2  # K+V, f16
    total = per_tok_layer * model["n_layers"] * ctx_tokens
    if ktype == "f16":
        factor = 1.0
    elif ktype == "q8_0":
        factor = 8.5 / 16.0
    elif ktype == "turbo4":
        factor = 4.125 / 16.0
    elif ktype == "turbo3":
        factor = 3.25 / 16.0
    else:
        factor = 1.0
    return int(total * factor)


def decode_tps_bw_bound(machine, active_bytes_per_token):
    """t/s decode ≈ BW_eff / octets lus par token (goulot #2 : bus mémoire)."""
    return machine["bw_eff_gbs"] * 1e9 / max(active_bytes_per_token, 1)


def gpu_hit_vs_overflow(model, machine, n_resident_experts):
    """Coût token : hits VRAM (BW GPU) vs overflow NPU (×16, Phase 1) ou PCIe."""
    epl = model["experts"]
    if epl == 0:
        return None
    p_hit = min(n_resident_experts / epl, 1.0)
    return p_hit


# ============================================================================
# MODES
# ============================================================================
def mode_raw(args):
    """Profilage RAW d'un GGUF local : en-tête + placement vs VRAM 8 Go."""
    from profile_model import read_gguf_header  # copie profiler_v4, non modifiée
    m = MACHINES[args.machine]
    # Contrat de la copie : (meta, tensors) avec tensors = [(name, dims, ttype, nbytes)]
    meta, tensors = read_gguf_header(args.gguf)
    n_tensors = len(tensors)
    total_bytes = sum(t[3] for t in tensors)

    def mget(*keys, default=None):
        for k in keys:
            if k in meta:
                return meta[k]
        return default

    arch = mget("general.architecture", default="?")
    print(f"== RAW {args.gguf} ==")
    print(f"machine: {m['label']}  VRAM {m['vram_gb']} GB  BW_eff {m['bw_eff_gbs']} GB/s  PCIe gen{m['pcie_gen']:.0f} x{m['pcie_lanes']} ({m['pcie_gbs_theo']} GB/s)")
    print(f"arch={arch} | {n_tensors} tenseurs | poids total {total_bytes/2**30:.2f} GiB")
    budget = m["vram_gb"] * 0.92 * (1 << 30)
    fit = total_bytes <= budget
    print(f"fit VRAM {m['vram_gb']} Go (garde 8%): {'OUI — tout résident, decode BW-bound' if fit else 'NON -> experts streaming/overflow (PCIe = goulot)'}")
    if not fit:
        excess = total_bytes - budget
        t = pcie_stream_time_s(m, excess)
        print(f"excès {excess/2**30:.2f} GiB -> {t*1000:.0f} ms par full-swap PCIe")

    # KV f16 estimé depuis les métadonnées (fallback si absentes)
    ctx = getattr(args, "ctx", 8192)
    n_layers = mget(f"{arch}.block_count", "block_count")
    n_kv_head = mget(f"{arch}.attention.head_count_kv", "attention.head_count_kv")
    n_head = mget(f"{arch}.attention.head_count", "attention.head_count")
    n_embd = mget(f"{arch}.embedding_length", "embedding_length")
    if all(v is not None for v in (n_layers, n_kv_head, n_head, n_embd)) and n_head:
        head_dim = n_embd // n_head
        kv_f16 = 2 * 2 * n_layers * n_kv_head * head_dim * ctx
        print(f"[KV f16 @ctx={ctx}] {kv_f16/2**30:.2f} GiB -> +KV fit: {'OUI' if total_bytes + kv_f16 <= budget else 'NON (KV turbo4 requis)'}")
    for k in ("general.name", f"{arch}.expert_count", f"{arch}.expert_used_count", f"{arch}.context_length"):
        if k in meta:
            print(f"  {k} = {meta[k]}")
    return meta, tensors


def mode_plan(args):
    """PLAN : placement MoE sur 8 Go avec goulots PCIe + bus mémoire + KV turbo4."""
    m = MACHINES[args.machine]
    mdl = MODELS[args.model]
    ctx = args.ctx
    kv = args.kv
    n_res = args.experts

    print(f"=== PLAN {m['label']} ===")
    print(f"backend {m['backend']} | BW théo {m['bw_theo_gbs']} GB/s | BW eff {m['bw_eff_gbs']} GB/s ({m['bw_eff_source']})")
    print(f"PCIe gen{m['pcie_gen']:.0f} x{m['pcie_lanes']} = {m['pcie_gbs_theo']} GB/s (goulot D2/FATE streaming experts)")

    if "turbo3-as-K" in m["kv_forbidden"] and kv == "turbo3":
        print("!! KV turbo3 INTERDIT en K (bug upstream garbage, rapport Phase 2 §5) -> forcé turbo4")
        kv = "turbo4"

    # KV cache
    kvb = kv_cache_bytes(mdl, ctx, kv)
    print(f"\n[KV] {mdl['label']} ctx={ctx} {kv}: {kvb/2**30:.2f} GiB", end="")
    if kv == "turbo4":
        print(" (×0.258 vs f16 — validé Phase 2, texte ≈ f16)")

    # Budget VRAM 8 Go
    budget = m["vram_gb"] * 0.92 * 1e9
    kv_after = budget - kvb
    print(f"[VRAM] budget utile {budget/2**30:.2f} GiB -> après KV: {kv_after/2**30:.2f} GiB pour poids+experts")

    # Résidence experts (MoE)
    if mdl["experts"] > 0:
        fmt = getattr(args, "expert_fmt", "q4")
        per_exp = m["expert_bytes_nvfp4"] if fmt == "nvfp4" else m["expert_bytes_q4"]
        max_res_by_vram = int(kv_after / per_exp / mdl["n_layers"])
        n_res_eff = min(n_res, mdl["experts"], max_res_by_vram)
        hit = gpu_hit_vs_overflow(mdl, m, n_res_eff)
        fmt_note = " (FP4 natif Blackwell)" if fmt == "nvfp4" and m["sm"] == "sm_120" else (" (émul — Pascal sans FP4 natif)" if fmt == "nvfp4" else "")
        print(f"[EXPERTS] {mdl['experts']}/couche × {per_exp/1e6:.2f} MB {fmt}{fmt_note} | résident max VRAM: {max_res_by_vram}/couche")
        print(f"  -> résidence {n_res_eff}/couche = hit GPU {hit:.2f} | overflow {1-hit:.2f}")

        # GOULOT PCIe : overflow experts streamés
        ovf_bytes = (1 - hit) * mdl["top_k"] * per_exp
        t_pcie = pcie_stream_time_s(m, ovf_bytes)
        print(f"  [PCIe] overflow/token ≈ {ovf_bytes/1e6:.1f} MB -> {t_pcie*1000:.2f} ms/tok (goulot dur si > 5 ms)")

        # Coût NPU (5070 uniquement) : overflow -> NPU ×16 (Phase 1)
        if m.get("npu"):
            print(f"  [NPU XDNA2] overflow traitable sur NPU (coût ≈ {NPU_OVERFLOW_COST:.0f}× hit GPU, sur-package Strix: zéro PCIe) — Voie A double-flux {VOIE_A_AGREGAT_TPS} t/s agrégat simulé")

    # Decode BW-bound : le terme dominant est le DENSE BACKBONE (85.2% du trafic,
    # vLLM #51197) + attention + lecture KV ; les experts résidents pèsent top_k×hit.
    dense_backbone_bytes = DENSE_BACKBONE_BYTES
    if mdl["experts"] > 0:
        act_bytes = mdl["top_k"] * per_exp * (gpu_hit_vs_overflow(mdl, m, n_res_eff) or 0) + dense_backbone_bytes
    else:
        act_bytes = 2.87e9  # W_eff 9B clean (calibré FLM)
    tps = decode_tps_bw_bound(m, act_bytes)
    print(f"\n[BUS MEMOIRE] octets/token ≈ {act_bytes/1e9:.2f} GB -> decode ≈ {tps:.1f} t/s (BW-bound, goulot #2)")
    if args.machine == "1080":
        print(f"  ancre 1080 mesurée: Qwen9B IQ4NL = {ANCHOR_1080_QWEN9B_IQ4NL_TPS} t/s (cohérence ordre de grandeur)")
    if m.get("npu"):
        print(f"  Voie A (5070+NPU double-flux): {VOIE_A_AGREGAT_TPS} t/s agrégat (simulé Phase 1, à confirmer machine cible)")


def mode_bench(args):
    """BENCH : roofline paramétrique (kind=gpu, quant, ctx) — pas de device requis."""
    m = MACHINES[args.machine]
    qbytes = {"4bit": 0.5, "8bit": 1.0, "16bit": 2.0}[args.q]
    active_params = 3.1e9 if args.kind == "moe" else 9.0e9
    act_bytes = active_params * qbytes
    tps = decode_tps_bw_bound(m, act_bytes)
    print(f"bench[{m['label']}] kind={args.kind} q={args.q} ctx={args.ctx}")
    print(f"  octets/token ≈ {act_bytes/1e9:.2f} GB -> {tps:.1f} t/s théorique BW-bound")
    print(f"  PCIe {m['pcie_gbs_theo']} GB/s -> swap 1 GiB en {1024/m['pcie_gbs_theo']*1000:.0f} ms (goulot D2)")


def mode_compare(_args):
    """Tableau comparatif 5070 vs 1080 — goulots côte à côte."""
    hdr = f"{'métrique':<28}{'5070 (cible)':<26}{'1080 (dev)':<22}"
    print(hdr); print("-" * len(hdr))
    rows = [
        ("BW théo GB/s", "384 (GDDR7)", "320 (GDDR5X)"),
        ("BW eff GB/s", "165 (51%, roofline)", "205 (64% Pascal)"),
        ("PCIe", "gen5 x8 = 31.5 GB/s", "gen3 x16 = 15.75 GB/s"),
        ("VRAM", "8 GB", "8 GB"),
        ("KV turbo4 (8k tok, 35B)", f"{kv_cache_bytes(MODELS['35b'], 8192, 'turbo4')/2**30:.2f} GiB", f"{kv_cache_bytes(MODELS['35b'], 8192, 'turbo4')/2**30:.2f} GiB"),
        ("K=turbo3", "INTERDIT (bug)", "INTERDIT (bug)"),
        ("NPU XDNA2", "oui (Voie A 61.5 t/s sim.)", "non"),
        ("sm", "sm_120", "sm_61 (patch D512)"),
        ("rôle", "cible perf", "banc validation logique"),
    ]
    for a, b, c in rows:
        print(f"{a:<28}{b:<26}{c:<22}")


def mode_sweep(args):
    """SWEEP résidence (ex. 32→256 experts/couche) : hit GPU, overflow MB/tok,
    latence PCIe ms/tok, decode t/s — un seul tableau, avec frontière VRAM."""
    m = MACHINES[args.machine]
    mdl = MODELS[args.model]
    if mdl["experts"] == 0:
        print("modèle dense (0 expert) — sweep inapplicable")
        return
    fmt = args.expert_fmt
    per_exp = m["expert_bytes_nvfp4"] if fmt == "nvfp4" else m["expert_bytes_q4"]
    kvb = kv_cache_bytes(mdl, args.ctx, args.kv)
    budget = m["vram_gb"] * 0.92 * 1e9
    kv_after = budget - kvb
    max_res = int(kv_after / per_exp / m["n_layers"])
    epl = mdl["experts"]
    to = min(args.to, epl)
    fmt_note = " (FP4 natif Blackwell)" if fmt == "nvfp4" and m["sm"] == "sm_120" else (" (émul — Pascal sans FP4 natif)" if fmt == "nvfp4" else "")
    hdr = f"{'rés/couche':>10} {'hit GPU':>8} {'ovf MB/tok':>11} {'PCIe ms/tok':>12} {'decode t/s':>11}  note"
    print(f"=== SWEEP résidence — {m['label']} — {mdl['label']} — experts {fmt}{fmt_note} — KV {args.kv} @ctx={args.ctx} ===")
    print(f"PCIe gen{m['pcie_gen']:.0f} x{m['pcie_lanes']} = {m['pcie_gbs_theo']} GB/s | BW eff {m['bw_eff_gbs']} GB/s | résident max VRAM: {max_res}/{epl}/couche")
    print(hdr)
    print("-" * len(hdr))
    n = args.frm
    while n <= to:
        hit = min(n / epl, 1.0)
        ovf = (1 - hit) * mdl["top_k"] * per_exp
        t_pcie = pcie_stream_time_s(m, ovf) * 1000
        act = mdl["top_k"] * per_exp * hit + DENSE_BACKBONE_BYTES
        tps = decode_tps_bw_bound(m, act)
        note = ""
        if n > max_res:
            note = "DEPASSE VRAM"
        elif n == max_res:
            note = "<- max VRAM"
        elif n == to and hit >= 1.0:
            note = "full résident"
        print(f"{n:>10} {hit:>8.2f} {ovf/1e6:>11.1f} {t_pcie:>12.2f} {tps:>11.1f}  {note}")
        n += args.step


def main():
    ap = argparse.ArgumentParser(description="GPU tier profiler (copies, sources intactes)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("raw");   p.add_argument("--machine", choices=list(MACHINES), required=True); p.add_argument("--gguf", required=True); p.add_argument("--ctx", type=int, default=8192); p.set_defaults(fn=mode_raw)
    p = sub.add_parser("plan");  p.add_argument("--machine", choices=list(MACHINES), required=True)
    p.add_argument("--model", choices=list(MODELS), default="35b")
    p.add_argument("--kv", default="turbo4", choices=["f16", "q8_0", "turbo3", "turbo4"])
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--expert-fmt", default="q4", choices=["q4", "nvfp4"])
    p.add_argument("--ctx", type=int, default=8192)
    p.set_defaults(fn=mode_plan)
    p = sub.add_parser("sweep"); p.add_argument("--machine", choices=list(MACHINES), required=True)
    p.add_argument("--model", choices=list(MODELS), default="35b")
    p.add_argument("--kv", default="turbo4", choices=["f16", "q8_0", "turbo3", "turbo4"])
    p.add_argument("--expert-fmt", default="nvfp4", choices=["q4", "nvfp4"])
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--from-res", dest="frm", type=int, default=32)
    p.add_argument("--to-res", dest="to", type=int, default=256)
    p.add_argument("--step", type=int, default=16)
    p.set_defaults(fn=mode_sweep)
    p = sub.add_parser("bench"); p.add_argument("--machine", choices=list(MACHINES), required=True)
    p.add_argument("--kind", default="moe", choices=["moe", "gpu"]); p.add_argument("--q", default="4bit", choices=["4bit", "8bit", "16bit"])
    p.add_argument("--ctx", type=int, default=8192); p.set_defaults(fn=mode_bench)
    p = sub.add_parser("compare"); p.set_defaults(fn=mode_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
