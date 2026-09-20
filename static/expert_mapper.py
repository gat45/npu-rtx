#!/usr/bin/env python3
"""expert_mapper.py — mapping d'un expert MoE Qwen 2560×640 vers les tuiles XDNA2.

P0 de la campagne (OUBLIS_DECISIONS §1). Objectif : produire la table
"expert → liste de (tile_M, tile_K, tile_N, offset, mode mmul)" qui alimente le cost model.

Contraintes XDNA2 vérifiées (VERIFICATION_NPU_PREFERENCES) :
- INT8 = chemin natif P0, mmul 8x8x8 = 512 MACs/cycle/tile
- 64 KB core-local memory = limite par core (tile A+B+C doit tenir)
- Q4NX legacy : row_chunk 32, col_chunk 256, block 5120 B (pour le layout de stockage)
- 640 et 2560 sont multiples de 8 → pas de pénalité d'alignement 8x8x8
Résultat : table JSON par tensor (gate/up/down), avec nombre de tuiles et taille DMA estimée.
"""

from model_parser import canonical_dims, expert_shape, params_per_expert
from quant_size_engine import expert_bytes

# Contraintes XDNA2
L1_CORE_BYTES = 64 * 1024           # 64 KB core-local
MMUL_INT8 = (8, 8, 8)               # tile (M,K,N) du mode 8x8x8 (512 MACs/cyc)
Q4NX_ROW_CHUNK = 32
Q4NX_COL_CHUNK = 256
Q4NX_BLOCK = 5120                    # 32×256 tile


def ceil_div(a, b):
    return -(-a // b)


def tiles_32x256(rows, cols):
    """Nombre de tuiles Q4NX-style 32×256 (avec padding de colonnes)."""
    return (ceil_div(rows, Q4NX_ROW_CHUNK), ceil_div(cols, Q4NX_COL_CHUNK))


def tile_bytes_q4nx(n_row_tiles, n_col_tiles):
    return n_row_tiles * n_col_tiles * Q4NX_BLOCK


def mmul_tiles(m, k, n, tile=MMUL_INT8):
    """Nombre de tuiles 8x8x8 pour une matmul M×K×N (INT8)."""
    tm, tk, tn = tile
    return (ceil_div(m, tm), ceil_div(k, tk), ceil_div(n, tn))


def l1_footprint_int8(m, k, n):
    """Octets L1 pour une tuile (A M×K + B K×N + C M×N) en INT8 (1 B/élément)."""
    return m * k + k * n + m * n


def max_l1_tile(m, k, n, budget=L1_CORE_BYTES):
    """Sous-tuile (M',N') qui tient dans le budget L1 pour K fixé, ou tuple de sous-tiles.

    Contrainte : M'*K + K*N' + M'*N' <= budget (INT8, 1 B/élément).
    Retourne (nb_M, nb_N, footprint) : découpage de la tuile M×N en blocs qui tiennent.
    """
    best = None
    for m_sub in range(8, m + 1, 8):
        for n_sub in range(1, n + 1):
            fp = m_sub * k + k * n_sub + m_sub * n_sub
            if fp <= budget:
                best = (m_sub, n_sub, fp)
    if best is None:
        # K seul trop gros → découper K aussi (m_sub=8, n_sub=8, K' max)
        k_max = (budget // (8 + 8 + 1))
        k_max = max(1, (k_max // 8) * 8)
        return {"M": 8, "N": 8, "K": min(k, k_max), "footprint":
               8 * min(k, k_max) + min(k, k_max) * 8 + 64}
    m_sub, n_sub, fp = best
    return {"M": m_sub, "N": n_sub, "K": k, "footprint": fp,
            "sub_tiles_M": ceil_div(m, m_sub), "sub_tiles_N": ceil_div(n, n_sub)}


def map_expert(dims):
    """Map un expert complet (gate + up + down) vers les tuiles XDNA2."""
    shapes = expert_shape(dims)
    ppe = params_per_expert(dims)
    result = {"params_per_expert": ppe, "tensors": {}}
    for name, (m, k, n) in shapes.items():
        # N = num_experts (512) → un seul expert = slice [*, *, e]
        tm, tk, tn = mmul_tiles(m, k, 1)          # un expert (expert dim = 1 slice)
        r32, c32 = tiles_32x256(m, n)             # layout stockage Q4NX
        l1 = l1_footprint_int8(m, k, 1)
        sub = max_l1_tile(m, k, 1)
        result["tensors"][name] = {
            "shape": (m, k, n),
            "expert_slice": (m, k),
            "mmul_tiles_8x8x8": {"M": tm, "K": tk, "N": tn},
            "mmul_total_tiles": tm * tk * tn,
            "q4nx_tiles_32x256": {"rows": r32, "cols": c32},
            "q4nx_storage_bytes": tile_bytes_q4nx(r32, c32),
            "l1_int8_footprint_bytes": l1,
            "fits_l1": l1 <= L1_CORE_BYTES,
            "l1_subtile": sub,
        }
    # tailles par format
    result["bytes_per_format"] = {
        fmt: expert_bytes(ppe, fmt) for fmt in
        ["BF16", "Q8_0", "Q6_K", "Q4", "NVFP4", "INT8", "Q3", "Q2"]
    }
    return result


if __name__ == "__main__":
    dims = canonical_dims({})
    r = map_expert(dims)
    print("params/expert:", r["params_per_expert"])
    for name, t in r["tensors"].items():
        print(f"{name:5s} {t['shape']}  mmul_8x8x8={t['mmul_total_tiles']} tuiles "
              f"(M{t['mmul_tiles_8x8x8']['M']} K{t['mmul_tiles_8x8x8']['K']} N{t['mmul_tiles_8x8x8']['N']}) "
              f"fits_L1={t['fits_l1']} (footprint {t['l1_int8_footprint_bytes']} B <= 64KB)")
    print("\nbytes par format (un expert/layer) :")
    for fmt, b in r["bytes_per_format"].items():
        print(f"  {fmt:5s} = {b/(1024*1024):.2f} MiB")