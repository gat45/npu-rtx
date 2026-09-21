#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""microbench_h2d.py — calibre H2D pageable / pinned / pinned+async (+D2H).
À lancer EN PREMIER sur la machine cible 5070 (remplace h2d_pageable/h2d_pinned
ASSUMED de moe_axis_profiler.py — ancres MaxDam 6.5/20 GB/s).
Usage : py microbench_h2d.py [--sizes 1,4,16,64,256] [--iters 30]
Sortie : tableau GB/s par taille + JSON microbench_h2d.json (provenance incluse).
"""
import argparse
import json
import platform
import time


def bench_copy(src, dst, iters):
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    if src.is_cuda or dst.is_cuda:
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    if src.is_cuda or dst.is_cuda:
        import torch
        torch.cuda.synchronize()
    return (src.numel() * src.element_size() * iters) / (time.perf_counter() - t0) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,4,16,64,256")  # MiB
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "CUDA requis"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    sizes = [int(s) for s in args.sizes.split(",")]

    rows = []
    for mib in sizes:
        n = mib * 1024 * 1024
        gpu = torch.empty(n, dtype=torch.uint8, device=dev)
        cpu = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        cpu_pg = torch.empty(n, dtype=torch.uint8, pin_memory=False)
        r = {"size_mib": mib}
        r["h2d_pinned"] = bench_copy(cpu, gpu, args.iters)
        r["h2d_pageable"] = bench_copy(cpu_pg, gpu, args.iters)
        r["d2h_pinned"] = bench_copy(gpu, cpu, args.iters)
        r["d2h_pageable"] = bench_copy(gpu, cpu_pg, args.iters)
        rows.append(r)
        del gpu, cpu, cpu_pg

    print(f"== microbench H2D/D2H — {name} — {platform.node()} ==")
    hdr = f"{'MiB':>6} {'H2D pinned':>11} {'H2D pageable':>13} {'D2H pinned':>11} {'D2H pageable':>13}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['size_mib']:>6} {r['h2d_pinned']:>9.1f} GB/s {r['h2d_pageable']:>10.1f} GB/s"
              f" {r['d2h_pinned']:>9.1f} GB/s {r['d2h_pageable']:>10.1f} GB/s")

    # Taille expert 35B : 1.25 MB (NVFP4) / 2.45 MB (Q4) → le régime "petit transfert" importe
    print("\nNote: les experts MoE sont de PETITS transferts (1.25–2.45 MB) —")
    print("le GB/s à 1–4 MiB est la bonne ancre pour t_miss, pas le pic 256 MiB.")

    out = {"gpu": name, "host": platform.node(), "date": time.strftime("%Y-%m-%d %H:%M"),
           "rows": rows, "note": "ancres pour moe_axis_profiler h2d_pinned/h2d_pageable"}
    with open("microbench_h2d.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n-> microbench_h2d.json écrit (provenance: gpu + host + date)")


if __name__ == "__main__":
    main()
