#!/usr/bin/env python3
"""ssd_ddr_pcie.py - microbench des transferts 1K -> 1G (Phase B).

Mesure SSD seq/random, memcpy DDR, et (si GPU dispo) H2D/D2H via une astuce :
ecrire/lire un gros buffer depuis numpy -> la courbe BW(size) remplace les placeholders
de memory_planner / conversion_matrix / bytes_per_token.

NOTE : H2D/D2H precis = benchmark CUDA/nvidia-smi ; ici on mesure memcpy CPU (DDR)
et lecture disque (SSD). Le PCIe GPU sera mesure sur la machine cible (5070).
"""

import os
import tempfile
import time


SIZES = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 1 * 1024 * 1024,
         4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024]


def bench_memcpy():
    print("== DDR memcpy (numpy, ~GB/s) ==")
    import numpy as np
    res = []
    for n in SIZES:
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)
        # warmup
        b[:] = a
        t0 = time.perf_counter()
        reps = max(1, 64 * 1024 * 1024 // n)
        for _ in range(reps):
            b[:] = a
        dt = time.perf_counter() - t0
        gbps = (n * reps) / dt / 1e9
        res.append({"size": n, "gbps": round(gbps, 1)})
        print(f"  {n/1024:8.0f} KiB : {gbps:6.1f} GB/s")
    return res


def bench_ssd():
    print("\n== SSD seq read (~GB/s) ==")
    res = []
    with tempfile.TemporaryDirectory() as d:
        fpath = os.path.join(d, "bench.bin")
        with open(fpath, "wb") as f:
            f.write(os.urandom(64 * 1024 * 1024))
        for n in SIZES:
            data = bytearray(n)
            with open(fpath, "rb") as f:
                f.read(n)  # warm
            t0 = time.perf_counter()
            reps = max(1, 64 * 1024 * 1024 // n)
            for _ in range(reps):
                with open(fpath, "rb") as f:
                    f.readinto(data)
            dt = time.perf_counter() - t0
            gbps = (n * reps) / dt / 1e9
            res.append({"size": n, "gbps": round(gbps, 2)})
            print(f"  {n/1024:8.0f} KiB : {gbps:5.2f} GB/s")
    return res


if __name__ == "__main__":
    m = bench_memcpy()
    s = bench_ssd()
    print("\nA utiliser pour remplacer les placeholders (memory_planner etc.)")
    print("memcpy:  min=%.1f max=%.1f GB/s" % (min(r['gbps'] for r in m), max(r['gbps'] for r in m)))
    print("ssd:     min=%.2f max=%.2f GB/s" % (min(r['gbps'] for r in s), max(r['gbps'] for r in s)))