#!/usr/bin/env python3
"""hw_discovery.py - source de verite materielle (Phase B, v2).

⚠️ CIBLE MATERIELLE : Ryzen 9 HX 365 (XDNA2) + RTX 5070 8 GB (Blackwell sm_120).
La machine dev (ex. GTX 1080 sm_61) sert UNIQUEMENT de test logique - ses valeurs
sont des ORDRES DE GRANDEUR, jamais des predictions pour la cible.

Produit un HardwareProfile ou chaque valeur porte sa provenance :
  MEASURED : mesure reelle (nvidia-smi, microbench, ...)
  DERIVED  : calculee depuis une mesure (ex. vram_available = total - wddm_reserve)
  ASSUMED  : hypothese documentee (a remplacer par mesure)
  UNKNOWN  : pas encore mesuree (ne jamais la traiter comme MEASURED)

Regle d'or : le Static Oracle ne transforme JAMAIS silencieusement ASSUMED en MEASURED.
Ecrit runs/<ts>/hw.json (avec provenance). Utilisable par feasibility + d2_planner.
"""

import json
import os
import platform
import subprocess
import sys
import time

TARGET = "Ryzen 9 HX 365 (XDNA2) + RTX 5070 8GB (Blackwell sm_120)"


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return r.stdout.strip()
    except Exception:
        return ""


def detect_nvidia():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                "--format=csv,noheader"])
    if not out:
        return None
    p = [x.strip() for x in out.split(",")]
    try:
        return {
            "name": {"v": p[0], "provenance": "MEASURED"},
            "vram_total_mib": {"v": int(p[1].split()[0]), "provenance": "MEASURED"},
            "vram_used_mib": {"v": int(p[2].split()[0]), "provenance": "MEASURED"},
            "driver": {"v": p[3], "provenance": "MEASURED"},
            "compute_cap": {"v": p[4], "provenance": "MEASURED"},
        }
    except (IndexError, ValueError):
        return {"raw": {"v": out, "provenance": "UNKNOWN"}}


def detect_cpu_ram():
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "?")
    try:
        import psutil
        ram_total = psutil.virtual_memory().total
        ram_avail = psutil.virtual_memory().available
        cores = psutil.cpu_count(logical=True)
        return {
            "cpu": {"v": cpu, "provenance": "MEASURED"},
            "cores": {"v": cores, "provenance": "MEASURED"},
            "ram_total_gib": {"v": round(ram_total/1024**3, 1), "provenance": "MEASURED"},
            "ram_avail_gib": {"v": round(ram_avail/1024**3, 1), "provenance": "MEASURED"},
        }
    except ImportError:
        return {"cpu": {"v": cpu, "provenance": "MEASURED"},
                "ram": {"v": None, "provenance": "UNKNOWN", "note": "psutil absent"}}


def detect_ssd():
    try:
        out = _run(["wmic", "diskdrive", "get", "Model,Size", "/format:csv"])
        disks = []
        for line in out.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1]:
                disks.append({"model": parts[1], "size_bytes": parts[2]})
        return {"disks": {"v": disks[:5], "provenance": "MEASURED"}}
    except Exception:
        return {"disks": {"v": None, "provenance": "UNKNOWN"}}


def detect_npu_amd():
    hints = [p for p in (r"C:\Program Files\AMD", r"C:\Program Files\RyzenAI") if os.path.exists(p)]
    return {
        "present": {"v": bool(hints), "provenance": "MEASURED"},
        "hints": {"v": hints, "provenance": "MEASURED"},
        "tile_geometry": {"v": None, "provenance": "UNKNOWN",
                          "note": "a mesurer (xrt-smi / amdxdna telemetry, Phase B2)"},
        "l1_capacity": {"v": 65536, "provenance": "ASSUMED", "source": "AMD docs 64KB core-local"},
        "dma_limits": {"v": None, "provenance": "UNKNOWN"},
        "measured_bandwidth": {"v": None, "provenance": "UNKNOWN",
                               "note": "BW DDR NPU 21.93 GB/s mesure FLM (reference)"},
    }


def detect_pcie():
    return {
        "link_gen": {"v": None, "provenance": "UNKNOWN"},
        "lanes": {"v": None, "provenance": "UNKNOWN"},
        "h2d_bw": {"v": None, "provenance": "UNKNOWN"},
        "d2h_bw": {"v": None, "provenance": "UNKNOWN"},
        "note": {"v": "a mesurer par microbench CUDA sur la machine cible (5070)", "provenance": "ASSUMED"},
    }


def detect_wddm():
    # Budget de residence variable (Microsoft WDDM) -> reserve prudente.
    return {"wddm_reservation_gib": {"v": 1.5, "provenance": "ASSUMED",
                                     "source": "Microsoft WDDM 2.0 (budget variable)"}}


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", ts)
    os.makedirs(out_dir, exist_ok=True)

    nv = detect_nvidia()
    gpu = {"nvidia": nv} if nv else {"nvidia": {"v": None, "provenance": "UNKNOWN"}}
    if nv and nv.get("vram_total_mib"):
        # DERIVED : vram_available = total - wddm_reserve
        total = nv["vram_total_mib"]["v"]
        reserve = detect_wddm()["wddm_reservation_gib"]["v"]
        gpu["vram_available_mib"] = {"v": total - int(reserve * 1024),
                                     "provenance": "DERIVED",
                                     "from": "vram_total - wddm_reservation"}

    profile = {
        "timestamp": ts,
        "target_hardware": TARGET,
        "dev_machine_note": "Ce profil = machine dev (test logique). La cible (HX365+5070 8GB) doit etre re-mesuree.",
        "cpu": detect_cpu_ram(),
        "gpu": gpu,
        "ssd": detect_ssd(),
        "xdna2": detect_npu_amd(),
        "pcie": detect_pcie(),
        "wddm": detect_wddm(),
        "runtime": {
            "os": {"v": platform.platform(), "provenance": "MEASURED"},
            "python": {"v": sys.version.split()[0], "provenance": "MEASURED"},
            "cuda_driver": {"v": nv["driver"]["v"] if nv else None, "provenance": "MEASURED"},
            "xrt": {"v": None, "provenance": "UNKNOWN"},
            "xdna_version": {"v": None, "provenance": "UNKNOWN"},
        },
        "legend": ["MEASURED", "DERIVED", "ASSUMED", "UNKNOWN"],
    }
    path = os.path.join(out_dir, "hw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)
    print(json.dumps(profile, indent=2, default=str))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()