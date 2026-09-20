#!/usr/bin/env python3
"""hw_discovery.py - detection hardware reelle (Phase B, collecteurs).

Detecte automatiquement : CPU, RAM, SSD, PCIe, GPU (NV), NPU (AMD XDNA2), versions.
Ecrit runs/<ts>/hw.json. Placeholders -> mesures reelles pour remplacer les constantes
statiques (memory_planner, conversion_matrix, bytes_per_token).

Utilise uniquement des outils standard (nvidia-smi, wmic/powershell, sys). Pas de DLL.
"""

import json
import os
import platform
import subprocess
import sys
import time


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
    parts = [p.strip() for p in out.split(",")]
    try:
        return {
            "name": parts[0], "vram_total_mib": int(parts[1].split()[0]),
            "vram_used_mib": int(parts[2].split()[0]), "driver": parts[3],
            "compute_cap": parts[4],
        }
    except (IndexError, ValueError):
        return {"raw": out}


def detect_cpu_ram():
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "?")
    try:
        import psutil
        ram_total = psutil.virtual_memory().total
        ram_avail = psutil.virtual_memory().available
        cores = psutil.cpu_count(logical=True)
        return {"cpu": cpu, "cores": cores, "ram_total_bytes": ram_total,
                "ram_avail_bytes": ram_avail, "ram_total_gib": round(ram_total/1024**3, 1),
                "ram_avail_gib": round(ram_avail/1024**3, 1)}
    except ImportError:
        return {"cpu": cpu, "note": "psutil absent -> RAM non mesuree"}


def detect_ssd():
    # Disques physiques + FS
    try:
        out = _run(["wmic", "diskdrive", "get", "Model,Size", "/format:csv"])
        disks = []
        for line in out.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1]:
                disks.append({"model": parts[1], "size_bytes": parts[2]})
        return {"disks": disks[:5]}
    except Exception:
        return {"note": "wmic indisponible"}


def detect_npu_amd():
    # AMD NPU : presence driver/accel + telemetry (placeholder - a completer avec
    # npu_perf_trace.sh / amdxdna telemetry quand dispo)
    hints = []
    for p in [r"C:\Windows\System32\DriverStore", r"C:\Windows\System32\amdvlk64.dll",
              r"C:\Program Files\AMD", r"C:\Program Files\RyzenAI"]:
        if os.path.exists(p):
            hints.append(p)
    return {"present_hints": hints,
            "note": "NPU AMD detecte via chemins ; telemetry XRT/amdxdna a ajouter (Phase B2)"}


def detect_versions():
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_driver": detect_nvidia()["driver"] if detect_nvidia() else "?",
    }


def detect_pcie():
    # Placeholder : a mesurer par microbench (ddr_pcie.py). Gen/link via lspci absent Windows.
    return {"note": "PCIe gen/link/lanes a mesurer par microbench (collectors/ssd_ddr_pcie.py)"}


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", ts)
    os.makedirs(out_dir, exist_ok=True)
    hw = {
        "timestamp": ts,
        "nvidia": detect_nvidia(),
        "cpu_ram": detect_cpu_ram(),
        "ssd": detect_ssd(),
        "npu_amd": detect_npu_amd(),
        "pcie": detect_pcie(),
        "versions": detect_versions(),
    }
    path = os.path.join(out_dir, "hw.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2, default=str)
    print(json.dumps(hw, indent=2, default=str))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()