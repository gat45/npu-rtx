# Étape 1 — Mesure capacité XDNA2 réelle (template)
# À exécuter sur machine avec amdxdna + XRT + profiler-v3

CAPACITES = {
    "xdna2_local_memory_mb": None,  # à mesurer via /sys/class/drm ou amdxdna ioctl
    "max_expert_q4_0_mb": 141,      # ~Q4_0 pour expert ~100M params (estimate)
    "dma_host_to_npu_us": 250,      # mesuré sur Strix Halo (estimation communauté)
    "bandwidth_npu_gbps": 226,      # iGPU-alike ; à vérifier séparément
    "thermal_throttle_t": 70,
}

def measure():
    # Appel au runtime : ggml-backend + amdxdna + profilage
    # Produit : xdna2_cap.json + xdna2_dma_profile.csv
    return CAPACITES
