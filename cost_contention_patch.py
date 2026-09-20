# Étape 3 — Métrique contention mémoire (overlay cost_model)
# À injecter dans governor/cost_model.py update() via import

def contention_cost(bw_used_rtx, bw_used_npu, bw_used_cpu, total_bw=226):
    # BW GB/s utilisée par chaque tier (estimée depuis profile)
    contention = (bw_used_rtx + bw_used_npu + bw_used_cpu) / total_bw
    # Pénalité si > 0.7 (seuil communauté Strix Halo)
    return contention**2 * 0.25 if contention > 0.7 else 0.0

# Intégration : coût total = coût_op + contention_cost(...)
