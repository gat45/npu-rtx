# ANALYSE FLAGS XRT — Comparaison 0x17 vs 0x1F

## Flag 0x17 (Actuel — Stable)
```
0x17 = 0001 0111
Bits: 0, 1, 2, 4 activés

- Bit 0 (0x01): HOST_ONLY
- Bit 1 (0x02): CACHEABLE  
- Bit 2 (0x04): P2P (NPU ↔ iGPU Direct)
- Bit 4 (0x10): CONTIGUOUS (Mémoire contiguë DMA)
```

## Flag 0x1F (Candidat — À tester)
```
0x1F = 0001 1111
Bits: 0, 1, 2, 4, 8 activés

Même que 0x17, PLUS:
- Bit 3 (0x08): NON_BLOCKING / ASYNC (?)
  → Permet au NPU de continuer à travailler sans attendre la réponse du CPU
  → Pourrait éliminer les "bulles" entre les couches

OU selon d'autres sources:
- Bit 3 (0x08): LOW_LATENCY
  → Priorité maximale sur le bus mémoire
  → Réduit la latence d'accès aux poids du modèle
```

## Autres Flags Potentiels

| Flag | Hex | Bits | Effet Supposé | Risque |
|------|-----|------|---------------|--------|
| 0x17 | 23 | 10111 | Stable actuel (P2P+Contiguous) | ✅ Aucun |
| 0x1F | 31 | 11111 | + Async/Low Latency | ⚠️ Moyen |
| 0x37 | 55 | 110111 | + ??? | ❌ Inconnu |
| 0x97 | 151 | 10010111 | SVME + tout le reste | ⚠️ Testé V12 (stable) |
| 0xB5 | 181 | 10110101 | GU_PRE + P2P + Cache | ✅ Testé V13 (stable) |

## Stratégie de Test

1. **V18.1 — Flag 0x1F (Async)**
   - Même alignement 152B que V14.1
   - Change juste 0x17 → 0x1F
   - Si ça marche: on débloque le mode asynchrone
   - Gain potentiel: +10-15% decode (moins d'attente CPU)

2. **V18.2 — Flag 0x97 (SVME Full Stack)**
   - Celui de V12/V14 — déjà testé et stable
   - 0x97 = HOST|PINNED|CACHE|P2P|SVME
   - Le SVME active la mémoire virtuelle partagée CPU/GPU/NPU
   - Gain potentiel: déjà mesuré à 46.0 t/s prefill

3. **V18.3 — Flag 0xB5 (Radeon Pusher)**
   - Celui de V13 — déjà testé et stable
   - 0xB5 = HOST|PINNED|CACHE|P2P|GU_PRE|SVME
   - Force le moteur Compute de l'iGPU
   - Gain potentiel: 43.0 t/s prefill

## Recommandation

Tester 0x1F d'abord (le plus risqué mais plus gros gain potentiel).
Si crash → revenir à 0x17 (stable) ou 0x97 (V14 God Mode).
