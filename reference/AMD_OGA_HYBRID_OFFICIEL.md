# Référence — AMD OGA Hybrid (officiel) — source ryzenai.docs.amd.com/en/main/hybrid_oga.html
# Fetched 2026-09-20 · Ryzen AI Software 1.8.0

## FAIT MAJEUR (corrige le RAPPORT initial)
AMD documente désormais OFFICIELLEMENT un mode LLM hybride NPU+iGPU dans le flux OGA
(OnnxRuntime GenAI) — version 1.8.0, OGA 0.14.0.

Modes d'exécution OGA :
- **Hybrid** : utilise NPU **ET** iGPU pour optimiser TTFT et TPS (prefill/decode partitionnés)
- **NPU-only** : NPU exclusif (2 types de modèles NPU)

Types de modèles NPU :
| Type | Max context (in+out) | Usage |
|------|----------------------|-------|
| Token Fusion | jusqu'à 16K tokens | long-context |
| Full Fusion | jusqu'à 4096 tokens | best throughput, séquences courtes |

## Processeurs supportés
- Strix Point et Krackan Point ✅ (donc notre Ryzen AI 9 365 = Strix Point ✅)
- Phoenix (PHX) / Hawk (HPT) ❌ NON supportés

## Modèles pré-optimisés AMD (à convertir, pas GGUF)
Llama-2/3, Mistral, DeepSeek-Distill, Qwen-2/2.5/3, Gemma-2/3, GPT-OSS, Phi-3/3.5/4.
Collections HF : amd/ryzen-ai-180-hybrid · ryzen-ai-180-npu-16k · ryzen-ai-180-npu-4k.

## Config importante pour le planner
- Performance mode : `cd C:\Windows\System32\AMD` puis `xrt-smi configure --pmode performance`
  (confirme le +44% TPS évoqué par la communauté)
- Long context hybride : dans genai_config.json ajouter
  `"hybrid_opt_chunk_context": "1"`, `"hybrid_opt_max_seq_length": "4096"`,
  `"hybrid_opt_free_after_prefill": "1"` (provider RyzenAI) + `"chunk_size": 2048` (search)
- Clé pour le planner : `hybrid_opt_free_after_prefill` = le NPU peut être libéré
  après le prefill → NPU dispo pour autre chose (decode GPU, ou experts MoE)

## Implications pour D2 Planner
1. Le chemin officiel AMD n'est PAS GGUF → NPU : ce sont des modèles ONNX pré-optimisés
   (hybrid ou NPU). Notre chemin GGUF→XDNA2 (repo xdna2-) reste communautaire = différent.
2. Le partitionnement NPU/iGPU officiel est **statique par phase** (prefill/decode).
   Notre MoE expert-level est dynamique = au-delà de l'officiel AMD.
3. `hybrid_opt_free_after_prefill` ouvre une fenêtre où le NPU est libre pendant le decode
   → candidat pour absorber les experts MoE en overflow pendant que le GPU décode. C'est
   exactement le slot que le planner doit cibler.