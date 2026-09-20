# Référence externe — 1bit-MONSTER / 1bit-MONSTER
# URL : https://github.com/1bit-MONSTER/1bit-MONSTER
# Licence : GPL-3.0 · 3 107 commits · 19 stars · Language C++26

## Ce que c'est
Moteur d'inférence LLM "model-agnostic" en un seul binaire C++26, zéro Python au runtime.
Détecte l'archi du modèle (GGUF, 1BP, ONNX, H1B, safetensors) et le route automatiquement
vers le meilleur backend : **NPU XDNA 2 (reverse-engineered en 4 jours), GPU (HIP/CUDA/Metal/Vulkan), CPU**.
Pas de config, pas de glue par modèle, pas d'interpréteur.

- 569 tokens d'architecture → 2 036 strings HuggingFace ; 326 034/326 277 checkpoints HF couverts (99.93%)
- Validé aujourd'hui : AMD Strix Halo (gfx1151, Ryzen AI Max+ 395 / Radeon 8060S)
- CUDA compilé mais non exécuté sur NVIDIA réel ; Vulkan/Metal partiels
- Format propriétaire **1BP** pour les poids NPU (à télécharger séparément, sha256 vérifié)
- Familles modèles auto-détectées : Zyphra (Zaya, Zamba2, BlackMamba), Qwen, Llama, Mistral, Gemma, Phi, Falcon, OLMo, Granite, SmolLM, DeepSeek, GPT-OSS, Laguna, Kimi, BitNet/Bonsai, Whisper
- **JARVIS** : pipeline vocal local complet (mic→STT→LLM→TTS→speaker) dans le même binaire

## Intérêt pour le D2 Planner (xdna2)
1. **Routing auto backend (NPU/GPU/CPU) = exactement ce que veut le planner** — preuve qu'une
   exécution multi-backend dynamique est possible et déjà démontrée sur XDNA2.
2. **Format 1BP pour NPU** — le planner doit traiter ce format (et le Q4_0 GGUF) pour savoir
   ce qui peut aller sur XDNA2. La quantification des poids NPU est un choix de pré-encodage.
3. **Model-agnostic + auto-détection** — confirme l'idée de `place_expert` : on peut choisir
   le backend par modèle/par op, pas par modèle fixé.
4. **XDNA2 reverse-engineered** — le backend NPU XDNA2 n'est PAS un SDK AMD fermé : il peut être
   réimplémenté (ce que le repo `xdna2-` fait aussi pour Qwen3.5-9B sur Ryzen AI 9 365).

## Différence clé avec notre machine (Ryzen AI 9 365 / RTX 5070)
- 1bit-MONSTER est validé sur **Strix Halo** (mémoire unifiée iGPU+CPU+NPU) ; notre machine a
  un **RTX 5070 8GB discret + NPU XDNA2 Strix Point**.
- Le routing 1bit (NPU↔GPU↔CPU) est le squelette du planner, mais il faut y injecter la
  **contention mémoire + coût DMA** (fait dans cost_contention_patch) car ici GPU et NPU ne
  partagent pas le même contrôleur de la même façon que Strix Halo.

## Pour aller plus loin
- Docs : docs/guides/architecture.md, docs/wiki/performance.md, docs/journey.md (reverse XDNA2)
- Format 1BP : packaging/model-download.sh (liste modèles pré-convertis)
- Communauté : Discord, Fluxer (fluxer.gg)