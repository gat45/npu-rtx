# Étape 4 — Validation llama.cpp + OllamaAMDNPU / XDNA backend
# Référence : xdna2-forensics/downloads/sources/llama-upstream (profiler_v3)
# Commandes à adapter selon build (NDK r27c, -GGML_HEXAGON=OFF, -GGML_OPENCL=ON si besoin)

BUILD_DIR = "E:/oneplus/ab-build-xdna/"  # à créer après clone / patch
SOURCE = "xdna2-forensics/downloads/sources/llama-upstream"

MODEL = "Qwen3.5-9B-Q4_0.gguf"  # ou sous-ensemble MoE pour test rapide

CMD_XDNA = (
    f"{BUILD_DIR}/llama-example -m {MODEL} "
    f"--device xdna2 -ngl 99 -n 32 --spec-type draft-mtp 2> xdna2_run.log"
)

CMD_CPU_FALLBACK = (
    f"{BUILD_DIR}/llama-example -m {MODEL} "
    f"-ngl 0 -n 32 > cpu_fallback.log"
)

def validate():
    # Mesure tok/s, vérifie backend engagé (pas fallback silencieux)
    # Critère : n_backends > 1, mirror=0, repack>0, dev identifié
    pass
