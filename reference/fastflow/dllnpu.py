"""
FastFlowLM — Q4 GEMM Test Suite
================================
Objectif  : Valider le kernel Custom_Q4_GEMM_Optimized (deqUant.dll)
            end-to-end avec vrais poids, comparaison Python vs DLL,
            et benchmark de performance.
Plateforme: Windows (AMD Ryzen AI / Strix), Python 3.9+
Dépendances: numpy, ctypes (stdlib)
"""

import json
import ctypes
import os
import sys
import time
import struct
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
FASTFLOW_DIR = r"C:\FastFlow_Strix"
RYZENAI_DIR  = r"C:\FastFlow\ryzenai_compiler\q4_cache"

PROXY_DLL    = os.path.join(FASTFLOW_DIR, "deqUant.dll")
META_JSON    = os.path.join(RYZENAI_DIR,  "layer_0_meta.json")
WEIGHTS_BIN  = os.path.join(RYZENAI_DIR,  "layer_0_q4weights.bin")

BENCH_RUNS   = 100          # Nombre d'itérations pour le benchmark
TOL_ABS      = 1e-3         # Tolérance absolue pour la validation numérique
TOL_REL      = 1e-2         # Tolérance relative

BANNER = "=" * 70


def banner(title: str):
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def step(n: int, total: int, msg: str):
    print(f"\n[{n}/{total}] {msg}")


def ok(msg: str):
    print(f"   ✅  {msg}")


def fail(msg: str):
    print(f"   ❌  {msg}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 1. VÉRIFICATION DES FICHIERS
# ──────────────────────────────────────────────────────────────────────────────
banner("FASTFLOWLM — Q4 GEMM END-TO-END VALIDATION")

TOTAL_STEPS = 8
step(1, TOTAL_STEPS, "Vérification des fichiers...")

for path, name in [
    (PROXY_DLL,   "deqUant.dll"),
    (META_JSON,   "layer_0_meta.json"),
    (WEIGHTS_BIN, "layer_0_q4weights.bin"),
]:
    if not os.path.exists(path):
        fail(f"{name} introuvable → {path}")
    ok(f"{name}  ({os.path.getsize(path):,} bytes)")


# ──────────────────────────────────────────────────────────────────────────────
# 2. CHARGEMENT DE LA DLL + SIGNATURE DE LA FONCTION
# ──────────────────────────────────────────────────────────────────────────────
step(2, TOTAL_STEPS, "Chargement de deqUant.dll et configuration ctypes...")

try:
    dll = ctypes.WinDLL(PROXY_DLL)
except OSError as e:
    fail(f"WinDLL: {e}")

# Prototype C attendu :
#   int Custom_Q4_GEMM_Optimized(
#       const uint8_t* q4_weights,   // poids Q4 packés (2 valeurs/byte)
#       const float*   activations,  // vecteur d'entrée [K]
#       float*         output,       // vecteur de sortie [M]
#       int            M,
#       int            K,
#       float          scale,
#       int            zero_point
#   );
#
# ⚠️  Si votre DLL a une signature différente, adaptez uniquement ce bloc.
try:
    fn = dll.Custom_Q4_GEMM_Optimized
    fn.restype  = ctypes.c_int
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # q4_weights
        ctypes.POINTER(ctypes.c_float),   # activations
        ctypes.POINTER(ctypes.c_float),   # output
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_float,                   # scale
        ctypes.c_int,                     # zero_point
    ]
    ok("Signature ctypes configurée")
except AttributeError:
    fail("Fonction Custom_Q4_GEMM_Optimized introuvable dans la DLL.\n"
         "   Vérifiez l'export avec : dumpbin /EXPORTS deqUant.dll")


# ──────────────────────────────────────────────────────────────────────────────
# 3. LECTURE DES MÉTADONNÉES
# ──────────────────────────────────────────────────────────────────────────────
step(3, TOTAL_STEPS, "Lecture des métadonnées Q4...")

with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

M       = int(meta["wts_shape_dim_0"])
K       = int(meta["wts_shape_dim_1"])
scale   = float(meta["wts_scale"])
zp      = int(meta["wts_zp"])
packed  = bool(meta.get("wts_packed", True))

print(f"   Node      : {meta.get('node_name', 'N/A')}")
print(f"   Shape     : {M} × {K}")
print(f"   Scale     : {scale}")
print(f"   ZeroPoint : {zp}")
print(f"   Packed    : {packed}")

expected_bytes = (M * K + 1) // 2   # 2 valeurs 4-bit par byte
ok(f"Métadonnées OK — {expected_bytes} bytes de poids attendus")


# ──────────────────────────────────────────────────────────────────────────────
# 4. LECTURE ET DÉCODAGE DES POIDS Q4
# ──────────────────────────────────────────────────────────────────────────────
step(4, TOTAL_STEPS, "Lecture et décodage des poids Q4...")

with open(WEIGHTS_BIN, "rb") as f:
    raw_bytes = f.read()

if len(raw_bytes) < expected_bytes:
    fail(f"Fichier trop court : {len(raw_bytes)} bytes lus, "
         f"{expected_bytes} attendus")

# Décodage 4-bit → tableau uint8 (little-nibble en premier, comme GGML/llama.cpp)
q4_bytes_arr = np.frombuffer(raw_bytes[:expected_bytes], dtype=np.uint8)
lo = (q4_bytes_arr & 0x0F).astype(np.uint8)
hi = (q4_bytes_arr >> 4).astype(np.uint8)

# Interleave : [lo0, hi0, lo1, hi1, ...]
q4_unpacked = np.empty(len(lo) + len(hi), dtype=np.uint8)
q4_unpacked[0::2] = lo
q4_unpacked[1::2] = hi
q4_unpacked = q4_unpacked[:M * K]

print(f"   Raw bytes lus : {len(raw_bytes):,}")
print(f"   Valeurs 4-bit : {len(q4_unpacked)} ({M}×{K})")
ok("Poids décodés")


# ──────────────────────────────────────────────────────────────────────────────
# 5. DÉQUANTIFICATION PYTHON (RÉFÉRENCE NUMÉRIQUE)
# ──────────────────────────────────────────────────────────────────────────────
step(5, TOTAL_STEPS, "Calcul de référence Python (déquantification + GEMV)...")

weights_f32 = scale * (q4_unpacked.astype(np.float32) - zp)   # [M*K]
W = weights_f32.reshape(M, K)

# Vecteur d'activation de test (reproductible)
rng = np.random.default_rng(seed=42)
activations = rng.standard_normal(K).astype(np.float32) * 0.1

# GEMV de référence : output[m] = Σ_k W[m,k] * act[k]
ref_output = W @ activations

print(f"   Activation  : [{activations[:4]} ...] shape=({K},)")
print(f"   Sortie réf  : [{ref_output[:4]} ...] shape=({M},)")
print(f"   |max|={np.abs(ref_output).max():.6f}  "
      f"mean={ref_output.mean():.6f}  std={ref_output.std():.6f}")
ok("Référence Python calculée")


# ──────────────────────────────────────────────────────────────────────────────
# 6. APPEL DU KERNEL DLL (AVEC VRAIS BUFFERS)
# ──────────────────────────────────────────────────────────────────────────────
step(6, TOTAL_STEPS, "Appel du kernel Custom_Q4_GEMM_Optimized (DLL)...")

# Buffers ctypes partagés avec numpy (zero-copy)
buf_weights = q4_bytes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
buf_act     = activations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dll_output  = np.zeros(M, dtype=np.float32)
buf_out     = dll_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

try:
    ret = fn(buf_weights, buf_act, buf_out, M, K, scale, zp)
    if ret != 0:
        fail(f"Le kernel a retourné le code d'erreur {ret}")
    ok(f"Kernel exécuté (ret={ret})")
except Exception as e:
    fail(f"Exception lors de l'appel DLL : {e}")

print(f"   Sortie DLL  : [{dll_output[:4]} ...] shape=({M},)")


# ──────────────────────────────────────────────────────────────────────────────
# 7. VALIDATION NUMÉRIQUE
# ──────────────────────────────────────────────────────────────────────────────
step(7, TOTAL_STEPS, "Validation numérique Python vs DLL...")

abs_diff = np.abs(dll_output - ref_output)
max_diff = abs_diff.max()
mse      = np.mean((dll_output - ref_output) ** 2)
rel_err  = (abs_diff / (np.abs(ref_output) + 1e-9)).max()

print(f"   Max diff abs : {max_diff:.2e}  (seuil={TOL_ABS:.2e})")
print(f"   MSE          : {mse:.2e}")
print(f"   Max err rel  : {rel_err:.2e}  (seuil={TOL_REL:.2e})")

worst_idx = int(np.argmax(abs_diff))
print(f"   Pire index   : [{worst_idx}]  "
      f"ref={ref_output[worst_idx]:.6f}  dll={dll_output[worst_idx]:.6f}")

if max_diff < TOL_ABS and rel_err < TOL_REL:
    ok(f"Validation RÉUSSIE — kernel numériquement correct ✓")
else:
    print(f"\n   ⚠️  Les sorties divergent au-delà des tolérances.")
    print(f"       Vérifiez l'ordre des nibbles (lo/hi) et le signe de zp.")


# ──────────────────────────────────────────────────────────────────────────────
# 8. BENCHMARK DE PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
step(8, TOTAL_STEPS, f"Benchmark — {BENCH_RUNS} itérations...")

# Warmup
for _ in range(5):
    fn(buf_weights, buf_act, buf_out, M, K, scale, zp)

t0 = time.perf_counter()
for _ in range(BENCH_RUNS):
    fn(buf_weights, buf_act, buf_out, M, K, scale, zp)
t1 = time.perf_counter()

total_s      = t1 - t0
avg_ms       = (total_s / BENCH_RUNS) * 1e3
throughput   = BENCH_RUNS / total_s
# OPs : M*K multiplications + M*K additions = 2*M*K par appel
gops_per_run = 2 * M * K / 1e9
gops_s       = gops_per_run * throughput

print(f"   Runs         : {BENCH_RUNS}")
print(f"   Temps total  : {total_s*1e3:.2f} ms")
print(f"   Latence moy. : {avg_ms:.4f} ms/call")
print(f"   Throughput   : {throughput:.1f} calls/s")
print(f"   GOPs/s       : {gops_s:.4f}")
ok("Benchmark terminé")


# ──────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ──────────────────────────────────────────────────────────────────────────────
banner("RÉSUMÉ")
print(f"  Shape         : {M} × {K}")
print(f"  Max diff abs  : {max_diff:.2e}")
print(f"  MSE           : {mse:.2e}")
print(f"  Latence       : {avg_ms:.4f} ms/call")
print(f"  Throughput    : {throughput:.1f} calls/s")
validation_str = "RÉUSSIE ✅" if max_diff < TOL_ABS else "ÉCHOUÉE ❌"
print(f"  Validation    : {validation_str}")
print(BANNER)
