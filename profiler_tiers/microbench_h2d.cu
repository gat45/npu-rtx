// microbench_h2d.cu — H2D/D2H pageable vs pinned, GTX 1080 (sm_61) — 2026-09-21
// Remplace les axes UNKNOWN h2d_pinned/h2d_pageable de AXES_1080.md
// Build : nvcc -O2 -arch=sm_61 microbench_h2d.cu -o microbench_h2d.exe
// Usage : microbench_h2d.exe [taille_max_mib]
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// bench une direction : retourne le meilleur GB/s sur reps itérations
static double bench(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind, int reps) {
    double best_gbs = 0.0;
    for (int it = 0; it < reps; ++it) {
        double t0 = now_ms();
        cudaError_t e = cudaMemcpy(dst, src, bytes, kind);
        if (e != cudaSuccess) return -1.0;
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) return -1.0;
        double gbs = bytes / ((now_ms() - t0) / 1e3);
        if (gbs > best_gbs) best_gbs = gbs;
    }
    return best_gbs;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA_ERR %s @line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main(int argc, char ** argv) {
    int max_mib = (argc > 1) ? atoi(argv[1]) : 256;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("provenance: gpu=%s sm=%d.%d vram=%.1f GiB | CUDA %d.%d | pcie_link_gen=%d width=%d\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1073741824.0,
           CUDART_VERSION / 1000, (CUDART_VERSION / 10) % 100,
           0, 0); // gen/width via nvidia-smi séparément
    printf("format: taille_MiB h2d_pageable_GBs h2d_pinned_GBs d2h_pageable_GBs d2h_pinned_GBs\n");

    const size_t sizes_mib[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    for (size_t smi : sizes_mib) {
        if ((int)smi > max_mib) break;
        size_t bytes = smi * 1048576;
        char * h_pg = (char *)malloc(bytes);
        char * h_pn = nullptr; CK(cudaMallocHost(&h_pn, bytes));
        char * d    = nullptr; CK(cudaMalloc(&d, bytes));

        // remplir pour éviter copie-zéro / compression
        for (size_t i = 0; i < bytes; i += 4096) h_pg[i] = (char)(i >> 12);
        memcpy(h_pn, h_pg, bytes);

        const int reps = (bytes <= 8u << 20) ? 200 : 50;
        double pg_h2d = bench(d, h_pg, bytes, cudaMemcpyHostToDevice, reps);
        double pn_h2d = bench(d, h_pn, bytes, cudaMemcpyHostToDevice, reps);
        double pg_d2h = bench(h_pg, d, bytes, cudaMemcpyDeviceToHost, reps);
        double pn_d2h = bench(h_pn, d, bytes, cudaMemcpyDeviceToHost, reps);

        printf("%8zu %15.2f %15.2f %15.2f %15.2f\n", smi,
               pg_h2d / 1e9, pn_h2d / 1e9, pg_d2h / 1e9, pn_d2h / 1e9);
        free(h_pg); CK(cudaFreeHost(h_pn)); CK(cudaFree(d));
    }
    printf("[OK] microbench_h2d\n");
    return 0;
}
