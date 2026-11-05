# -*- coding: utf-8 -*-
"""
Benchmark BLAS / LAPACK : DGEMM (niveau 3 BLAS) et DGESV (LAPACK, LU) sur de grandes matrices.
Usage : python benchmark_blas.py
"""

import os, sys, time, json, platform
import numpy as np
from scipy.linalg import lapack, blas
from threadpoolctl import threadpool_info, threadpool_limits

rng = np.random.default_rng(2026)

def best_time(fn, rep=3):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)

def matmul_naive(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    Bt = list(zip(*B))
    for i in range(n):
        Ai = A[i]
        for j in range(n):
            C[i][j] = sum(a * b for a, b in zip(Ai, Bt[j]))
    return C

def gauss_naive(A, b):
    n = len(A)
    A = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(A[i][k]))
        A[k], A[p] = A[p], A[k]
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n + 1):
                A[i][j] -= f * A[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x

if __name__ == '__main__':
    info = threadpool_info()
    backend = ", ".join(sorted({f"{d.get('internal_api')} ({d.get('version')})" for d in info if d.get('user_api') == 'blas'}))
    ncpu = os.cpu_count()
    print("Bibliotheque BLAS/LAPACK active :", backend, "| coeurs logiques :", ncpu)

    sizes = [100, 200, 400, 800, 1600]
    for n in sizes:
        A = np.asfortranarray(rng.standard_normal((n, n)))
        B = np.asfortranarray(rng.standard_normal((n, n)))
        b = rng.standard_normal(n)
        
        t_gemm = best_time(lambda: blas.dgemm(1.0, A, B))
        gflops_gemm = 2 * n**3 / t_gemm / 1e9
        
        t_gesv = best_time(lambda: lapack.dgesv(A.copy(order="F"), b))
        gflops_gesv = (2/3) * n**3 / t_gesv / 1e9
        
        print(f"n={n:5d} | DGEMM: {t_gemm*1e3:8.2f} ms ({gflops_gemm:6.1f} GFLOPS) | DGESV: {t_gesv*1e3:8.2f} ms ({gflops_gesv:6.1f} GFLOPS)")
