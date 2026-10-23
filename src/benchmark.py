"""
Module de benchmark haute performance pour BLAS et LAPACK.
Compare le code Python pur naïf face aux micro-noyaux optimisés OpenBLAS et Intel MKL.
Mesure les débits en GFLOPS et la scalabilité OpenMP multithread.
"""

from typing import Dict, List, Tuple
import time
import numpy as np
from scipy.linalg import lapack, blas

try:
    from threadpoolctl import threadpool_limits, threadpool_info
except ImportError:
    import contextlib
    @contextlib.contextmanager
    def threadpool_limits(limits=None, user_api=None):
        yield
    def threadpool_info():
        return [{'user_api': 'blas', 'internal_api': 'SciPy OpenBLAS/MKL (Native)', 'version': 'N/A'}]


def get_blas_info() -> Dict[str, str]:
    """
    Retourne les informations sur la bibliothèque BLAS/LAPACK active en mémoire.
    """
    info = threadpool_info()
    backends = [
        f"{d.get('internal_api', 'unknown')} ({d.get('version', '')})"
        for d in info if d.get('user_api') == 'blas'
    ]
    return {
        "backend": ", ".join(backends) if backends else "SciPy BLAS/LAPACK",
        "num_threads": str(len(info)),
    }


def time_function(fn, repetitions: int = 3) -> float:
    """
    Mesure le temps d'exécution minimal sur un nombre de répétitions (warm-up inclus).
    """
    # Warm-up
    fn()
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def matmul_naive(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Produit matriciel naïf O(n^3) en pur Python interprété (sans vectorisation).
    """
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    B_T = list(zip(*B))
    for i in range(n):
        for j in range(n):
            C[i][j] = sum(a * b for a, b in zip(A[i], B_T[j]))
    return C


def benchmark_dgemm(matrix_sizes: List[int], threads: int = 4) -> List[Dict[str, float]]:
    """
    Benchmark de la routine BLAS 3 DGEMM (2 * n^3 FLOPs).
    """
    results = []
    with threadpool_limits(limits=threads, user_api='blas'):
        for n in matrix_sizes:
            rng = np.random.default_rng(2026)
            A = np.asfortranarray(rng.standard_normal((n, n)), dtype=np.float64)
            B = np.asfortranarray(rng.standard_normal((n, n)), dtype=np.float64)
            
            t_blas = time_function(lambda: blas.dgemm(1.0, A, B))
            gflops = (2.0 * (n ** 3)) / (t_blas * 1e9)
            
            results.append({
                "n": n,
                "time_ms": t_blas * 1000.0,
                "gflops": gflops,
                "threads": threads,
            })
    return results


def benchmark_dgesv(matrix_sizes: List[int], threads: int = 4) -> List[Dict[str, float]]:
    """
    Benchmark de la routine LAPACK DGESV (2/3 * n^3 FLOPs).
    """
    results = []
    with threadpool_limits(limits=threads, user_api='blas'):
        for n in matrix_sizes:
            rng = np.random.default_rng(2026)
            A = np.asfortranarray(rng.standard_normal((n, n)), dtype=np.float64)
            b = rng.standard_normal(n)
            
            t_lapack = time_function(lambda: lapack.dgesv(A.copy(order="F"), b))
            gflops = ((2.0 / 3.0) * (n ** 3)) / (t_lapack * 1e9)
            
            results.append({
                "n": n,
                "time_ms": t_lapack * 1000.0,
                "gflops": gflops,
                "threads": threads,
            })
    return results


def benchmark_thread_scaling(matrix_size: int = 1600, thread_counts: List[int] = [1, 2, 4]) -> List[Dict[str, float]]:
    """
    Évalue le passage à l'échelle (speedup et efficacité parallèle) en variant le nombre de threads OpenMP.
    """
    rng = np.random.default_rng(2026)
    A = np.asfortranarray(rng.standard_normal((matrix_size, matrix_size)), dtype=np.float64)
    B = np.asfortranarray(rng.standard_normal((matrix_size, matrix_size)), dtype=np.float64)
    
    # Référence 1 thread
    with threadpool_limits(limits=1, user_api='blas'):
        t_base = time_function(lambda: blas.dgemm(1.0, A, B))
        
    scaling_data = []
    for th in thread_counts:
        with threadpool_limits(limits=th, user_api='blas'):
            t_curr = time_function(lambda: blas.dgemm(1.0, A, B))
            speedup = t_base / t_curr
            efficiency = (speedup / th) * 100.0
            gflops = (2.0 * (matrix_size ** 3)) / (t_curr * 1e9)
            scaling_data.append({
                "threads": th,
                "time_ms": t_curr * 1000.0,
                "speedup": speedup,
                "efficiency": efficiency,
                "gflops": gflops,
            })
    return scaling_data
