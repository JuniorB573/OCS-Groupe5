"""
Module de benchmark comparatif haute performance pour le modèle VIH.
Confronte les méthodes lentes naïves (Python pur) aux micro-noyaux BLAS 3, LAPACK et Intel MKL.
Mesure :
- Temps d'exécution (ms) et facteurs d'accélération (Speedup).
- Débits en GFLOPS et débit de patients simulés / seconde.
- Passage à l'échelle multithread OpenMP / Intel MKL.
"""

from typing import Dict, List, Tuple, Any
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

from .cohort_model import (
    DEFAULT_PARAMS,
    build_cohort_parameters,
    initialize_cohort_state,
    simulate_cohort_rk4,
)
from .naive_model import simulate_cohort_naive_rk4
from .lapack_analysis import solve_ridge_cholesky_lapack, solve_ridge_naive_inverse


def get_hpc_system_info() -> Dict[str, str]:
    """
    Détecte la bibliothèque BLAS/LAPACK active en mémoire (Intel MKL, OpenBLAS, Accelerate).
    """
    info = threadpool_info()
    backends = [
        f"{d.get('internal_api', 'unknown')} ({d.get('version', '')})"
        for d in info if d.get('user_api') == 'blas'
    ]
    return {
        "blas_backend": ", ".join(backends) if backends else "SciPy BLAS/LAPACK",
        "detected_threads": str(len(info)),
    }


def measure_execution_time(fn, warmup: int = 1, repetitions: int = 3) -> float:
    """
    Mesure le temps d'exécution minimal sur un nombre de répétitions après phase de chauffe.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def benchmark_simulation_naive_vs_mkl(
    cohort_sizes: List[int] = [50, 100, 200, 500],
    days: float = 30.0,
    dt: float = 0.05,
    threads: int = 4,
) -> List[Dict[str, Any]]:
    """
    Compare directement l'intégration temporelle naïve (boucle Python pur)
    face à l'intégration matricielle BLAS 3 / Intel MKL.
    """
    results = []
    p = DEFAULT_PARAMS

    for n in cohort_sizes:
        S, eta, M, A = build_cohort_parameters(n_patients=n, params=p, seed=2026)
        Y0 = initialize_cohort_state(n_patients=n, S=S, params=p, seed=2026)

        # Préparation des données listes pour la méthode naïve
        y0_list = Y0.tolist()
        s_list = S[:, 0].tolist()
        eta_list = eta.tolist()

        # 1. Mesure méthode naïve lente
        t_naive = measure_execution_time(
            lambda: simulate_cohort_naive_rk4(
                y0_list=y0_list,
                s_list=s_list,
                eta_list=eta_list,
                d=p["d"],
                delta=p["delta"],
                p=p["p"],
                c=p["c"],
                beta_s=p["beta_s"],
                beta_r=p["beta_r"],
                mu=p["mu"],
                days=days,
                dt=dt,
            ),
            warmup=1,
            repetitions=2,
        )

        # 2. Mesure méthode optimisée BLAS 3 / MKL
        with threadpool_limits(limits=threads, user_api="blas"):
            t_mkl = measure_execution_time(
                lambda: simulate_cohort_rk4(
                    Y0=Y0,
                    S=S,
                    eta=eta,
                    M=M,
                    A=A,
                    beta_s=p["beta_s"],
                    beta_r=p["beta_r"],
                    days=days,
                    dt=dt,
                    save_interval=max(1, int(days / dt)),
                ),
                warmup=1,
                repetitions=3,
            )

        speedup = t_naive / t_mkl if t_mkl > 0 else 0.0
        throughput_mkl = n / t_mkl if t_mkl > 0 else 0.0

        results.append({
            "n_patients": n,
            "days_simulated": days,
            "time_naive_ms": t_naive * 1000.0,
            "time_mkl_ms": t_mkl * 1000.0,
            "speedup": speedup,
            "throughput_patients_per_sec": throughput_mkl,
        })

    return results


def benchmark_large_cohort_scaling(
    cohort_sizes: List[int] = [1000, 2000, 5000, 10000],
    days: float = 60.0,
    dt: float = 0.05,
    threads: int = 4,
) -> List[Dict[str, Any]]:
    """
    Évalue la montée en charge de la version MKL pour de très grandes cohortes de patients.
    Calcule le débit en GFLOPS effectifs.
    """
    results = []
    p = DEFAULT_PARAMS
    n_steps = int(days / dt)

    with threadpool_limits(limits=threads, user_api="blas"):
        for n in cohort_sizes:
            S, eta, M, A = build_cohort_parameters(n_patients=n, params=p, seed=2026)
            Y0 = initialize_cohort_state(n_patients=n, S=S, params=p, seed=2026)

            t_sim = measure_execution_time(
                lambda: simulate_cohort_rk4(
                    Y0=Y0,
                    S=S,
                    eta=eta,
                    M=M,
                    A=A,
                    beta_s=p["beta_s"],
                    beta_r=p["beta_r"],
                    days=days,
                    dt=dt,
                    save_interval=n_steps,
                ),
                warmup=1,
                repetitions=3,
            )

            # Nombre d'opérations FLOPs :
            # À chaque évaluation de RHS :
            # - Flux non-linéaires : ~6 * N FLOPs
            # - Y @ M.T : 2 * N * 5 * 5 = 50 * N FLOPs
            # - Phi @ A.T : 2 * N * 2 * 5 = 20 * N FLOPs
            # - Additions : 2 * 5 * N = 10 * N FLOPs
            # Total par RHS = ~86 * N FLOPs
            # RK4 = 4 évaluations RHS + mise à jour (10 * N) = ~354 * N FLOPs par pas
            total_flops = float(n_steps) * 354.0 * float(n)
            gflops = total_flops / (t_sim * 1e9) if t_sim > 0 else 0.0

            results.append({
                "n_patients": n,
                "time_ms": t_sim * 1000.0,
                "gflops": gflops,
                "throughput_patients_per_sec": n / t_sim if t_sim > 0 else 0.0,
            })

    return results


def benchmark_lapack_cholesky_vs_naive(
    cohort_sizes: List[int] = [1000, 5000, 10000, 20000],
    n_features: int = 15,
) -> List[Dict[str, Any]]:
    """
    Compare l'analyse clinique par LAPACK Cholesky (DPOTRF+DPOTRS+DSYRK)
    face à l'inversion explicite naïve (np.linalg.inv).
    """
    results = []
    rng = np.random.default_rng(2026)

    for n in cohort_sizes:
        # Données cliniques synthétiques
        X = rng.standard_normal((n, n_features))
        y = rng.standard_normal(n)

        # 1. LAPACK Cholesky
        t_lapack = measure_execution_time(
            lambda: solve_ridge_cholesky_lapack(X, y, alpha=1e-3),
            warmup=1,
            repetitions=3,
        )

        # 2. Inversion naïve explicite
        t_naive = measure_execution_time(
            lambda: solve_ridge_naive_inverse(X, y, alpha=1e-3),
            warmup=1,
            repetitions=3,
        )

        speedup = t_naive / t_lapack if t_lapack > 0 else 0.0

        results.append({
            "n_patients": n,
            "n_features": n_features,
            "time_lapack_ms": t_lapack * 1000.0,
            "time_naive_ms": t_naive * 1000.0,
            "speedup": speedup,
        })

    return results


def benchmark_thread_scalability(
    n_patients: int = 10000,
    thread_counts: List[int] = [1, 2, 4, 8],
    days: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    Mesure le passage à l'échelle (Speedup et Efficacité) en fonction du nombre de threads MKL.
    """
    p = DEFAULT_PARAMS
    S, eta, M, A = build_cohort_parameters(n_patients=n_patients, params=p, seed=2026)
    Y0 = initialize_cohort_state(n_patients=n_patients, S=S, params=p, seed=2026)

    # Référence 1 thread (temps séquentiel T1)
    with threadpool_limits(limits=1, user_api="blas"):
        t_base = measure_execution_time(
            lambda: simulate_cohort_rk4(
                Y0=Y0, S=S, eta=eta, M=M, A=A,
                beta_s=p["beta_s"], beta_r=p["beta_r"],
                days=days, dt=0.05, save_interval=int(days / 0.05),
            ),
            warmup=1,
            repetitions=3,
        )

    results = []
    for th in thread_counts:
        with threadpool_limits(limits=th, user_api="blas"):
            t_curr = measure_execution_time(
                lambda: simulate_cohort_rk4(
                    Y0=Y0, S=S, eta=eta, M=M, A=A,
                    beta_s=p["beta_s"], beta_r=p["beta_r"],
                    days=days, dt=0.05, save_interval=int(days / 0.05),
                ),
                warmup=1,
                repetitions=3,
            )

        speedup = t_base / t_curr
        efficiency = (speedup / th) * 100.0

        results.append({
            "threads": th,
            "time_ms": t_curr * 1000.0,
            "speedup": speedup,
            "efficiency_percent": efficiency,
        })

    return results
