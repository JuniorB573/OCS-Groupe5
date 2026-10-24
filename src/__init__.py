"""
Package vih_sida_hpc : Modélisation HPC de la dynamique du VIH à l'échelle d'une cohorte clinique.
Optimisé pour BLAS 3, LAPACK et Intel MKL.
"""

from .cohort_model import (
    DEFAULT_PARAMS,
    build_cohort_parameters,
    initialize_cohort_state,
    evaluate_cohort_rhs,
    simulate_cohort_rk4,
)
from .naive_model import (
    evaluate_single_patient_rhs,
    simulate_cohort_naive_rk4,
)
from .lapack_analysis import (
    build_clinical_design_matrix,
    solve_ridge_cholesky_lapack,
    solve_ridge_naive_inverse,
    compute_cohort_pca_lapack,
)
from .benchmark import (
    get_hpc_system_info,
    benchmark_simulation_naive_vs_mkl,
    benchmark_large_cohort_scaling,
    benchmark_lapack_cholesky_vs_naive,
    benchmark_thread_scalability,
)

__all__ = [
    "DEFAULT_PARAMS",
    "build_cohort_parameters",
    "initialize_cohort_state",
    "evaluate_cohort_rhs",
    "simulate_cohort_rk4",
    "evaluate_single_patient_rhs",
    "simulate_cohort_naive_rk4",
    "build_clinical_design_matrix",
    "solve_ridge_cholesky_lapack",
    "solve_ridge_naive_inverse",
    "compute_cohort_pca_lapack",
    "get_hpc_system_info",
    "benchmark_simulation_naive_vs_mkl",
    "benchmark_large_cohort_scaling",
    "benchmark_lapack_cholesky_vs_naive",
    "benchmark_thread_scalability",
]
