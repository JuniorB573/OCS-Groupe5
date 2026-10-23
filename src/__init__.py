"""
Package vih-sida-hpc
Modélisation mathématique et simulation haute performance du VIH-SIDA
(BLAS, LAPACK et Intel oneMKL)

Auteurs : Groupe 5 -- ENSGMM / UNSTIM (Abomey, Bénin)
- ADOGOUN Déo Gracias Mahugnon
- BIAHOU Ahossou Junior Marie-Clémenceau
- SYLLA IMOROU Izou Dine
Encadrant : Dr. Carlos AGOSSOU
"""

__version__ = "1.0.0"

from .lapack_solvers import (
    solve_linear_system,
    balance_matrix,
    compute_eigenvalues,
    compute_svd_norm,
)
from .intra_host import (
    DEFAULT_INTRA_PARAMS,
    compute_r0,
    rhs_intra,
    jacobian_intra,
    disease_free_equilibrium,
    endemic_equilibrium,
    solve_equilibrium_newton,
    simulate_euler_implicit,
    simulate_euler_explicit,
)
from .population import (
    DEFAULT_POP_PARAMS,
    compute_pop_r0,
    rhs_population,
    jacobian_population,
    simulate_population,
)
from .benchmark import (
    benchmark_dgemm,
    benchmark_dgesv,
    benchmark_thread_scaling,
)
