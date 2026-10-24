"""
Tests unitaires et de validation d'équivalence numérique :
1. Équivalence stricte entre l'implémentation naïve (Python pur) et matricielle (BLAS 3 / MKL).
2. Préservation des invariants biologiques (positivité des populations cellulaires et virales).
3. Cohérence et stabilité numérique des solveurs LAPACK (Cholesky vs Inversion).
"""

import pytest
import numpy as np

from src.cohort_model import (
    DEFAULT_PARAMS,
    build_cohort_parameters,
    initialize_cohort_state,
    evaluate_cohort_rhs,
    simulate_cohort_rk4,
)
from src.naive_model import (
    evaluate_single_patient_rhs,
    simulate_cohort_naive_rk4,
)
from src.lapack_analysis import (
    build_clinical_design_matrix,
    solve_ridge_cholesky_lapack,
    solve_ridge_naive_inverse,
    compute_cohort_pca_lapack,
)


def test_rhs_numerical_equivalence():
    """
    Vérifie que le second membre matriciel BLAS 3 dY/dt
    calcule exactement la même valeur que l'évaluation scalaire naïve.
    """
    n_patients = 10
    p = DEFAULT_PARAMS
    S, eta, M, A = build_cohort_parameters(n_patients=n_patients, params=p, seed=42)
    Y = initialize_cohort_state(n_patients=n_patients, S=S, params=p, seed=42)

    # 1. Évaluation matricielle vectorisée (BLAS 3)
    dY_blas = evaluate_cohort_rhs(
        Y, S, eta, M, A,
        beta_s=p["beta_s"], beta_r=p["beta_r"]
    )

    # 2. Évaluation scalaire naïve boucle par boucle
    dY_naive = np.zeros_like(dY_blas)
    for k in range(n_patients):
        res = evaluate_single_patient_rhs(
            y=Y[k].tolist(),
            s_k=float(S[k, 0]),
            eta_k=float(eta[k]),
            d=p["d"],
            delta=p["delta"],
            p=p["p"],
            c=p["c"],
            beta_s=p["beta_s"],
            beta_r=p["beta_r"],
            mu=p["mu"],
        )
        dY_naive[k] = res

    # Comparaison numérique stricte
    np.testing.assert_allclose(
        dY_blas, dY_naive,
        rtol=1e-12, atol=1e-12,
        err_msg="Discordance numérique entre le second membre BLAS 3 et la version naïve !"
    )


def test_trajectory_numerical_equivalence():
    """
    Vérifie que la trajectoire temporelle complète RK4 matricielle
    est identique au bit près à la simulation séquentielle naïve.
    """
    n_patients = 8
    days = 15.0
    dt = 0.05
    p = DEFAULT_PARAMS

    S, eta, M, A = build_cohort_parameters(n_patients=n_patients, params=p, seed=123)
    Y0 = initialize_cohort_state(n_patients=n_patients, S=S, params=p, seed=123)

    # 1. Simulation matricielle BLAS 3
    times, traj_mkl = simulate_cohort_rk4(
        Y0, S, eta, M, A,
        beta_s=p["beta_s"], beta_r=p["beta_r"],
        days=days, dt=dt, save_interval=int(days / dt)
    )
    final_mkl = traj_mkl[-1]

    # 2. Simulation naïve
    final_naive_list = simulate_cohort_naive_rk4(
        y0_list=Y0.tolist(),
        s_list=S[:, 0].tolist(),
        eta_list=eta.tolist(),
        d=p["d"],
        delta=p["delta"],
        p=p["p"],
        c=p["c"],
        beta_s=p["beta_s"],
        beta_r=p["beta_r"],
        mu=p["mu"],
        days=days,
        dt=dt,
    )
    final_naive = np.array(final_naive_list, dtype=np.float64)

    # Erreur relative maximale
    max_rel_err = np.max(np.abs(final_mkl - final_naive) / (np.abs(final_mkl) + 1e-10))
    assert max_rel_err < 1e-10, f"Erreur relative trop élevée : {max_rel_err}"


def test_biological_invariants():
    """
    Vérifie la non-négativité des compartiments cellulaires et viraux.
    """
    n_patients = 100
    p = DEFAULT_PARAMS
    S, eta, M, A = build_cohort_parameters(n_patients=n_patients, params=p, seed=777)
    Y0 = initialize_cohort_state(n_patients=n_patients, S=S, params=p, seed=777)

    times, trajectory = simulate_cohort_rk4(
        Y0, S, eta, M, A,
        beta_s=p["beta_s"], beta_r=p["beta_r"],
        days=60.0, dt=0.05, save_interval=20
    )

    # Tous les états doivent rester positifs ou nuls
    assert np.all(trajectory >= 0.0), "Violation biologique : populations négatives détectées !"


def test_lapack_cholesky_equivalence():
    """
    Vérifie que la résolution de régression Ridge via LAPACK Cholesky (DPOTRF+DPOTRS)
    fournit les mêmes coefficients que l'inversion explicite standard.
    """
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((500, 8))
    y = rng.standard_normal(500)

    w_lapack, stats_lapack = solve_ridge_cholesky_lapack(X, y, alpha=1e-2)
    w_naive, stats_naive = solve_ridge_naive_inverse(X, y, alpha=1e-2)

    np.testing.assert_allclose(
        w_lapack, w_naive,
        rtol=1e-10, atol=1e-10,
        err_msg="Écart de coefficients entre Cholesky LAPACK et l'inversion explicite !"
    )
    assert abs(stats_lapack["rmse"] - stats_naive["rmse"]) < 1e-10


def test_lapack_pca():
    """
    Vérifie le calcul spectral ACP via LAPACK DSYEVD (valeurs propres décroissantes et orthogonales).
    """
    rng = np.random.default_rng(2026)
    data = rng.standard_normal((200, 5))

    projected, eigenvalues, components = compute_cohort_pca_lapack(data, n_components=2)

    # 1. Dimensions attendues
    assert projected.shape == (200, 2)
    assert eigenvalues.shape == (5,)
    assert components.shape == (5, 2)

    # 2. Valeurs propres décroissantes
    assert np.all(np.diff(eigenvalues) <= 0), "Les valeurs propres ne sont pas triées de façon décroissante !"

    # 3. Orthogonalité des composantes principales
    identity_approx = components.T @ components
    np.testing.assert_allclose(identity_approx, np.eye(2), atol=1e-12)
