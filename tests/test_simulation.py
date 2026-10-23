"""
Suite de tests unitaires et d'intégration scientifique
Projet VIH/SIDA HPC - Groupe 5 (ENSGMM / UNSTIM)
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Ajout de la racine au sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.lapack_solvers import (
    solve_linear_system,
    balance_matrix,
    compute_eigenvalues,
    compute_svd_norm,
)
from src.intra_host import (
    DEFAULT_INTRA_PARAMS,
    compute_r0,
    compute_critical_efficacy,
    rhs_intra,
    jacobian_intra,
    disease_free_equilibrium,
    endemic_equilibrium,
    solve_equilibrium_newton,
    simulate_euler_implicit,
)
from src.population import (
    DEFAULT_POP_PARAMS,
    compute_pop_r0,
    rhs_population,
    simulate_population,
)


class TestLapackSolvers(unittest.TestCase):
    """Validation des liaisons Fortran LAPACK."""
    
    def test_dgesv_linear_system(self):
        # Système 3x3 connu : A x = b
        A = np.array([
            [3.0, 1.0, -1.0],
            [1.0, 4.0, 1.0],
            [2.0, 1.0, 2.0]
        ], dtype=np.float64)
        b = np.array([1.0, 7.0, 3.0], dtype=np.float64)
        
        x, piv, info = solve_linear_system(A, b)
        self.assertEqual(info, 0)
        self.assertEqual(len(piv), 3)
        # Vérification du résidu ||A x - b|| < 1e-14
        residual = np.linalg.norm(A @ x - b)
        self.assertLess(residual, 1e-14)
        
    def test_dgeequ_matrix_balancing(self):
        # Matrice mal conditionnée par écart d'échelles
        A = np.array([
            [1e4, 1e-2],
            [1e-1, 1e2]
        ], dtype=np.float64)
        A_bal, r, c, cond_ratio = balance_matrix(A)
        self.assertEqual(A_bal.shape, (2, 2))
        self.assertGreater(cond_ratio, 0.0)
        self.assertLessEqual(np.max(np.abs(A_bal)), 1.0 + 1e-6)

    def test_dgeev_eigenvalues(self):
        # Matrice de rotation 2D avec valeurs propres imaginaires pures +/- i
        A = np.array([
            [0.0, -1.0],
            [1.0, 0.0]
        ], dtype=np.float64)
        eigvals = compute_eigenvalues(A)
        self.assertEqual(len(eigvals), 2)
        np.testing.assert_allclose(np.abs(eigvals.imag), [1.0, 1.0])
        np.testing.assert_allclose(eigvals.real, [0.0, 0.0], atol=1e-15)

    def test_dgesdd_svd_norm(self):
        A = np.diag([5.0, 2.0, -1.0])
        sigma_max = compute_svd_norm(A)
        self.assertTrue(np.isclose(sigma_max, 5.0))


class TestIntraHostModel(unittest.TestCase):
    """Validation du modèle compartimental intra-hôte (T, I, V)."""

    def test_r0_baseline(self):
        # R0 = beta * p * (s/d) / (delta * c) = 4.6e-5 * 1000 * 1000 / (0.5 * 23) = 4.0
        r0 = compute_r0(DEFAULT_INTRA_PARAMS, eta=0.0)
        self.assertTrue(np.isclose(r0, 4.0, rtol=1e-3))

    def test_critical_efficacy(self):
        # eta_c = 1 - 1/4 = 0.75 (75%)
        eta_c = compute_critical_efficacy(DEFAULT_INTRA_PARAMS)
        self.assertTrue(np.isclose(eta_c, 0.75, rtol=1e-3))

    def test_r0_with_treatment(self):
        # Avec eta = 0.75, R0(eta) = 1.0
        r0_crit = compute_r0(DEFAULT_INTRA_PARAMS, eta=0.75)
        self.assertTrue(np.isclose(r0_crit, 1.0, rtol=1e-3))
        # Avec eta = 0.90, R0(eta) = 0.40 < 1 (extinction virale)
        r0_sub = compute_r0(DEFAULT_INTRA_PARAMS, eta=0.90)
        self.assertTrue(np.isclose(r0_sub, 0.40, rtol=1e-3))

    def test_disease_free_equilibrium_is_root(self):
        e0 = disease_free_equilibrium(DEFAULT_INTRA_PARAMS)
        np.testing.assert_allclose(e0, [1000.0, 0.0, 0.0])
        f_e0 = rhs_intra(0.0, e0, DEFAULT_INTRA_PARAMS)
        np.testing.assert_allclose(f_e0, [0.0, 0.0, 0.0], atol=1e-14)

    def test_endemic_equilibrium_is_root(self):
        estar = endemic_equilibrium(DEFAULT_INTRA_PARAMS, eta=0.0)
        self.assertIsNotNone(estar)
        self.assertTrue(np.isclose(estar[0], 250.0, rtol=1e-3))
        # f(E*) doit être nul
        f_estar = rhs_intra(0.0, estar, DEFAULT_INTRA_PARAMS, eta=0.0)
        np.testing.assert_allclose(f_estar, [0.0, 0.0, 0.0], atol=1e-12)

    def test_newton_raphson_dgesv_solver(self):
        # Départ éloigné de E*
        y0_guess = np.array([400.0, 5.0, 200.0])
        y_star, iters, success = solve_equilibrium_newton(y0_guess, eta=0.0)
        self.assertTrue(success)
        self.assertLess(iters, 15)
        expected_star = endemic_equilibrium(DEFAULT_INTRA_PARAMS, eta=0.0)
        np.testing.assert_allclose(y_star, expected_star, rtol=1e-6)

    def test_euler_implicit_stability(self):
        y0 = np.array([1000.0, 1.0, 10.0])
        t_span = (0.0, 10.0)
        # Pas relativement large h = 0.1
        t_eval, y_hist = simulate_euler_implicit(y0, t_span, h=0.1, eta=0.0)
        self.assertEqual(len(t_eval), len(y_hist))
        self.assertFalse(np.any(np.isnan(y_hist)))
        self.assertTrue(np.all(y_hist >= 0.0))


class TestPopulationModel(unittest.TestCase):
    """Validation du modèle macroscopique S - I1 - I2."""

    def test_pop_r0_calibration(self):
        r0 = compute_pop_r0(DEFAULT_POP_PARAMS)
        self.assertTrue(np.isclose(r0, 1.80, rtol=1e-2))

    def test_population_conservation_and_positivity(self):
        y0 = (0.001, 0.0)
        t_span = (0.0, 30.0)
        t_eval, s, i1, i2 = simulate_population(y0, t_span, n_points=100)
        
        # Positivité
        self.assertTrue(np.all(s >= 0.0))
        self.assertTrue(np.all(i1 >= 0.0))
        self.assertTrue(np.all(i2 >= 0.0))
        
        # Somme des proportions = 1.0 à tout instant
        total = s + i1 + i2
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
