"""
Module d'analyse clinique et pronostique exploitant LAPACK et BLAS.
Implémente :
- La régression clinique multivariée par factorisation de Cholesky (DPOTRF + DPOTRS).
- L'Analyse en Composantes Principales (ACP) clinique via DSYEVD (Divide & Conquer).
- La comparaison avec l'inversion naïve explicite.
"""

from typing import Tuple, Dict
import numpy as np
from scipy.linalg import lapack, blas


def build_clinical_design_matrix(
    Y0: np.ndarray,
    eta: np.ndarray,
    S: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit la matrice de design clinique X (N, P) et le vecteur cible y (N,).
    Variables explicatives :
    - Biais / Constante (1.0)
    - Nadir initial des CD4 sains (T0)
    - Charge virale initiale totale log10(V_s + V_r)
    - Niveau d'observance du patient (eta)
    - Production thymique individuelle (s_k)
    
    Cible y : Charge virale résiduelle log10(V_s + V_r + 1)
    """
    n_patients = Y0.shape[0]
    
    # 4 biomarqueurs initiaux + 1 terme de biais
    X = np.ones((n_patients, 5), dtype=np.float64, order="F")
    X[:, 1] = Y0[:, 0] / 1000.0                       # Nadir CD4 normalisé
    X[:, 2] = np.log10(Y0[:, 3] + Y0[:, 4] + 1.0)      # Log charge virale initiale
    X[:, 3] = eta                                      # Observance trithérapie
    X[:, 4] = S[:, 0] / 10.0                           # Production thymique relative
    
    return X


def solve_ridge_cholesky_lapack(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Résout le problème de régression clinique régularisée Ridge (X^T X + alpha*I) w = X^T y
    en utilisant les micro-noyaux BLAS 3 et LAPACK :
    1. DSYRK : calcul optimisé du produit de Gram symétrique G = X^T X (O(N * P^2)).
    2. DPOTRF : factorisation de Cholesky G = L @ L^T (O(1/3 * P^3)).
    3. DPOTRS : descente-remontée triangulaire pour résoudre sans inversion explicite.
    
    Returns:
        weights: Vecteur des coefficients de régression w (P,)
        stats: Métriques de performance et résidus
    """
    X_f = np.asfortranarray(X, dtype=np.float64)
    y_f = np.asfortranarray(y, dtype=np.float64)
    n, p = X_f.shape

    # 1. Produit de Gram G = X^T @ X via BLAS 3 dsyrk
    # G est symétrique (p, p)
    G = blas.dsyrk(alpha=1.0, a=X_f, trans=1, lower=1)
    # Remplissage de la partie supérieure pour symétrie complète
    G = G + np.tril(G, -1).T

    # Ajout de la régularisation de Tikhonov sur la diagonale
    np.fill_diagonal(G, G.diagonal() + alpha)

    # 2. Second membre b = X^T @ y via BLAS 2 dgemv
    b = blas.dgemv(alpha=1.0, a=X_f, x=y_f, trans=1)

    # 3. Factorisation de Cholesky via LAPACK dpotrf
    # G = L @ L^T
    L, info_potrf = lapack.dpotrf(G, lower=1)
    if info_potrf != 0:
        raise ValueError(f"DPOTRF a échoué (matrice non définie positive, info={info_potrf})")

    # 4. Résolution triangulaire via LAPACK dpotrs
    w, info_potrs = lapack.dpotrs(L, b, lower=1)
    if info_potrs != 0:
        raise ValueError(f"DPOTRS a échoué (info={info_potrs})")

    # Calcul du résidu clinique
    predictions = X_f @ w
    residuals = y_f - predictions
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1.0 - np.sum(residuals ** 2) / np.sum((y_f - np.mean(y_f)) ** 2))

    return w, {"rmse": rmse, "r2": r2, "info_potrf": info_potrf}


def solve_ridge_naive_inverse(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Résolution par inversion explicite naïve w = (X^T X + alpha*I)^(-1) X^T y.
    Moins stable numériquement et non exploitante de la structure symétrique de Cholesky.
    """
    G = X.T @ X + alpha * np.eye(X.shape[1])
    G_inv = np.linalg.inv(G)
    w = G_inv @ (X.T @ y)

    predictions = X @ w
    residuals = y - predictions
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1.0 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))

    return w, {"rmse": rmse, "r2": r2}


def compute_cohort_pca_lapack(
    trajectory_final: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyse en Composantes Principales (ACP) de l'état clinique final de la cohorte
    via la routine LAPACK DSYEVD (Divide & Conquer Eigensolver).
    
    Parameters:
        trajectory_final: Matrice d'états finaux des N patients (N, 5)
        n_components: Nombre d'axes principaux à conserver (défaut: 2)
        
    Returns:
        projected: Coordonnées des patients sur les axes principaux (N, n_components)
        eigenvalues: Valeurs propres (variances expliquées) triées
        components: Vecteurs propres (axes directeurs)
    """
    # Centrage et réduction
    mean = np.mean(trajectory_final, axis=0)
    std = np.std(trajectory_final, axis=0)
    std[std == 0.0] = 1.0
    Z = (trajectory_final - mean) / std

    # Matrice de covariance (5, 5) via BLAS 3 dsyrk
    n_patients = Z.shape[0]
    Cov = blas.dsyrk(alpha=1.0 / (n_patients - 1), a=Z, trans=1, lower=1)
    Cov = Cov + np.tril(Cov, -1).T

    # Diagonalisation spectrale via LAPACK dsyevd (Divide & Conquer)
    w, v, info = lapack.dsyevd(Cov, compute_v=1, lower=1)
    if info != 0:
        raise ValueError(f"DSYEVD a échoué (info={info})")

    # Tri décroissant des valeurs propres
    idx = np.argsort(w)[::-1]
    sorted_w = w[idx]
    sorted_v = v[:, idx]

    # Projection des patients sur les composantes principales via BLAS 3 DGEMM
    projected = blas.dgemm(alpha=1.0, a=Z, b=sorted_v[:, :n_components])

    return projected, sorted_w, sorted_v[:, :n_components]
