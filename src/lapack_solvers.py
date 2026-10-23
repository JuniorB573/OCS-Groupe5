"""
Module des solveurs et routines LAPACK (scipy.linalg.lapack)
Optimisés pour micro-noyaux Intel MKL et OpenBLAS.
"""

from typing import Tuple
import numpy as np
from scipy.linalg import lapack


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Résout Ax = b via factorisation LU avec pivotage partiel (routine DGESV).
    
    Parameters:
        A: Matrice carrée (n, n), ordre Fortran
        b: Vecteur second membre (n,)
        
    Returns:
        x: Solution du système
        pivots: Vecteur des indices de permutation
        info: Code de statut (0 si succès)
    """
    A_f = np.asfortranarray(A, dtype=np.float64)
    b_f = np.asfortranarray(b, dtype=np.float64)
    lu, piv, x, info = lapack.dgesv(A_f, b_f)
    if info != 0:
        raise ValueError(f"DGESV a échoué avec info = {info} (matrice singulière)")
    return x, piv, info


def balance_matrix(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Équilibre une matrice A par mise à l'échelle diagonale des lignes et colonnes (routine DGEEQU).
    Ramène le conditionnement kappa(A) à des valeurs stables.
    
    Returns:
        A_balanced: Matrice équilibrée R @ A @ C
        R: Facteurs d'échelle de lignes
        C: Facteurs d'échelle de colonnes
        cond_ratio: Ratio de conditionnement
    """
    A_f = np.asfortranarray(A, dtype=np.float64)
    r, c, rowcnd, colcnd, amax, info = lapack.dgeequ(A_f)
    if info != 0:
        raise ValueError(f"DGEEQU a échoué avec info = {info}")
    
    # Matrice équilibrée : A_eq = diag(R) @ A @ diag(C)
    A_balanced = (r[:, None] * A_f) * c[None, :]
    return A_balanced, r, c, min(rowcnd, colcnd)


def compute_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Calcule le spectre complet (valeurs propres complexes) d'une matrice générale via DGEEV.
    
    Returns:
        eigenvalues: Vecteur des valeurs propres complexes triées par partie réelle décroissante.
    """
    A_f = np.asfortranarray(A, dtype=np.float64)
    wr, wi, vl, vr, info = lapack.dgeev(A_f, compute_vl=0, compute_vr=0)
    if info != 0:
        raise ValueError(f"DGEEV a échoué avec info = {info}")
    w = wr + 1j * wi
    return w[np.argsort(-w.real)]


def compute_svd_norm(A: np.ndarray) -> float:
    """
    Calcule la plus grande valeur singulière (norme-2 spectrale) via routine DGESDD.
    """
    A_f = np.asfortranarray(A, dtype=np.float64)
    u, s, vt, info = lapack.dgesdd(A_f, compute_uv=0)
    if info != 0:
        raise ValueError(f"DGESDD a échoué avec info = {info}")
    return float(s[0])


def estimate_condition_number(A: np.ndarray) -> float:
    """
    Estime le conditionnement en norme-1 via factorisation DGETRF puis estimation DGECON (O(n^2)).
    """
    A_f = np.asfortranarray(A, dtype=np.float64)
    anorm = float(np.linalg.norm(A_f, 1))
    lu, piv, info = lapack.dgetrf(A_f)
    if info != 0:
        raise ValueError(f"DGETRF a échoué avec info = {info}")
    rcond, info = lapack.dgecon(lu, anorm, norm="1")
    if info != 0:
        raise ValueError(f"DGECON a échoué avec info = {info}")
    return 1.0 / rcond if rcond > 0 else np.inf
