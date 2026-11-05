# -*- coding: utf-8 -*-
"""
Mini-projet OCS -- Groupe 5 -- Modelisation du VIH-SIDA avec LAPACK (Python)
===========================================================================
Partie A : modele intra-hote (T, I, V) avec traitement antiretroviral d'efficacite eta
Partie B : modele de population (S, I1, I2) avec mortalites naturelle et liee au SIDA

Routines LAPACK (via scipy.linalg.lapack) :
  dgetrf / dgetrs / dgesv : factorisation LU et resolution de systemes lineaires
  dgecon                  : estimation du conditionnement (norme 1)
  dgeev                   : valeurs propres et vecteurs propres (matrice generale)
  dgesdd                  : decomposition en valeurs singulieres (norme spectrale)
  dsyevd                  : valeurs propres d'une matrice symetrique (reactivite)
"""

import os, json, time
import numpy as np
from scipy.linalg import lapack, expm, hilbert
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Outils LAPACK
# ---------------------------------------------------------------------------
def eigvals_lapack(A):
    wr, wi, _, _, info = lapack.dgeev(np.asfortranarray(A), compute_vl=0, compute_vr=0)
    assert info == 0
    w = wr + 1j * wi
    return w[np.argsort(-w.real)]

def lu_solve(A, b):
    """Resolution par dgetrf + dgetrs ; retourne x, rcond (dgecon), pivots."""
    A = np.asfortranarray(A, dtype=float)
    lu, piv, info = lapack.dgetrf(A)
    assert info == 0
    x, info = lapack.dgetrs(lu, piv, b)
    rcond, _ = lapack.dgecon(lu, np.linalg.norm(A, 1), norm="1")
    return x, rcond, piv

def cond1(A):
    A = np.asfortranarray(A, dtype=float)
    lu, piv, info = lapack.dgetrf(A)
    rcond, _ = lapack.dgecon(lu, np.linalg.norm(A, 1), norm="1")
    return 1.0 / rcond

def sigma_max(Mx):
    u, s, vt, info = lapack.dgesdd(np.asfortranarray(Mx), compute_uv=0)
    return s[0]

def sym_eigmax(Mx):
    return lapack.dsyevd(np.asfortranarray(Mx), compute_v=0)[0].max()

# ===========================================================================
# PARTIE A : MODELE INTRA-HOTE
# ===========================================================================
PA = dict(s=10.0,      # production de CD4+ (cellules / L / jour)
          d=0.01,      # mortalite naturelle des CD4+ (jour^-1) => T0 = s/d = 1000
          delta=0.5,   # mortalite des cellules infectees (jour^-1)
          p=1000.0,    # production virale (virions / cellule infectee / jour)
          c=23.0,      # clairance virale (jour^-1)
          beta=4.6e-5, # infectivite (L / virion / jour)
          eta=0.0)     # efficacite du traitement antiretroviral

def R0_h(P, eta=None):
    e = P["eta"] if eta is None else eta
    return (1 - e) * P["beta"] * P["p"] * (P["s"] / P["d"]) / (P["delta"] * P["c"])

def rhs_h(t, y, P, eta=None):
    T, I, V = y
    e = P["eta"] if eta is None else eta
    inf = (1 - e) * P["beta"] * T * V
    return np.array([P["s"] - P["d"] * T - inf, inf - P["delta"] * I, P["p"] * I - P["c"] * V])

def jac_h(y, P, eta=None):
    T, I, V = y
    e = P["eta"] if eta is None else eta
    b = (1 - e) * P["beta"]
    return np.array([[-P["d"] - b * V, 0.0, -b * T],
                     [b * V, -P["delta"], b * T],
                     [0.0, P["p"], -P["c"]]])

def dfe_h(P):
    return np.array([P["s"] / P["d"], 0.0, 0.0])

def ee_h(P, eta=None):
    e = P["eta"] if eta is None else eta
    R0 = R0_h(P, e)
    T0 = P["s"] / P["d"]
    Ts = T0 / R0
    Is = (P["s"] - P["d"] * Ts) / P["delta"]
    Vs = P["p"] * Is / P["c"]
    return np.array([Ts, Is, Vs])

def newton_h(y0, P, eta=None, tol=1e-12, maxit=50):
    """Newton : J(y) dy = -f(y), resolu par dgesv."""
    y = np.array(y0, float)
    scale = np.array([P["s"] / P["d"], 1.0, 1.0])
    for k in range(maxit):
        f = rhs_h(0, y, P, eta)
        lu, piv, dy, info = lapack.dgesv(np.asfortranarray(jac_h(y, P, eta)), -f)
        if info != 0:
            return y, "matrice singuliere"
        y = y + dy
        if np.linalg.norm(dy / scale) < tol:
            return y, "converge"
    return y, "maxit atteint"

# ===========================================================================
# PARTIE B : MODELE DE POPULATION (S, I1, I2)
# ===========================================================================
PB = dict(Lam=0.04, mu=0.015, sigma=0.10, mu2=0.50, eps=0.30)

def R0_B(P, beta):
    return beta / (P["sigma"] + P["Lam"]) * (1 + P["eps"] * P["sigma"] / (P["mu2"] + P["Lam"]))

R0B = 1.8
betaB = R0B / R0_B(PB, 1.0)
PB["beta"] = betaB

def f_B(y, P):
    i1, i2 = y
    s = 1 - i1 - i2
    F = i1 + P["eps"] * i2
    return np.array([P["beta"] * s * F - (P["sigma"] + P["Lam"]) * i1 + P["mu2"] * i1 * i2,
                     P["sigma"] * i1 - (P["mu2"] + P["Lam"]) * i2 + P["mu2"] * i2 ** 2])

def J_B(y, P):
    i1, i2 = y
    s = 1 - i1 - i2
    F = i1 + P["eps"] * i2
    return np.array([[-P["beta"] * F + P["beta"] * s - (P["sigma"] + P["Lam"]) + P["mu2"] * i2,
                      -P["beta"] * F + P["beta"] * P["eps"] * s + P["mu2"] * i1],
                     [P["sigma"], -(P["mu2"] + P["Lam"]) + 2 * P["mu2"] * i2]])

if __name__ == '__main__':
    print("Modele VIH-SIDA avec LAPACK initialise avec succes.")
