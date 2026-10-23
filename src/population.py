"""
Module de modélisation épidémiologique de population (Échelle macroscopique).
Modèle compartimental S - I1 - I2 à incidence standardisée avec surmortalité liée au SIDA.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp

DEFAULT_POP_PARAMS: Dict[str, float] = {
    "Lam": 0.040,      # Taux de natalité brute (an^-1)
    "mu": 0.015,       # Mortalité naturelle (an^-1) -> espérance de vie naturelle ~67 ans
    "sigma": 0.10,     # Taux de progression clinique I1 -> I2 (an^-1)
    "mu2": 0.50,       # Surmortalité liée au stade SIDA (an^-1) -> survie ~2 ans
    "eps": 0.30,       # Contagiosité relative au stade SIDA (eps < 1)
    "beta": 0.2387,    # Taux de transmission calibré pour R0 = 1.80
}


def compute_pop_r0(params: Optional[Dict[str, float]] = None) -> float:
    """
    Calcule le nombre de reproduction de base R0_pop par analyse NGM en proportions :
    R0 = (beta / (sigma + Lam)) * [1 + (eps * sigma) / (mu2 + Lam)]
    """
    p = params or DEFAULT_POP_PARAMS
    denom1 = p["sigma"] + p["Lam"]
    denom2 = p["mu2"] + p["Lam"]
    return (p["beta"] / denom1) * (1.0 + (p["eps"] * p["sigma"] / denom2))


def rhs_population(t: float, y: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Équations différentielles en proportions (i1, i2) avec s = 1 - i1 - i2.
    """
    p = params or DEFAULT_POP_PARAMS
    i1, i2 = y
    s = max(0.0, 1.0 - i1 - i2)
    force = p["beta"] * (i1 + p["eps"] * i2)
    
    # di1/dt
    di1 = force * s - (p["sigma"] + p["Lam"]) * i1 + p["mu2"] * i1 * i2
    # di2/dt
    di2 = p["sigma"] * i1 - (p["mu2"] + p["Lam"]) * i2 + p["mu2"] * (i2 ** 2)
    return np.array([di1, di2], dtype=np.float64)


def jacobian_population(y: np.ndarray, params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Matrice Jacobienne 2x2 du modèle en proportions.
    """
    p = params or DEFAULT_POP_PARAMS
    i1, i2 = y
    s = max(0.0, 1.0 - i1 - i2)
    force = p["beta"] * (i1 + p["eps"] * i2)
    
    j11 = p["beta"] * (s - (i1 + p["eps"] * i2)) - (p["sigma"] + p["Lam"]) + p["mu2"] * i2
    j12 = -p["beta"] * (i1 + p["eps"] * i2) + p["beta"] * p["eps"] * s + p["mu2"] * i1
    j21 = p["sigma"]
    j22 = -(p["mu2"] + p["Lam"]) + 2.0 * p["mu2"] * i2
    return np.array([[j11, j12], [j21, j22]], dtype=np.float64)


def simulate_population(
    y0_prop: Tuple[float, float],
    t_span: Tuple[float, float],
    n_points: int = 500,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simule la dynamique épidémiologique et démographique sur t_span.
    
    Returns:
        t_eval: Vecteur temps (années)
        s_arr: Proportions de susceptibles
        i1_arr: Proportions d'infectés asymptomatiques
        i2_arr: Proportions de stade SIDA
    """
    p = params or DEFAULT_POP_PARAMS
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    sol = solve_ivp(
        fun=lambda t, y: rhs_population(t, y, p),
        t_span=t_span,
        y0=y0_prop,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )
    
    i1 = sol.y[0]
    i2 = sol.y[1]
    s = np.maximum(0.0, 1.0 - i1 - i2)
    return sol.t, s, i1, i2
