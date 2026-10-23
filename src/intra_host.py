"""
Module de modélisation intra-hôte du VIH (Échelle du patient).
Modèle compartimental non-linéaire de Nowak, May & Perelson (T, I, V).
"""

from typing import Dict, Tuple, Optional
import numpy as np
from .lapack_solvers import solve_linear_system

DEFAULT_INTRA_PARAMS: Dict[str, float] = {
    "s": 10.0,           # Production de CD4+ saines (cellules/uL/jour)
    "d": 0.01,           # Mortalité naturelle des CD4+ (jour^-1) -> T0 = s/d = 1000
    "delta": 0.50,       # Mortalité des cellules infectées (jour^-1)
    "p": 1000.0,         # Production virale par cellule (virions/cellule/jour)
    "c": 23.0,           # Clairance virale plasmatique (jour^-1)
    "beta": 4.6e-5,      # Infectivité virale (uL/virion/jour)
    "eta": 0.0,          # Efficacité du traitement antirétroviral (0 <= eta <= 1)
}


def compute_r0(params: Optional[Dict[str, float]] = None, eta: Optional[float] = None) -> float:
    """
    Calcule le nombre de reproduction de base R0(eta) par la méthode NGM.
    R0 = (1 - eta) * beta * p * (s / d) / (delta * c)
    """
    p = params or DEFAULT_INTRA_PARAMS
    eff = p["eta"] if eta is None else eta
    t0 = p["s"] / p["d"]
    return (1.0 - eff) * p["beta"] * p["p"] * t0 / (p["delta"] * p["c"])


def compute_critical_efficacy(params: Optional[Dict[str, float]] = None) -> float:
    """
    Calcule l'efficacité critique de traitement pour éradiquer l'infection :
    eta_c = 1 - 1 / R0(0).
    """
    r0_0 = compute_r0(params, eta=0.0)
    return 1.0 - (1.0 / r0_0)


def rhs_intra(t: float, y: np.ndarray, params: Optional[Dict[str, float]] = None, eta: Optional[float] = None) -> np.ndarray:
    """
    Second membre du système différentiel intra-hôte (dT/dt, dI/dt, dV/dt).
    """
    p = params or DEFAULT_INTRA_PARAMS
    eff = p["eta"] if eta is None else eta
    T, I, V = y
    inf = (1.0 - eff) * p["beta"] * T * V
    dT = p["s"] - p["d"] * T - inf
    dI = inf - p["delta"] * I
    dV = p["p"] * I - p["c"] * V
    return np.array([dT, dI, dV], dtype=np.float64)


def jacobian_intra(y: np.ndarray, params: Optional[Dict[str, float]] = None, eta: Optional[float] = None) -> np.ndarray:
    """
    Matrice Jacobienne 3x3 du système intra-hôte.
    """
    p = params or DEFAULT_INTRA_PARAMS
    eff = p["eta"] if eta is None else eta
    T, I, V = y
    b = (1.0 - eff) * p["beta"]
    return np.array([
        [-p["d"] - b * V, 0.0, -b * T],
        [b * V, -p["delta"], b * T],
        [0.0, p["p"], -p["c"]],
    ], dtype=np.float64, order="F")


def disease_free_equilibrium(params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Équilibre sans infection (DFE) : E0 = (s/d, 0, 0).
    """
    p = params or DEFAULT_INTRA_PARAMS
    return np.array([p["s"] / p["d"], 0.0, 0.0], dtype=np.float64)


def endemic_equilibrium(params: Optional[Dict[str, float]] = None, eta: Optional[float] = None) -> Optional[np.ndarray]:
    """
    Équilibre endémique (EE) : E* = (T*, I*, V*) existe si R0 > 1.
    """
    p = params or DEFAULT_INTRA_PARAMS
    r0 = compute_r0(p, eta)
    if r0 <= 1.0:
        return None
    t0 = p["s"] / p["d"]
    Ts = t0 / r0
    Is = (p["s"] - p["d"] * Ts) / p["delta"]
    Vs = (p["p"] / p["c"]) * Is
    return np.array([Ts, Is, Vs], dtype=np.float64)


def solve_equilibrium_newton(
    y0: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    eta: Optional[float] = None,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> Tuple[np.ndarray, int, bool]:
    """
    Résout f(y*) = 0 par la méthode de Newton-Raphson avec le solveur LAPACK DGESV.
    """
    y = np.array(y0, dtype=np.float64)
    for it in range(1, max_iter + 1):
        f = rhs_intra(0.0, y, params, eta)
        if np.linalg.norm(f) < tol:
            return y, it, True
        J = jacobian_intra(y, params, eta)
        # Résout J * dy = -f via DGESV
        dy, _, info = solve_linear_system(J, -f)
        y += dy
        if np.linalg.norm(dy) < tol:
            return y, it, True
    return y, max_iter, False


def simulate_euler_implicit(
    y0: np.ndarray,
    t_span: Tuple[float, float],
    h: float,
    params: Optional[Dict[str, float]] = None,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intégration temporelle inconditionnellement A-stable par schéma d'Euler implicite
    avec linéarisation de Newton résolue par LAPACK DGESV à chaque pas de temps.
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / h))
    t_eval = np.linspace(t0, tf, n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)
    y_history[0] = y0
    
    y = np.array(y0, dtype=np.float64)
    I_mat = np.eye(len(y0), dtype=np.float64, order="F")
    
    for k in range(n_steps):
        # [I - h * J(y)] * Delta_y = h * f(y)
        f_val = rhs_intra(t_eval[k], y, params, eta)
        J_val = jacobian_intra(y, params, eta)
        M = I_mat - h * J_val
        delta_y, _, _ = solve_linear_system(M, h * f_val)
        y = y + delta_y
        y = np.maximum(y, 0.0)  # Préservation stricte de la positivité
        y_history[k + 1] = y
        
    return t_eval, y_history


def simulate_euler_explicit(
    y0: np.ndarray,
    t_span: Tuple[float, float],
    h: float,
    params: Optional[Dict[str, float]] = None,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intégration par Euler explicite (démontre la divergence/explosion numérique pour h > 0.08j).
    """
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / h))
    t_eval = np.linspace(t0, tf, n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)
    y_history[0] = y0
    
    y = np.array(y0, dtype=np.float64)
    for k in range(n_steps):
        f_val = rhs_intra(t_eval[k], y, params, eta)
        y = y + h * f_val
        y_history[k + 1] = y
        if np.any(np.isnan(y)) or np.any(np.abs(y) > 1e10):
            # Explosion numérique constatée
            y_history[k + 1:] = np.nan
            break
            
    return t_eval, y_history
