"""
Module de modélisation haute performance d'une cohorte de patients VIH sous trithérapie.
Formulation matricielle globale : dY/dt = S + Y @ M.T + Phi(Y) @ A.T
Exploite BLAS 3 (DGEMM), MKL VML et l'intégration Runge-Kutta 4 matricielle.
"""

from typing import Dict, Tuple, Optional
import numpy as np


# Paramètres biologiques par défaut (échelle journalière)
DEFAULT_PARAMS: Dict[str, float] = {
    "s_mean": 10.0,       # Taux moyen de néo-génération thymique des CD4+ (cellules/uL/jour)
    "s_std": 1.5,         # Écart-type inter-individuel de la production thymique
    "d": 0.01,            # Mortalité naturelle des CD4+ sains (jour^-1) -> T0 = s/d ~ 1000
    "delta": 0.50,        # Mortalité induite des cellules infectées (jour^-1)
    "p": 1000.0,          # Production virale par cellule infectée (virions/cellule/jour)
    "c": 23.0,            # Clairance plasmatique des virions libres (jour^-1)
    "beta_s": 4.6e-5,     # Infectiosité de la souche sauvage sensible (uL/virion/jour)
    "beta_r": 3.2e-5,     # Infectiosité de la souche mutante résistante (coût de fitness: beta_r < beta_s)
    "mu": 1.2e-4,         # Taux de mutation de résistance lors de la transcription inverse
    "eta_mean": 0.85,     # Efficacité moyenne de la trithérapie (observance)
    "eta_std": 0.12,      # Hétérogénéité d'observance dans la cohorte
}


def build_cohort_parameters(
    n_patients: int,
    params: Optional[Dict[str, float]] = None,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère les paramètres hétérogènes d'une cohorte de N patients.
    
    Returns:
        S: Matrice de source thymique (N, 5), ordre Fortran
        eta: Vecteur d'observance/efficacité du traitement (N,)
        M: Matrice linéaire de mortalité et clairance (5, 5), ordre Fortran
        A: Matrice stœchiométrique des flux d'infection et de mutation (5, 2), ordre Fortran
    """
    p = params or DEFAULT_PARAMS
    rng = np.random.default_rng(seed)

    # 1. Variabilité inter-individuelle
    s_k = rng.normal(p["s_mean"], p["s_std"], size=n_patients)
    s_k = np.clip(s_k, 5.0, 20.0)

    # Observance de la trithérapie entre 0 (arrêt total) et 0.99 (excellente observance)
    eta = rng.normal(p["eta_mean"], p["eta_std"], size=n_patients)
    eta = np.clip(eta, 0.0, 0.99)

    # 2. Matrice S de source (N, 5) en ordre Fortran (colonne 0 = production CD4)
    S = np.zeros((n_patients, 5), dtype=np.float64, order="F")
    S[:, 0] = s_k

    # 3. Matrice linéaire M (5, 5) : [T, I_s, I_r, V_s, V_r]
    # dY/dt = ... + Y @ M.T
    d = p["d"]
    delta = p["delta"]
    prod_p = p["p"]
    c = p["c"]

    M = np.array([
        [-d, 0.0, 0.0, 0.0, 0.0],
        [0.0, -delta, 0.0, 0.0, 0.0],
        [0.0, 0.0, -delta, 0.0, 0.0],
        [0.0, prod_p, 0.0, -c, 0.0],
        [0.0, 0.0, prod_p, 0.0, -c],
    ], dtype=np.float64, order="F")

    # 4. Matrice stœchiométrique A (5, 2)
    # Flux : Phi = [Phi_s, Phi_r] où Phi_s = (1-eta)*beta_s*T*V_s et Phi_r = beta_r*T*V_r
    mu = p["mu"]
    A = np.array([
        [-1.0, -1.0],      # dT/dt perd les cellules infectées par souche sauvage et résistante
        [1.0 - mu, 0.0],   # dI_s/dt reçoit les infections sauvages sans mutation
        [mu, 1.0],         # dI_r/dt reçoit les mutants et les infections résistantes
        [0.0, 0.0],        # dV_s/dt régie par M
        [0.0, 0.0],        # dV_r/dt régie par M
    ], dtype=np.float64, order="F")

    return S, eta, M, A


def initialize_cohort_state(
    n_patients: int,
    S: np.ndarray,
    params: Optional[Dict[str, float]] = None,
    seed: int = 2026,
) -> np.ndarray:
    """
    Initialise l'état clinique de la cohorte au début de la mise sous trithérapie (t=0).
    Patients en phase chronique d'infection :
    - T0 ~ 400 - 800 cellules/uL
    - I_s ~ 1 - 10 cellules/uL, I_r ~ 0.01 - 0.1 cellules/uL
    - V_s ~ 10^4 - 10^5 virions/mL, V_r ~ 10 - 100 virions/mL
    
    Returns:
        Y0: Matrice d'état initiale (N, 5), ordre Fortran
    """
    p = params or DEFAULT_PARAMS
    rng = np.random.default_rng(seed)

    Y0 = np.zeros((n_patients, 5), dtype=np.float64, order="F")

    # CD4 cibles sains
    T_eq = S[:, 0] / p["d"]
    Y0[:, 0] = T_eq * rng.uniform(0.4, 0.7, size=n_patients)

    # Cellules infectées sauvages et mutantes
    Y0[:, 1] = rng.uniform(2.0, 8.0, size=n_patients)        # I_s
    Y0[:, 2] = rng.uniform(0.01, 0.05, size=n_patients)      # I_r

    # Charges virales
    Y0[:, 3] = (p["p"] / p["c"]) * Y0[:, 1] * rng.uniform(0.9, 1.1, size=n_patients)  # V_s
    Y0[:, 4] = (p["p"] / p["c"]) * Y0[:, 2] * rng.uniform(0.9, 1.1, size=n_patients)  # V_r

    return Y0


def evaluate_cohort_rhs(
    Y: np.ndarray,
    S: np.ndarray,
    eta: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    beta_s: float,
    beta_r: float,
) -> np.ndarray:
    """
    Calcule le second membre matriciel de la cohorte dY/dt.
    Opérations 100% BLAS 3 et vectorisées :
    dY/dt = S + Y @ M.T + Phi(Y) @ A.T
    
    Parameters:
        Y: Matrice d'état (N, 5), ordre Fortran
        S: Matrice de source (N, 5)
        eta: Efficacité du traitement par patient (N,)
        M: Matrice linéaire (5, 5)
        A: Matrice stœchiométrique (5, 2)
        beta_s, beta_r: Taux d'infectiosité
        
    Returns:
        dY: Dérivée temporelle (N, 5), ordre Fortran
    """
    # 1. Calcul des flux d'infection non-linéaires Phi (N, 2)
    # T = Y[:, 0], V_s = Y[:, 3], V_r = Y[:, 4]
    # Efficacité sur souche sauvage = eta
    # Efficacité résiduelle de la multithérapie sur souche mutante = max(0, eta - 0.35)
    eta_r = np.maximum(0.0, eta - 0.35)
    phi_s = (1.0 - eta) * beta_s * Y[:, 0] * Y[:, 3]
    phi_r = (1.0 - eta_r) * beta_r * Y[:, 0] * Y[:, 4]

    Phi = np.empty((Y.shape[0], 2), dtype=np.float64, order="F")
    Phi[:, 0] = phi_s
    Phi[:, 1] = phi_r

    # 2. Terme linéaire : Y @ M.T (BLAS 3 DGEMM)
    # 3. Terme non-linéaire : Phi @ A.T (BLAS 3 DGEMM)
    # dY = S + Y @ M.T + Phi @ A.T
    dY = S + (Y @ M.T) + (Phi @ A.T)
    return dY


def simulate_cohort_rk4(
    Y0: np.ndarray,
    S: np.ndarray,
    eta: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    beta_s: float,
    beta_r: float,
    days: float = 365.0,
    dt: float = 0.05,
    save_interval: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intégrateur Runge-Kutta d'ordre 4 matriciel pour la cohorte de N patients.
    
    Returns:
        times: Vecteur des temps enregistrés (K,)
        trajectory: Tenseur des états enregistrés (K, N, 5)
    """
    n_steps = int(np.ceil(days / dt))
    Y = Y0.copy(order="F")

    record_indices = list(range(0, n_steps + 1, save_interval))
    if record_indices[-1] != n_steps:
        record_indices.append(n_steps)

    times = np.array([idx * dt for idx in record_indices], dtype=np.float64)
    trajectory = np.zeros((len(record_indices), Y.shape[0], 5), dtype=np.float64)

    record_ptr = 0
    trajectory[record_ptr] = Y
    record_ptr += 1

    for step in range(1, n_steps + 1):
        # K1
        k1 = evaluate_cohort_rhs(Y, S, eta, M, A, beta_s, beta_r)

        # K2
        Y_k2 = Y + 0.5 * dt * k1
        k2 = evaluate_cohort_rhs(Y_k2, S, eta, M, A, beta_s, beta_r)

        # K3
        Y_k3 = Y + 0.5 * dt * k2
        k3 = evaluate_cohort_rhs(Y_k3, S, eta, M, A, beta_s, beta_r)

        # K4
        Y_k4 = Y + dt * k3
        k4 = evaluate_cohort_rhs(Y_k4, S, eta, M, A, beta_s, beta_r)

        # Mise à jour RK4 matricielle
        Y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Non-négativité biologique garantie
        np.maximum(Y, 0.0, out=Y)

        if step in record_indices[record_ptr:]:
            trajectory[record_ptr] = Y
            record_ptr += 1

    return times, trajectory
