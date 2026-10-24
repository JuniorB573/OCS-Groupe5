"""
Module de référence naïf en Python pur (sans vectorisation).
Implémente la boucle séquentielle patient par patient avec listes et types scalaires.
Sert de point de comparaison direct pour mesurer les gains d'accélération (Speedup) de BLAS/LAPACK/MKL.
"""

from typing import List, Tuple


def evaluate_single_patient_rhs(
    y: List[float],
    s_k: float,
    eta_k: float,
    d: float,
    delta: float,
    p: float,
    c: float,
    beta_s: float,
    beta_r: float,
    mu: float,
) -> List[float]:
    """
    Calcule le second membre scalaire pour un seul patient k : [dT, dIs, dIr, dVs, dVr].
    """
    T, I_s, I_r, V_s, V_r = y

    # Flux d'infection (avec efficacité résiduelle sur mutant)
    eta_r = max(0.0, eta_k - 0.35)
    phi_s = (1.0 - eta_k) * beta_s * T * V_s
    phi_r = (1.0 - eta_r) * beta_r * T * V_r

    # Dérivées
    dT = s_k - d * T - phi_s - phi_r
    dIs = (1.0 - mu) * phi_s - delta * I_s
    dIr = mu * phi_s + phi_r - delta * I_r
    dVs = p * I_s - c * V_s
    dVr = p * I_r - c * V_r

    return [dT, dIs, dIr, dVs, dVr]


def simulate_cohort_naive_rk4(
    y0_list: List[List[float]],
    s_list: List[float],
    eta_list: List[float],
    d: float,
    delta: float,
    p: float,
    c: float,
    beta_s: float,
    beta_r: float,
    mu: float,
    days: float = 30.0,
    dt: float = 0.05,
) -> List[List[float]]:
    """
    Simulation de cohorte en boucle pure Python séquentielle (non vectorisée).
    Effectue l'intégration patient après patient, pas de temps après pas de temps.
    
    Returns:
        y_final: Liste des états finaux de chaque patient [T, Is, Ir, Vs, Vr].
    """
    n_patients = len(y0_list)
    n_steps = int(days / dt)

    # Copie en liste Python standard
    current_state = [[float(v) for v in patient] for patient in y0_list]

    for k in range(n_patients):
        s_k = s_list[k]
        eta_k = eta_list[k]
        y = current_state[k]

        for _ in range(n_steps):
            # K1
            k1 = evaluate_single_patient_rhs(y, s_k, eta_k, d, delta, p, c, beta_s, beta_r, mu)

            # K2
            y_k2 = [y[j] + 0.5 * dt * k1[j] for j in range(5)]
            k2 = evaluate_single_patient_rhs(y_k2, s_k, eta_k, d, delta, p, c, beta_s, beta_r, mu)

            # K3
            y_k3 = [y[j] + 0.5 * dt * k2[j] for j in range(5)]
            k3 = evaluate_single_patient_rhs(y_k3, s_k, eta_k, d, delta, p, c, beta_s, beta_r, mu)

            # K4
            y_k4 = [y[j] + dt * k3[j] for j in range(5)]
            k4 = evaluate_single_patient_rhs(y_k4, s_k, eta_k, d, delta, p, c, beta_s, beta_r, mu)

            # Mise à jour RK4
            for j in range(5):
                val = y[j] + (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                y[j] = val if val > 0.0 else 0.0

        current_state[k] = y

    return current_state
