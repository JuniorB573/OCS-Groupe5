# -*- coding: utf-8 -*-
"""
Simulation du modele intra-hote (T, I, V) avec LAPACK -- Groupe 5
=================================================================
Ce script reproduit les resultats numeriques des chapitres 7, 8 et 9 :
  1. conditionnement de la jacobienne et equilibrage par DGEEQU ;
  2. schema d'Euler implicite lineaire resolu par DGESV (+ Euler explicite) ;
  3. verification : stabilite, ordre de convergence, point fixe ;
  4. primo-infection, efficacite therapeutique, rebond apres arret du traitement ;
  5. scenarios d'intervention du modele de population (S, I1, I2).

Usage : python simulation_intra.py   (vih_lapack.py doit etre dans le meme dossier)
"""

import numpy as np
from scipy.linalg import lapack
from scipy.integrate import solve_ivp

import vih_lapack as v

P = dict(v.PA)                       # parametres du modele intra-hote
Y0 = np.array([1000.0, 0.0, 1e-3])   # sujet sain + trace de virus


# ---------------------------------------------------------------------------
# Schemas d'integration
# ---------------------------------------------------------------------------
def euler_implicite(y0, h, t_final, eta_fn=lambda t: P["eta"]):
    """Euler implicite lineaire : (I - h J) dy = h f(y), resolu par DGESV."""
    n = int(round(t_final / h))
    y = np.array(y0, float)
    traj = [y.copy()]
    for k in range(n):
        eta = eta_fn(k * h)
        f = v.rhs_h(0.0, y, P, eta)
        J = v.jac_h(y, P, eta)
        M = np.asfortranarray(np.eye(3) - h * J)
        _, _, dy, info = lapack.dgesv(M, h * f)
        assert info == 0, "DGESV a echoue"
        y = np.maximum(y + dy, 0.0)          # positivite numerique
        traj.append(y.copy())
    return np.array(traj)


def euler_explicite(y0, h, t_final):
    n = int(round(t_final / h))
    y = np.array(y0, float)
    with np.errstate(all="ignore"):
        for _ in range(n):
            y = y + h * v.rhs_h(0.0, y, P)
    return y


def reference(y0, t_final, eta_fn=lambda t: P["eta"]):
    """Solution de reference : solveur implicite d'ordre eleve (Radau)."""
    return solve_ivp(lambda t, y: v.rhs_h(t, y, P, eta_fn(t)), [0, t_final], y0,
                     method="Radau", rtol=1e-12, atol=1e-14, dense_output=True)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ee = v.ee_h(P)
    print("R0(0) =", v.R0_h(P), "| eta_c =", 1 - 1 / v.R0_h(P))
    print("Equilibre endemique E* =", ee)
    print("Valeurs propres en E* (DGEEV) :", v.eigvals_lapack(v.jac_h(ee, P)))

    # 1. Conditionnement et equilibrage (DGEEQU)
    print("\n--- Equilibrage de la jacobienne ---")
    for nom, y in [("E*", ee), ("E0", v.dfe_h(P)), ("pic", np.array([60.0, 300.0, 17000.0]))]:
        J = v.jac_h(y, P)
        r, c, *_ = lapack.dgeequ(np.asfortranarray(J))
        print(f"{nom:4s} kappa1(J) = {v.cond1(J):.3g} | apres equilibrage = "
              f"{v.cond1(np.diag(r) @ J @ np.diag(c)):.3g}")

    # 2-3. Verification du schema
    print("\n--- Verification : ordre de convergence a t = 10 j ---")
    ref10 = reference(Y0, 10.0).y[:, -1]
    erreurs = []
    for h in [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]:
        y = euler_implicite(Y0, h, 10.0)[-1]
        erreurs.append(np.linalg.norm((y - ref10) / np.maximum(np.abs(ref10), 1e-12)))
    for i, (h, e) in enumerate(zip([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125], erreurs)):
        ordre = "" if i == 0 else f"{np.log2(erreurs[i - 1] / e):.2f}"
        print(f"h = {h:7.4f}  erreur = {e:9.3g}  ordre = {ordre}")

    print("\n--- Stabilite : Euler explicite a h = 0.1 j sur 30 j ---")
    print("explicite :", euler_explicite(Y0, 0.1, 30.0), "(valeurs non finies = divergence)")
    print("implicite :", euler_implicite(Y0, 0.1, 30.0)[-1])

    print("\n--- Point fixe : implicite h = 0.1 j a t = 600 j ---")
    y600 = euler_implicite(Y0, 0.1, 600.0)[-1]
    print("erreur relative a E* :", np.abs(y600 - ee) / ee)

    # 4. Primo-infection
    print("\n--- Primo-infection (Radau) ---")
    sol = reference(Y0, 400.0)
    tt = np.linspace(0, 400, 40001)
    Y = sol.sol(tt)
    i = Y[2].argmax()
    print(f"pic de V = {Y[2, i]:.4g} virions/uL au jour {tt[i]:.1f} ; T minimal = {Y[0].min():.1f}")

    # Efficacite therapeutique, a partir de l'equilibre chronique
    print("\n--- Traitement a partir de E* ---")
    for eta in (0.50, 0.70, 0.95):
        s = solve_ivp(lambda t, y: v.rhs_h(t, y, P, eta), [0, 1500], ee, method="Radau",
                      rtol=1e-10, atol=1e-20, dense_output=True)
        t2 = np.linspace(0, 1500, 150001)
        V = s.sol(t2)[2]
        indet = t2[np.argmax(V < 0.05)] if (V < 0.05).any() else None
        print(f"eta = {eta:.2f} | R0 = {v.R0_h(P, eta):.2f} | indetectable (V<0.05) au jour {indet} "
              f"| V(1500) = {V[-1]:.4g}")

    # Rebond apres arret du traitement (eta = 0.95 jusqu'au jour 100)
    print("\n--- Arret du traitement au jour 100 ---")
    eta_fn = lambda t: 0.95 if t < 100 else 0.0
    s1 = solve_ivp(lambda t, y: v.rhs_h(t, y, P, eta_fn(t)), [0, 100], ee, method="Radau",
                   rtol=1e-10, atol=1e-14)
    y100 = s1.y[:, -1]
    s2 = solve_ivp(lambda t, y: v.rhs_h(t, y, P, 0.0), [100, 500], y100, method="Radau",
                   rtol=1e-10, atol=1e-14, dense_output=True)
    t3 = np.linspace(100, 500, 40001)
    V3 = s2.sol(t3)[2]
    j = V3.argmax()
    print(f"T(100) = {y100[0]:.0f} ; pic de rebond V = {V3[j]:.4g} au jour {t3[j]:.1f} "
          f"(= {V3[j] / ee[2]:.1f} fois V*)")

    # 5. Modele de population : scenarios (a partir de l'an 30)
    print("\n--- Modele de population : scenarios ---")
    P0 = dict(v.PB)
    for nom, bf, m2 in [("sans intervention", 1.0, 0.5), ("prevention (beta x 0.5)", 0.5, 0.5),
                        ("traitement (mu2 = 0.1)", 1.0, 0.1), ("prevention + traitement", 0.5, 0.1)]:
        def f(t, y, bf=bf, m2=m2):
            Q = dict(P0)
            if t >= 30:
                Q["beta"] = P0["beta"] * bf
                Q["mu2"] = m2
            return v.f_B(y, Q)
        s = solve_ivp(f, [0, 300], [1e-3, 0.0], rtol=1e-9, atol=1e-12, max_step=0.5)
        Q = dict(P0)
        Q["beta"] = P0["beta"] * bf
        Q["mu2"] = m2
        print(f"{nom:26s} R0 = {v.R0_B(Q, Q['beta']):.3f} | prevalence a 300 ans = "
              f"{100 * s.y[:, -1].sum():.2f} %")
