# -*- coding: utf-8 -*-
"""
Script de simulation interactif du modele VIH-SIDA -- Groupe 5
==============================================================
Ce script permet de simuler facilement le modele intra-hote (T, I, V)
ou le modele en population (S, I1, I2) avec vos propres parametres.

Exemples d'utilisation en ligne de commande :
  python simuler.py --aide
  python simuler.py --scenar primo
  python simuler.py --scenar traitement --eta 0.95 --jours 180
  python simuler.py --scenar rebond --arret 100 --jours 250
  python simuler.py --modele population --scenario prevention
"""

import sys
import argparse
import numpy as np
from scipy.linalg import lapack
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

import vih_lapack as v

# ---------------------------------------------------------------------------
# Solveur Euler implicite via LAPACK DGESV
# ---------------------------------------------------------------------------
def resoudre_intra_implicite(t_span, y0, P, eta_fn=lambda t: 0.0, h=0.05):
    """Integration par Euler implicite lineaire resolu avec LAPACK DGESV."""
    t0, tf = t_span
    n = int(np.ceil((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    
    for k in range(n):
        tk = t[k]
        eta = eta_fn(tk)
        fk = v.rhs_h(tk, y[k], P, eta)
        Jk = v.jac_h(y[k], P, eta)
        # Systeme lineaire : (I - h * J) * dy = h * f
        M = np.asfortranarray(np.eye(3) - h * Jk)
        rhs = h * fk
        _, _, dy, info = lapack.dgesv(M, rhs)
        if info != 0:
            raise RuntimeError(f"DGESV a echoue a l'etape {k} (t={tk:.2f})")
        # Mise a jour avec projection sur le domaine positif
        y[k + 1] = np.maximum(y[k] + dy, 0.0)
        
    return t, y


# ---------------------------------------------------------------------------
# Scenarios de simulation
# ---------------------------------------------------------------------------
def simuler_primo(jours=60, h=0.05, save_fig="simulation_primo.png"):
    P = dict(v.PA)
    P["eta"] = 0.0
    y0 = [1000.0, 0.0, 1e-3]
    print(f"\n[1/3] Simulation de la PRIMO-INFECTION sur {jours} jours (sans traitement)...")
    t, y = resoudre_intra_implicite((0, jours), y0, P, h=h)
    
    T, I, V = y[:, 0], y[:, 1], y[:, 2]
    idx_pic = np.argmax(V)
    t_pic = t[idx_pic]
    v_pic = V[idx_pic]
    t_min_T = t[np.argmin(T)]
    min_T = np.min(T)
    
    print(f" -> Pic de charge virale : {v_pic:,.1f} virions/uL au jour {t_pic:.1f}")
    print(f" -> Niveau minimal de CD4+ : {min_T:.1f} cellules/uL au jour {t_min_T:.1f}")
    print(f" -> Equilibre asymptotique E* attendu : T*={P['s']/(P['d']*4):.0f}, V*={P['p']*15/P['c']:.1f}")
    
    # Graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, T, 'b-', lw=2, label="CD4+ sains T(t)")
    ax1.plot(t, I, 'orange', lw=2, label="CD4+ infectes I(t)")
    ax1.set_ylabel("Cellules / uL")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title(f"Simulation VIH Intra-Hote : Primo-infection (Euler Implicite DGESV)")
    
    ax2.semilogy(t, np.maximum(V, 1e-3), 'r-', lw=2, label="Charge virale V(t)")
    ax2.axvline(t_pic, color='gray', linestyle='--', alpha=0.7, label=f"Pic viral: j {t_pic:.1f}")
    ax2.set_xlabel("Temps (jours)")
    ax2.set_ylabel("Virions / uL (log)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_fig, dpi=200)
    print(f" -> Figure sauvegardee sous : {save_fig}")
    return t, y


def simuler_traitement(eta=0.95, jours=180, h=0.05, save_fig="simulation_traitement.png"):
    P = dict(v.PA)
    # Partir de l'equilibre endemique non traite
    E_star = np.array([250.0, 15.0, 15.0 * P["p"] / P["c"]])
    r0_therap = v.R0_h(P, eta)
    eta_c = 1.0 - 1.0 / v.R0_h(P, 0.0)
    
    print(f"\n[2/3] Simulation du TRAITEMENT ANTIRETROVIRAL (eta = {eta*100:.1f} %)...")
    print(f" -> Seuil critique d'efficacite eta_c = {eta_c*100:.1f} %")
    print(f" -> R0 sous traitement = {r0_therap:.2f} ({'Eradication potentielle' if r0_therap < 1 else 'Controle partiel'})")
    
    t, y = resoudre_intra_implicite((0, jours), E_star, P, eta_fn=lambda _: eta, h=h)
    T, I, V = y[:, 0], y[:, 1], y[:, 2]
    
    # Seuil d'indetectabilite (ex: 0.05 virion/uL = 50 copies/mL)
    undetect = np.where(V < 0.05)[0]
    t_undetect = t[undetect[0]] if len(undetect) > 0 else None
    if t_undetect:
        print(f" -> Charge virale indetectable (< 50 copies/mL) des le jour {t_undetect:.1f}")
    else:
        print(f" -> Charge virale finale : {V[-1]:.2f} virions/uL (pas d'indetectabilite atteinte)")
    print(f" -> Restauration immunitaire finale CD4+ : {T[-1]:.1f} / uL (depart : {T[0]:.1f})")
    
    # Graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, T, 'b-', lw=2, label="CD4+ sains T(t)")
    ax1.axhline(P["s"]/P["d"], color='darkgreen', linestyle=':', label="Niveau sain T0=1000")
    ax1.set_ylabel("Cellules / uL")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right")
    ax1.set_title(f"Simulation du Traitement ARV (eta = {eta*100:.1f}%, R0 = {r0_therap:.2f})")
    
    ax2.semilogy(t, np.maximum(V, 1e-6), 'r-', lw=2, label="Charge virale V(t)")
    ax2.axhline(0.05, color='gray', linestyle='--', label="Seuil indetectabilite (50 cop/mL)")
    if t_undetect:
        ax2.axvline(t_undetect, color='purple', linestyle='--', label=f"Indetectable j {t_undetect:.1f}")
    ax2.set_xlabel("Temps (jours)")
    ax2.set_ylabel("Virions / uL (log)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_fig, dpi=200)
    print(f" -> Figure sauvegardee sous : {save_fig}")
    return t, y


def simuler_rebond(eta=0.95, t_arret=100, jours=250, h=0.05, save_fig="simulation_rebond.png"):
    P = dict(v.PA)
    E_star = np.array([250.0, 15.0, 15.0 * P["p"] / P["c"]])
    print(f"\n[3/3] Simulation de l'ARRET DU TRAITEMENT au jour {t_arret}...")
    
    def eta_fn(t):
        return eta if t < t_arret else 0.0
        
    t, y = resoudre_intra_implicite((0, jours), E_star, P, eta_fn=eta_fn, h=h)
    T, I, V = y[:, 0], y[:, 1], y[:, 2]
    
    # Rebond apres arret
    apres = np.where(t >= t_arret)[0]
    idx_rebond = apres[np.argmax(V[apres])]
    t_rebond = t[idx_rebond]
    v_rebond = V[idx_rebond]
    ratio = v_rebond / E_star[2]
    
    print(f" -> CD4+ au moment de l'arret (j={t_arret}) : {T[np.where(t >= t_arret)[0][0]]:.1f} / uL")
    print(f" -> Pic de rebond viral : {v_rebond:,.1f} virions/uL au jour {t_rebond:.1f}")
    print(f" -> Facteur d'amplification par rapport a l'equilibre chronique : x{ratio:.1f}")
    
    # Graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, T, 'b-', lw=2, label="CD4+ sains T(t)")
    ax1.axvline(t_arret, color='black', linestyle='--', lw=1.5, label=f"Arret du traitement (j {t_arret})")
    ax1.set_ylabel("Cellules / uL")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title(f"Simulation du Rebond Viral apres Arret du Traitement (j {t_arret})")
    
    ax2.semilogy(t, np.maximum(V, 1e-4), 'r-', lw=2, label="Charge virale V(t)")
    ax2.axvline(t_arret, color='black', linestyle='--', lw=1.5)
    ax2.axvline(t_rebond, color='darkred', linestyle=':', label=f"Pic de rebond (j {t_rebond:.1f}, x{ratio:.1f})")
    ax2.axhline(E_star[2], color='green', linestyle=':', label=f"Equilibre E*={E_star[2]:.0f}")
    ax2.set_xlabel("Temps (jours)")
    ax2.set_ylabel("Virions / uL (log)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_fig, dpi=200)
    print(f" -> Figure sauvegardee sous : {save_fig}")
    return t, y


def simuler_population(annees=300, save_fig="simulation_population.png"):
    print(f"\n[Simulation Population] Comparaison des 4 strategies sur {annees} ans...")
    scenarios = [
        ("Sans intervention",         1.0, 0.50),
        ("Prevention (beta x 0.5)",   0.5, 0.50),
        ("Traitement (mu2 = 0.10)",   1.0, 0.10),
        ("Combinaison (prev+trait)",  0.5, 0.10),
    ]
    
    plt.figure(figsize=(9, 5))
    y0 = [0.01, 0.001] # 1% infectes asymptomatiques, 0.1% SIDA
    
    for nom, fac_beta, mu2 in scenarios:
        P = dict(v.PB)
        P["beta"] = v.PB["beta"] * fac_beta
        P["mu2"] = mu2
        r0 = v.R0_B(P, P["beta"])
        
        def f_sys(t, y):
            i1, i2 = y
            lam = P["Lam"]
            s = 1.0 - i1 - i2
            f1 = P["beta"] * s * (i1 + P["eps"] * i2) - (P["sigma"] + lam) * i1
            f2 = P["sigma"] * i1 - (P["mu2"] + lam) * i2
            return [f1, f2]
            
        sol = solve_ivp(f_sys, (0, annees), y0, t_eval=np.linspace(0, annees, 301), method="Radau")
        prev = (sol.y[0] + sol.y[1]) * 100 # %
        plt.plot(sol.t, prev, lw=2, label=f"{nom} (R0 = {r0:.2f}, prev. finale = {prev[-1]:.1f}%)")
        print(f" -> {nom:<26} | R0 = {r0:.3f} | Prevalence a {annees} ans = {prev[-1]:.2f} %")
        
    plt.xlabel("Temps (annees)")
    plt.ylabel("Prevalence totale VIH+SIDA (%)")
    plt.title("Simulation Epidemiologique en Population : Impact des Interventions")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=200)
    print(f" -> Figure sauvegardee sous : {save_fig}")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulation numerique des modeles VIH (Groupe 5)")
    parser.add_argument("--modele", choices=["intra", "population", "tous"], default="intra",
                        help="Modele a simuler : intra (T, I, V) ou population (S, I1, I2)")
    parser.add_argument("--scenar", choices=["primo", "traitement", "rebond", "tous"], default="tous",
                        help="Scenario intra-hote : primo, traitement, rebond ou tous")
    parser.add_argument("--eta", type=float, default=0.95,
                        help="Efficacite du traitement antiretroviral (ex: 0.95 pour 95%%)")
    parser.add_argument("--jours", type=int, default=180,
                        help="Duree de simulation en jours (intra-hote)")
    parser.add_argument("--arret", type=int, default=100,
                        help="Jour d'arret du traitement pour le scenario rebond")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("   MINI-PROJET OCS - SIMULATION DU MODELE VIH/SIDA (GROUPE 5)")
    print("=" * 70)
    
    if args.modele in ["intra", "tous"]:
        if args.scenar in ["primo", "tous"]:
            simuler_primo(jours=min(args.jours, 60))
        if args.scenar in ["traitement", "tous"]:
            simuler_traitement(eta=args.eta, jours=args.jours)
        if args.scenar in ["rebond", "tous"]:
            simuler_rebond(eta=args.eta, t_arret=args.arret, jours=max(args.jours, args.arret + 150))
            
    if args.modele in ["population", "tous"]:
        simuler_population()
        
    print("\n" + "=" * 70)
    print("Simulations terminees avec succes. Tous les graphiques ont ete generes !")
    print("=" * 70)

if __name__ == "__main__":
    main()
