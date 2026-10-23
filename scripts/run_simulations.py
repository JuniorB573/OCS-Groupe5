#!/usr/bin/env python3
"""
Script d'exécution des simulations scientifiques (intra-hôte et population)
Projet VIH/SIDA HPC - Groupe 5 (ENSGMM / UNSTIM)
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ajout du dossier racine au sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.intra_host import (
    DEFAULT_INTRA_PARAMS,
    compute_r0,
    compute_critical_efficacy,
    disease_free_equilibrium,
    endemic_equilibrium,
    solve_equilibrium_newton,
    simulate_euler_implicit,
    simulate_euler_explicit,
)
from src.population import (
    DEFAULT_POP_PARAMS,
    compute_pop_r0,
    simulate_population,
)

# Configuration graphique professionnelle
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
})

FIG_DIR = root_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_intra_host_simulations():
    print("=" * 70)
    print("1. SIMULATION INTRA-HÔTE : DYNAMIQUE CELLULAIRE ET EFFET DU TRAITEMENT")
    print("=" * 70)
    
    r0_0 = compute_r0(eta=0.0)
    eta_c = compute_critical_efficacy()
    print(f"[*] R0 sans traitement (eta = 0.0) : {r0_0:.3f}")
    print(f"[*] Efficacité antirétrovirale critique : eta_c = {eta_c * 100:.2f} %")
    
    # 1. Équilibres stationnaires via Newton-LAPACK (DGESV)
    y0_guess = np.array([500.0, 10.0, 1000.0])
    y_star, iters, ok = solve_equilibrium_newton(y0_guess, eta=0.0)
    print(f"[*] Convergence Newton-DGESV en {iters} itérations :")
    print(f"    T* = {y_star[0]:.2f} CD4+/uL | I* = {y_star[1]:.2f} cellules/uL | V* = {y_star[2]:.2f} virions/uL")
    
    # 2. Simulations temporelles avec différents profils de traitement
    t_span = (0.0, 120.0)  # 120 jours
    h = 0.05               # Pas de temps
    y0 = np.array([1000.0, 1.0, 10.0])  # Infection primaire aiguë
    
    efficacies = [0.0, 0.50, 0.85, 0.95]
    colors = ['#c0392b', '#e67e22', '#2980b9', '#27ae60']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    
    for eta, col in zip(efficacies, colors):
        t_eval, y_hist = simulate_euler_implicit(y0, t_span, h=h, eta=eta)
        r0_val = compute_r0(eta=eta)
        label = f"$\\eta = {eta*100:.0f}\\%$ ($R_0 = {r0_val:.2f}$)"
        
        axes[0].plot(t_eval, y_hist[:, 0], label=label, color=col)
        axes[1].plot(t_eval, y_hist[:, 1], label=label, color=col)
        axes[2].plot(t_eval, y_hist[:, 2], label=label, color=col)
        
    axes[0].set_ylabel(r"Lymphocytes $T$ sains ($\mu\mathrm{L}^{-1}$)")
    axes[0].set_title(r"Dynamique intra-hôte du VIH : Lymphocytes CD4+ sains $T(t)$")
    axes[0].legend(loc="lower left", framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel(r"Cellules infectées $I$ ($\mu\mathrm{L}^{-1}$)")
    axes[1].set_title(r"Cellules infectées productrices $I(t)$")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel(r"Charge virale $V$ (virions/$\mu\mathrm{L}$)")
    axes[2].set_xlabel("Temps (jours après primo-infection)")
    axes[2].set_yscale("log")
    axes[2].set_title(r"Charge virale libre $V(t)$ (échelle log)")
    axes[2].legend(loc="upper right", framealpha=0.9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = FIG_DIR / "intra_host_treatment_response.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Graphique généré avec succès : {out_path}")


def run_stiffness_comparison():
    print("\n" + "=" * 70)
    print("2. RAIDEUR NUMÉRIQUE : EULER IMPLICITE (LAPACK DGESV) VS EXPLICITE")
    print("=" * 70)
    
    t_span = (0.0, 30.0)
    y0 = np.array([1000.0, 1.0, 10.0])
    
    # Avec h = 0.1 jour (au-dessus du seuil de stabilité d'Euler explicite)
    h_test = 0.10
    print(f"[*] Test de raideur avec pas de discrétisation h = {h_test} jour")
    
    t_imp, y_imp = simulate_euler_implicit(y0, t_span, h=h_test, eta=0.0)
    t_exp, y_exp = simulate_euler_explicit(y0, t_span, h=h_test, eta=0.0)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_imp, y_imp[:, 2], label="Euler implicite (A-stable, DGESV)", color="#27ae60", lw=2.5)
    
    valid_idx = ~np.isnan(y_exp[:, 2])
    ax.plot(t_exp[valid_idx], y_exp[valid_idx, 2], '--', label="Euler explicite (instabilité numérique)", color="#c0392b", lw=2)
    
    ax.set_yscale("log")
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel(r"Charge virale $V(t)$ (virions/$\mu\mathrm{L}$)")
    ax.set_title("Stabilité inconditionnelle Euler Implicite + LAPACK DGESV vs Euler Explicite")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = FIG_DIR / "stiffness_stability_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Graphique de stabilité généré : {out_path}")


def run_population_simulations():
    print("\n" + "=" * 70)
    print("3. SIMULATION MACRO-ÉPIDÉMIOLOGIQUE : MODÈLE S - I1 - I2 SUR 50 ANS")
    print("=" * 70)
    
    r0_pop = compute_pop_r0()
    print(f"[*] R0 populationnel : {r0_pop:.3f}")
    
    # 0.1% d'infectés initiaux dans une population saine
    y0 = (0.001, 0.0)
    t_span = (0.0, 50.0)
    
    t_eval, s_arr, i1_arr, i2_arr = simulate_population(y0, t_span, n_points=500)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t_eval, s_arr * 100, label=r"Susceptibles $s(t)$", color="#2980b9", lw=2.2)
    ax.plot(t_eval, i1_arr * 100, label=r"Infectés asymptomatiques $i_1(t)$", color="#e67e22", lw=2.2)
    ax.plot(t_eval, i2_arr * 100, label=r"Stade SIDA $i_2(t)$", color="#c0392b", lw=2.2)
    ax.plot(t_eval, (i1_arr + i2_arr) * 100, ':', label=r"Prévalence totale $(i_1 + i_2)(t)$", color="#8e44ad", lw=2.5)
    
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Proportion de la population (%)")
    ax.set_title(r"Dynamique macro-épidémiologique du VIH/SIDA sur 50 ans ($R_0^{\mathrm{pop}} = 1.80$)")
    ax.legend(loc="center right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = FIG_DIR / "population_epidemic_dynamics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Graphique populationnel généré : {out_path}")
    print("\n[V] TOUTES LES SIMULATIONS ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS.")


if __name__ == "__main__":
    run_intra_host_simulations()
    run_stiffness_comparison()
    run_population_simulations()
