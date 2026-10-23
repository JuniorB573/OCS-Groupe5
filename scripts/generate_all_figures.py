#!/usr/bin/env python3
"""
Générateur complet des figures scientifiques (Fig 3 à Fig 19)
Projet de Modélisation VIH/SIDA HPC - Groupe 5 (ENSGMM / UNSTIM)
Encadrant : Dr. Carlos AGOSSOU
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import lapack, hilbert
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ajout du dossier racine au sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.lapack_solvers import (
    solve_linear_system,
    balance_matrix,
    compute_eigenvalues,
    estimate_condition_number,
)
from src.intra_host import (
    DEFAULT_INTRA_PARAMS,
    compute_r0,
    compute_critical_efficacy,
    rhs_intra,
    jacobian_intra,
    disease_free_equilibrium,
    endemic_equilibrium,
    solve_equilibrium_newton,
    simulate_euler_implicit,
    simulate_euler_explicit,
)
from src.population import (
    DEFAULT_POP_PARAMS,
    compute_pop_r0,
    rhs_population,
    jacobian_population,
    simulate_population,
)
from src.benchmark import (
    benchmark_dgemm,
    benchmark_dgesv,
)

# Configuration graphique soignée
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'lines.linewidth': 1.8,
})

FIG_DIR = root_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig3_hilbert():
    """Fig 3 : Erreur relative, résidu et conditionnement sur matrices de Hilbert."""
    print("  -> Génération Fig 3 : Matrices de Hilbert et DGESV...")
    sizes = np.arange(2, 13)
    kappas = []
    residuals = []
    errors = []
    
    for n in sizes:
        H = hilbert(n)
        x_exact = np.ones(n, dtype=np.float64)
        b = H @ x_exact
        
        # Résolution par DGESV
        x_num, _, _ = solve_linear_system(H, b)
        
        # Conditionnement norme 1
        anorm = float(np.linalg.norm(H, 1))
        lu, piv, _ = lapack.dgetrf(np.asfortranarray(H))
        rcond, _ = lapack.dgecon(lu, anorm, norm="1")
        kappa = 1.0 / rcond if rcond > 0 else 1e18
        
        # Résidu et erreur relative
        res = np.linalg.norm(H @ x_num - b, 1) / (anorm * np.linalg.norm(x_num, 1))
        err = np.linalg.norm(x_num - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)
        
        kappas.append(kappa)
        residuals.append(res)
        errors.append(err)
        
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    
    ax1.semilogy(sizes, kappas, 'o-', color='#c0392b', label=r"Conditionnement $\kappa_1(H_n)$")
    ax2.semilogy(sizes, errors, 's--', color='#2980b9', label=r"Erreur relative $\|\hat{x}-x\|_\infty / \|x\|_\infty$")
    ax2.semilogy(sizes, residuals, '^:', color='#27ae60', label=r"Résidu normalisé $\|H\hat{x}-b\| / (\|H\|\|\hat{x}\|)$")
    
    eps_mach = np.finfo(float).eps
    ax2.axhline(eps_mach, color='#7f8c8d', linestyle='-.', alpha=0.7, label=r"Précision machine $\varepsilon_{\mathrm{mach}}$")
    
    ax1.set_xlabel("Dimension de la matrice de Hilbert $n$")
    ax1.set_ylabel(r"$\kappa_1(H_n)$", color='#c0392b')
    ax2.set_ylabel("Erreur relative et résidu", color='#2980b9')
    ax1.set_title("Stabilité rétrograde de LAPACK DGESV sur matrices de Hilbert")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_hilbert.png", dpi=300)
    plt.close()


def fig4_cond_jac():
    """Fig 4 : Évolution du conditionnement de la matrice Jacobienne J(t)."""
    print("  -> Génération Fig 4 : Conditionnement de la Jacobienne J(t)...")
    y0 = np.array([1000.0, 1.0, 10.0])
    t_span = (0.0, 100.0)
    t_eval, y_hist = simulate_euler_implicit(y0, t_span, h=0.05, eta=0.0)
    
    cond_list = []
    for y in y_hist:
        J = jacobian_intra(y, eta=0.0)
        try:
            anorm = float(np.linalg.norm(J, 1))
            lu, piv, info = lapack.dgetrf(np.asfortranarray(J))
            rcond, _ = lapack.dgecon(lu, anorm, norm="1")
            cond = 1.0 / rcond if rcond > 0 else np.nan
        except Exception:
            cond = np.linalg.cond(J, 1)
        cond_list.append(cond)
        
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    
    ax1.plot(t_eval, cond_list, color='#8e44ad', lw=2.2, label=r"Conditionnement $\kappa_1(J(t))$")
    ax2.plot(t_eval, y_hist[:, 2], color='#e74c3c', lw=1.8, linestyle='--', label=r"Charge virale $V(t)$")
    
    ax1.set_xlabel("Temps (jours après infection)")
    ax1.set_ylabel(r"Conditionnement $\kappa_1(J)$", color='#8e44ad')
    ax2.set_ylabel(r"Charge virale $V(t)$ (virions/$\mu\mathrm{L}$)", color='#e74c3c')
    ax2.set_yscale("log")
    ax1.set_title("Conditionnement de la Jacobienne le long de la trajectoire")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_cond_jac.png", dpi=300)
    plt.close()


def fig5_newton():
    """Fig 5 : Convergence quadratique de la méthode de Newton avec DGESV."""
    print("  -> Génération Fig 5 : Convergence quadratique de Newton-DGESV...")
    # Point de départ
    y = np.array([400.0, 5.0, 200.0], dtype=np.float64)
    y_star = endemic_equilibrium(eta=0.0)
    
    errors = []
    residuals = []
    
    for it in range(15):
        err = np.linalg.norm(y - y_star)
        f = rhs_intra(0.0, y, eta=0.0)
        res = np.linalg.norm(f)
        errors.append(err)
        residuals.append(res)
        
        if res < 1e-13:
            break
        J = jacobian_intra(y, eta=0.0)
        dy, _, _ = solve_linear_system(J, -f)
        y += dy
        
    iters = np.arange(len(errors))
    
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogy(iters, errors, 'o-', color='#2980b9', lw=2, label=r"Erreur $\|y_k - y^*\|_2$")
    ax.semilogy(iters, residuals, 's--', color='#e67e22', lw=2, label=r"Résidu $\|f(y_k)\|_2$")
    ax.set_xlabel("Itération de Newton $k$")
    ax.set_ylabel("Norme euclidienne")
    ax.set_title("Convergence quadratique de Newton-Raphson avec LAPACK DGESV")
    ax.set_xticks(iters)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_newton.png", dpi=300)
    plt.close()


def fig6_raideur():
    """Fig 6 : Test de raideur numérique (Euler Explicite vs Euler Implicite)."""
    print("  -> Génération Fig 6 : Raideur numérique et stabilité...")
    t_span = (0.0, 25.0)
    y0 = np.array([1000.0, 1.0, 10.0])
    
    # Pas stable et pas instable
    t_imp, y_imp = simulate_euler_implicit(y0, t_span, h=0.10, eta=0.0)
    t_exp_stable, y_exp_stable = simulate_euler_explicit(y0, t_span, h=0.04, eta=0.0)
    t_exp_unstable, y_exp_unstable = simulate_euler_explicit(y0, t_span, h=0.09, eta=0.0)
    
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t_imp, y_imp[:, 2], label=r"Euler Implicite ($h = 0.10$ j, A-stable)", color='#27ae60', lw=2.4)
    ax.plot(t_exp_stable, y_exp_stable[:, 2], '--', label=r"Euler Explicite ($h = 0.04$ j, sous le seuil)", color='#2980b9', lw=1.8)
    
    val_idx = ~np.isnan(y_exp_unstable[:, 2]) & (y_exp_unstable[:, 2] < 1e8)
    ax.plot(t_exp_unstable[val_idx], y_exp_unstable[val_idx, 2], ':', label=r"Euler Explicite ($h = 0.09$ j, explosion)", color='#c0392b', lw=2.2)
    
    ax.set_yscale("log")
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel(r"Charge virale $V(t)$ (virions/$\mu\mathrm{L}$)")
    ax.set_title("Raideur différentielle du système intra-hôte : Euler Implicite vs Explicite")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_raideur.png", dpi=300)
    plt.close()


def fig7_convergence():
    """Fig 7 : Ordre de convergence temporel O(h) du schéma d'Euler Implicite."""
    print("  -> Génération Fig 7 : Ordre de convergence O(h)...")
    t_span = (0.0, 5.0)
    y0 = np.array([1000.0, 1.0, 10.0])
    
    # Solution de référence avec pas ultra-fin h_ref = 0.0005
    _, y_ref_hist = simulate_euler_implicit(y0, t_span, h=0.0005, eta=0.0)
    y_exact = y_ref_hist[-1]
    
    h_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125])
    errors = []
    
    for h in h_values:
        _, y_h = simulate_euler_implicit(y0, t_span, h=h, eta=0.0)
        err = np.linalg.norm(y_h[-1] - y_exact, np.inf)
        errors.append(err)
        
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog(h_values, errors, 'o-', color='#e67e22', lw=2.2, label=r"Erreur globale $\|y(T; h) - y_{\mathrm{ref}}(T)\|_\infty$")
    ax.loglog(h_values, errors[0] * (h_values / h_values[0]), '--', color='#7f8c8d', label=r"Pente théorique d'ordre 1 ($\mathcal{O}(h)$)")
    
    ax.set_xlabel(r"Pas de discrétisation temporelle $h$ (jours)")
    ax.set_ylabel("Erreur absolue globale")
    ax.set_title("Vérification numérique de l'ordre de convergence d'Euler Implicite")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_convergence.png", dpi=300)
    plt.close()


def fig8_infection_primaire():
    """Fig 8 : Dynamique aiguë de primo-infection (pic virale et chute des CD4+)."""
    print("  -> Génération Fig 8 : Primo-infection aiguë...")
    t_span = (0.0, 120.0)
    y0 = np.array([1000.0, 1.0, 10.0])
    t_eval, y_hist = simulate_euler_implicit(y0, t_span, h=0.05, eta=0.0)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(t_eval, y_hist[:, 0], color='#2980b9', lw=2.2, label=r"Lymphocytes CD4+ sains $T(t)$")
    l2 = ax2.plot(t_eval, y_hist[:, 2], color='#c0392b', lw=2.2, label=r"Charge virale $V(t)$ (virions/$\mu\mathrm{L}$)")
    
    ax1.set_xlabel("Temps (jours après contamination)")
    ax1.set_ylabel(r"Lymphocytes CD4+ ($\mu\mathrm{L}^{-1}$)", color='#2980b9')
    ax2.set_ylabel(r"Virions libres ($\mu\mathrm{L}^{-1}$)", color='#c0392b')
    ax2.set_yscale("log")
    
    # Annotation du pic viral
    idx_peak = np.argmax(y_hist[:, 2])
    t_peak = t_eval[idx_peak]
    v_peak = y_hist[idx_peak, 2]
    ax2.annotate(f'Pic virémique aigu\n({t_peak:.1f} j, {v_peak:.0f} virions)',
                 xy=(t_peak, v_peak), xytext=(t_peak + 15, v_peak * 0.4),
                 arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=9, fontweight='bold', color='#c0392b')
                 
    ax1.set_title("Cinétique naturelle de primo-infection par le VIH-1")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_infection_primaire.png", dpi=300)
    plt.close()


def fig9_portrait_phase():
    """Fig 9 : Portrait de phase dans l'espace d'état (T, V)."""
    print("  -> Génération Fig 9 : Portrait de phase (T, V)...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    init_points = [
        [1000.0, 1.0, 10.0],
        [800.0, 5.0, 50.0],
        [500.0, 20.0, 500.0],
        [200.0, 10.0, 1000.0],
    ]
    colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad']
    
    for y0, col in zip(init_points, colors):
        t_eval, y_hist = simulate_euler_implicit(np.array(y0), (0.0, 150.0), h=0.05, eta=0.0)
        ax.plot(y_hist[:, 0], y_hist[:, 2], color=col, lw=1.8, label=f"CI : $T_0={y0[0]:.0f}, V_0={y0[2]:.0f}$")
        ax.plot(y0[0], y0[2], 'o', color=col)
        
    # Équilibre endémique
    estar = endemic_equilibrium(eta=0.0)
    ax.plot(estar[0], estar[2], 'r*', markersize=14, label=r"Attracteur endémique $E^*$")
    
    ax.set_xlabel(r"Cellules T CD4+ saines ($\mu\mathrm{L}^{-1}$)")
    ax.set_ylabel(r"Charge virale $V$ (virions/$\mu\mathrm{L}$)")
    ax.set_yscale("log")
    ax.set_title(r"Portrait de phase et convergence vers l'équilibre endémique $E^*$")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_portrait_phase.png", dpi=300)
    plt.close()


def fig10_r0_eta():
    """Fig 10 : Courbe R0(eta) et seuil critique eta_c = 75%."""
    print("  -> Génération Fig 10 : Seuil critique R0(eta)...")
    eta_vals = np.linspace(0.0, 1.0, 200)
    r0_vals = [compute_r0(eta=e) for e in eta_vals]
    eta_c = compute_critical_efficacy()
    
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(eta_vals * 100, r0_vals, color='#c0392b', lw=2.4, label=r"$\mathcal{R}_0(\eta) = (1-\eta)\mathcal{R}_0(0)$")
    ax.axhline(1.0, color='#27ae60', linestyle='--', lw=2.0, label=r"Seuil épidémique critique $\mathcal{R}_0 = 1$")
    ax.axvline(eta_c * 100, color='#2980b9', linestyle=':', lw=2.0, label=rf"Efficacité critique $\eta_c = {eta_c*100:.1f}\%$")
    
    # Zones de contrôle et d'endémie
    ax.fill_between(eta_vals * 100, r0_vals, 1.0, where=(np.array(r0_vals) >= 1.0), color='#e74c3c', alpha=0.15, label="Zone d'infection chronique")
    ax.fill_between(eta_vals * 100, r0_vals, 1.0, where=(np.array(r0_vals) <= 1.0), color='#2ecc71', alpha=0.15, label="Zone d'extinction virale")
    
    ax.set_xlabel(r"Efficacité du traitement antirétroviral $\eta$ (%)")
    ax.set_ylabel(r"Nombre de reproduction de base $\mathcal{R}_0$")
    ax.set_title(r"Sensibilité de $\mathcal{R}_0$ à l'efficacité de la trithérapie")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_r0_eta.png", dpi=300)
    plt.close()


def fig11_traitement():
    """Fig 11 : Efficacité du traitement antirétroviral continu (eta = 95%)."""
    print("  -> Génération Fig 11 : Réponse au traitement continu...")
    t_span = (0.0, 90.0)
    y0 = np.array([1000.0, 1.0, 10.0])
    
    t_eval, y_treat = simulate_euler_implicit(y0, t_span, h=0.05, eta=0.95)
    _, y_notreat = simulate_euler_implicit(y0, t_span, h=0.05, eta=0.0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    
    ax1.plot(t_eval, y_notreat[:, 0], '--', color='#c0392b', label=r"Sans traitement ($\eta = 0\%$)")
    ax1.plot(t_eval, y_treat[:, 0], '-', color='#27ae60', lw=2.2, label=r"Trithérapie efficace ($\eta = 95\%$)")
    ax1.set_ylabel(r"CD4+ sains $T(t)$ ($\mu\mathrm{L}^{-1}$)")
    ax1.set_title("Restauration du pool de lymphocytes T CD4+")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_eval, y_notreat[:, 2], '--', color='#c0392b', label=r"Sans traitement ($\eta = 0\%$)")
    ax2.plot(t_eval, y_treat[:, 2], '-', color='#27ae60', lw=2.2, label=r"Trithérapie efficace ($\eta = 95\%$)")
    ax2.axhline(1.0, color='#7f8c8d', linestyle=':', label="Seuil de détectabilité plasmatique")
    ax2.set_yscale("log")
    ax2.set_xlabel("Temps (jours)")
    ax2.set_ylabel(r"Charge virale $V(t)$ (virions/$\mu\mathrm{L}$)")
    ax2.set_title("Effondrement de la virémie sous antirétroviraux")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_traitement.png", dpi=300)
    plt.close()


def fig12_bifurcation_intra():
    """Fig 12 : Diagramme de bifurcation transcritique intra-hôte."""
    print("  -> Génération Fig 12 : Bifurcation transcritique...")
    r0_array = np.linspace(0.2, 5.0, 300)
    v_star = []
    
    p = DEFAULT_INTRA_PARAMS
    t0 = p["s"] / p["d"]
    
    for r0 in r0_array:
        if r0 <= 1.0:
            v_star.append(0.0)
        else:
            ts = t0 / r0
            is_val = (p["s"] - p["d"] * ts) / p["delta"]
            vs = (p["p"] / p["c"]) * is_val
            v_star.append(vs)
            
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(r0_array[r0_array <= 1.0], [0.0]*sum(r0_array <= 1.0), color='#27ae60', lw=2.5, label="DFE stable ($E_0$)")
    ax.plot(r0_array[r0_array >= 1.0], [0.0]*sum(r0_array >= 1.0), '--', color='#7f8c8d', lw=1.8, label="DFE instable")
    ax.plot(r0_array[r0_array >= 1.0], [v for r, v in zip(r0_array, v_star) if r >= 1.0], color='#c0392b', lw=2.5, label="Équilibre endémique stable ($E^*$)")
    
    ax.axvline(1.0, color='#2980b9', linestyle=':', lw=1.8, label=r"Point de bifurcation transcritique ($\mathcal{R}_0 = 1$)")
    ax.set_xlabel(r"Nombre de reproduction de base $\mathcal{R}_0$")
    ax.set_ylabel(r"Charge virale stationnaire $V^*$ (virions/$\mu\mathrm{L}$)")
    ax.set_title("Diagramme de bifurcation transcritique du modèle intra-hôte")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_bifurcation_intra.png", dpi=300)
    plt.close()


def fig13_amplification():
    """Fig 13 : Phénomène de rebond viral lors d'un arrêt thérapeutique."""
    print("  -> Génération Fig 13 : Rebond viral après interruption...")
    # Phase 1 : Primo infection sans traitement (0 à 30j)
    y0 = np.array([1000.0, 1.0, 10.0])
    t1, y1 = simulate_euler_implicit(y0, (0.0, 30.0), h=0.05, eta=0.0)
    
    # Phase 2 : Traitement fort (30 à 100j, eta = 0.95)
    t2, y2 = simulate_euler_implicit(y1[-1], (30.0, 100.0), h=0.05, eta=0.95)
    
    # Phase 3 : Arrêt thérapeutique brutal (100 à 160j, eta = 0.0)
    t3, y3 = simulate_euler_implicit(y2[-1], (100.0, 160.0), h=0.05, eta=0.0)
    
    t_tot = np.concatenate([t1, t2[1:], t3[1:]])
    y_tot = np.vstack([y1, y2[1:], y3[1:]])
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_tot, y_tot[:, 2], color='#c0392b', lw=2.2, label=r"Charge virale $V(t)$")
    
    ax.axvspan(30, 100, color='#2ecc71', alpha=0.2, label=r"Période sous trithérapie ($\eta = 95\%$)")
    ax.axvline(100, color='#e74c3c', linestyle='--', lw=2.0, label="Interruption du traitement (j = 100)")
    
    ax.set_yscale("log")
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel(r"Charge virale libre $V(t)$ (virions/$\mu\mathrm{L}$)")
    ax.set_title("Rebond virologique aigu suite à l'arrêt du traitement")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_amplification.png", dpi=300)
    plt.close()


def fig14_sensibilite():
    """Fig 14 : Indices de sensibilité normalisés des paramètres sur R0."""
    print("  -> Génération Fig 14 : Analyse de sensibilité de R0...")
    # R0 = beta * p * s / (d * delta * c)
    # Indices analytiques : Upsilon_x = (x / R0) * (d R0 / dx)
    params = ['beta (infectivité)', 'p (production virale)', 's (source CD4+)',
              'd (mortalité CD4+)', 'delta (mortalité infectées)', 'c (clairance virale)']
    indices = [+1.0, +1.0, +1.0, -1.0, -1.0, -1.0]
    colors = ['#2980b9' if idx > 0 else '#e74c3c' for idx in indices]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(params, indices, color=colors, height=0.55)
    ax.axvline(0.0, color='black', lw=1.0)
    ax.set_xlabel(r"Indice de sensibilité normalisé $\Upsilon_p^{\mathcal{R}_0} = \frac{\partial \mathcal{R}_0}{\partial p} \frac{p}{\mathcal{R}_0}$")
    ax.set_title(r"Sensibilité paramétrique de $\mathcal{R}_0$ (Échelle intra-hôte)")
    ax.set_xlim(-1.3, 1.3)
    
    for bar, val in zip(bars, indices):
        ax.text(val + (0.05 if val > 0 else -0.15), bar.get_y() + bar.get_height()/2.0,
                f"{val:+.1f}", va='center', fontweight='bold', color=bar.get_facecolor())
                
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig14_sensibilite.png", dpi=300)
    plt.close()


def fig16_population_dynamique():
    """Fig 16 : Dynamique macro-épidémiologique sur 50 ans."""
    print("  -> Génération Fig 16 : Dynamique populationnelle sur 50 ans...")
    y0 = (0.001, 0.0)
    t_span = (0.0, 50.0)
    t_eval, s_arr, i1_arr, i2_arr = simulate_population(y0, t_span, n_points=500)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_eval, s_arr * 100, label=r"Susceptibles $s(t)$", color='#2980b9', lw=2.2)
    ax.plot(t_eval, i1_arr * 100, label=r"Infectés asymptomatiques $i_1(t)$", color='#e67e22', lw=2.2)
    ax.plot(t_eval, i2_arr * 100, label=r"Stade SIDA $i_2(t)$", color='#c0392b', lw=2.2)
    ax.plot(t_eval, (i1_arr + i2_arr) * 100, ':', label=r"Prévalence totale $(i_1 + i_2)(t)$", color='#8e44ad', lw=2.5)
    
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Proportion de la population (%)")
    ax.set_title(r"Dynamique épidémiologique du VIH/SIDA à l'échelle de la population ($R_0^{\mathrm{pop}} = 1.80$)")
    ax.legend(loc="center right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig16_population_dynamique.png", dpi=300)
    plt.close()


def fig17_bifurcation_pop():
    """Fig 17 : Diagramme de bifurcation épidémiologique."""
    print("  -> Génération Fig 17 : Bifurcation populationnelle...")
    beta_range = np.linspace(0.05, 0.50, 150)
    p = DEFAULT_POP_PARAMS
    
    prev_endemic = []
    for b in beta_range:
        params_tmp = p.copy()
        params_tmp["beta"] = b
        r0 = compute_pop_r0(params_tmp)
        if r0 <= 1.0:
            prev_endemic.append(0.0)
        else:
            # Simulation à long terme pour atteindre l'équilibre
            _, _, i1, i2 = simulate_population((0.01, 0.0), (0.0, 200.0), n_points=50, params=params_tmp)
            prev_endemic.append((i1[-1] + i2[-1]) * 100)
            
    fig, ax = plt.subplots(figsize=(8, 4.8))
    beta_crit = p["beta"] / compute_pop_r0(p)
    
    ax.plot(beta_range[beta_range <= beta_crit], [0.0]*sum(beta_range <= beta_crit), color='#27ae60', lw=2.5, label="Équilibre sans infection stable")
    ax.plot(beta_range[beta_range >= beta_crit], [v for b, v in zip(beta_range, prev_endemic) if b >= beta_crit], color='#c0392b', lw=2.5, label="Prévalence endémique stationnaire")
    ax.axvline(beta_crit, color='#2980b9', linestyle=':', lw=2.0, label=r"Taux critique $\beta_c$ ($\mathcal{R}_0^{\mathrm{pop}} = 1$)")
    
    ax.set_xlabel(r"Taux de contact / transmission $\beta$ ($\text{an}^{-1}$)")
    ax.set_ylabel("Prévalence stationnaire du VIH (%)")
    ax.set_title("Bifurcation transcritique du modèle épidémiologique de population")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig17_bifurcation_pop.png", dpi=300)
    plt.close()


def fig18_interventions():
    """Fig 18 : Comparaison de scénarios d'interventions de santé publique."""
    print("  -> Génération Fig 18 : Scénarios d'interventions épidémiologiques...")
    t_span = (0.0, 40.0)
    y0 = (0.05, 0.01)  # Épidémie déjà installée (6% de prévalence)
    
    p_base = DEFAULT_POP_PARAMS.copy()
    
    # Scénario 1 : Base (sans intervention additionnelle, R0 = 1.80)
    t, _, i1_base, i2_base = simulate_population(y0, t_span, params=p_base)
    
    # Scénario 2 : Promotion du préservatif (réduction de transmission de 30%)
    p_condom = p_base.copy()
    p_condom["beta"] *= 0.70
    _, _, i1_cond, i2_cond = simulate_population(y0, t_span, params=p_condom)
    
    # Scénario 3 : Dépistage massif + TARV précoce (réduction transmission de 55%)
    p_tarv = p_base.copy()
    p_tarv["beta"] *= 0.45
    p_tarv["sigma"] *= 0.50  # Ralentissement de la progression vers le SIDA
    _, _, i1_tarv, i2_tarv = simulate_population(y0, t_span, params=p_tarv)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, (i1_base + i2_base) * 100, color='#c0392b', lw=2.2, label=r"Scénario de référence ($\mathcal{R}_0 = 1.80$)")
    ax.plot(t, (i1_cond + i2_cond) * 100, color='#e67e22', lw=2.2, label=r"Préservatif & Prévention ($\beta \downarrow 30\%$, $\mathcal{R}_0 = 1.26$)")
    ax.plot(t, (i1_tarv + i2_tarv) * 100, color='#27ae60', lw=2.4, label=r"Dépistage + TARV universel ($\mathcal{R}_0 = 0.81 < 1$)")
    
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Prévalence globale de l'infection (%)")
    ax.set_title("Impact comparé des stratégies de santé publique sur la prévalence")
    ax.legend(framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig18_interventions.png", dpi=300)
    plt.close()


def fig19_benchmark_perf():
    """Fig 19 : Performance arithmétique crête BLAS/LAPACK (GFLOPS et temps)."""
    print("  -> Génération Fig 19 : Benchmark de performances HPC...")
    sizes = [200, 400, 800, 1200, 1600, 2400]
    res_gemm = benchmark_dgemm(sizes, threads=4)
    res_gesv = benchmark_dgesv(sizes, threads=4)
    
    n_vals = [r['n'] for r in res_gemm]
    gflops_gemm = [r['gflops'] for r in res_gemm]
    gflops_gesv = [r['gflops'] for r in res_gesv]
    
    time_gemm = [r['time_ms'] for r in res_gemm]
    time_gesv = [r['time_ms'] for r in res_gesv]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    # 1. Débit de calcul en GFLOPS
    ax1.plot(n_vals, gflops_gemm, 'o-', color='#2980b9', lw=2.2, label="BLAS 3 : DGEMM")
    ax1.plot(n_vals, gflops_gesv, 's-', color='#27ae60', lw=2.2, label="LAPACK : DGESV")
    ax1.set_xlabel("Taille de la matrice $N$")
    ax1.set_ylabel("Débit de calcul (GFLOPS)")
    ax1.set_title("Débit arithmétique effectif")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temps d'exécution en ms (échelle log)
    ax2.plot(n_vals, time_gemm, 'o--', color='#2980b9', lw=2.0, label="DGEMM (ms)")
    ax2.plot(n_vals, time_gesv, 's--', color='#27ae60', lw=2.0, label="DGESV (ms)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Taille de la matrice $N$")
    ax2.set_ylabel("Temps d'exécution (ms, échelle log)")
    ax2.set_title("Temps de résolution numérique")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig19_benchmark_perf.png", dpi=300)
    plt.close()


def generate_all():
    print("=" * 80)
    print("   GÉNÉRATION INTÉGRALE DES 16 FIGURES SCIENTIFIQUES (FIG 3 À FIG 19)")
    print("   Projet VIH/SIDA HPC - Groupe 5 (ENSGMM / UNSTIM)")
    print("=" * 80)
    
    fig3_hilbert()
    fig4_cond_jac()
    fig5_newton()
    fig6_raideur()
    fig7_convergence()
    fig8_infection_primaire()
    fig9_portrait_phase()
    fig10_r0_eta()
    fig11_traitement()
    fig12_bifurcation_intra()
    fig13_amplification()
    fig14_sensibilite()
    fig16_population_dynamique()
    fig17_bifurcation_pop()
    fig18_interventions()
    fig19_benchmark_perf()
    
    print("\n[V] TOUTES LES 16 FIGURES ONT ÉTÉ GÉNÉRÉES AVEC SUCCÈS DANS LE DOSSIER 'figures/'.")
    print("=" * 80)


if __name__ == "__main__":
    generate_all()
