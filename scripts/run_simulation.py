"""
Script principal de simulation clinique de la cohorte VIH (5 000 patients sous trithérapie).
Intégration temporelle haute performance via BLAS 3 / MKL et analyse pronostique via LAPACK.
Génère les graphiques d'évolution clinique et de détection des résistances.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Permet l'import de src quel que soit le dossier d'exécution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cohort_model import (
    DEFAULT_PARAMS,
    build_cohort_parameters,
    initialize_cohort_state,
    simulate_cohort_rk4,
)
from src.lapack_analysis import (
    build_clinical_design_matrix,
    solve_ridge_cholesky_lapack,
    compute_cohort_pca_lapack,
)


def main():
    print("=" * 75)
    print("  SIMULATION CLINIQUE HAUTE PERFORMANCE D'UNE COHORTE VIH (BLAS 3 / MKL)")
    print("=" * 75)

    n_patients = 5000
    days = 365.0
    dt = 0.05
    p = DEFAULT_PARAMS

    os.makedirs("figures", exist_ok=True)

    print(f"\n[1/4] Initialisation de la cohorte ({n_patients:,} patients, horizon {days} jours)...")
    S, eta, M, A = build_cohort_parameters(n_patients=n_patients, params=p, seed=2026)
    Y0 = initialize_cohort_state(n_patients=n_patients, S=S, params=p, seed=2026)

    print(f"  - CD4 initiaux moyens : {np.mean(Y0[:, 0]):.1f} ± {np.std(Y0[:, 0]):.1f} cellules/uL")
    print(f"  - Charge virale sauvage initiale moyenne : {np.mean(Y0[:, 3]):.1e} virions/mL")
    print(f"  - Observance moyenne du traitement (eta) : {np.mean(eta)*100:.1f} %")

    print("\n[2/4] Intégration temporelle matricielle (BLAS 3 / MKL RK4)...")
    t0 = time.perf_counter()
    times, trajectory = simulate_cohort_rk4(
        Y0=Y0,
        S=S,
        eta=eta,
        M=M,
        A=A,
        beta_s=p["beta_s"],
        beta_r=p["beta_r"],
        days=days,
        dt=dt,
        save_interval=20,
    )
    t_elapsed = time.perf_counter() - t0
    print(f"  -> Simulation achevée en {t_elapsed*1000:.2f} ms ({n_patients / t_elapsed:,.0f} patients/sec)")

    final_state = trajectory[-1]
    T_final = final_state[:, 0]
    Vs_final = final_state[:, 3]
    Vr_final = final_state[:, 4]
    V_total_final = Vs_final + Vr_final

    # Définition clinique : suppression virale si V_total < 50 copies/mL
    suppressed_mask = V_total_final < 50.0
    rebound_mask = (V_total_final >= 50.0) & (Vr_final > Vs_final)
    non_adherent_mask = (V_total_final >= 50.0) & (Vr_final <= Vs_final)

    rate_suppressed = np.mean(suppressed_mask) * 100.0
    rate_rebound = np.mean(rebound_mask) * 100.0
    rate_non_adh = np.mean(non_adherent_mask) * 100.0

    print("\n[3/4] Bilan Clinique à t = 365 jours :")
    print(f"  - Succès virologique (charge indétectable < 50 cop/mL) : {rate_suppressed:.1f} %")
    print(f"  - Échec par rebond de résistance virale (V_r > V_s)     : {rate_rebound:.1f} %")
    print(f"  - Échec par sous-observance thérapeutique              : {rate_non_adh:.1f} %")
    print(f"  - Taux moyen de CD4 final : {np.mean(T_final):.1f} cellules/uL")

    print("\n[4/4] Analyse Pronostique Multivariée (LAPACK Cholesky & ACP DSYEVD)...")
    X = build_clinical_design_matrix(Y0, eta, S)
    y_target = np.log10(V_total_final + 1.0)
    w_cholesky, stats = solve_ridge_cholesky_lapack(X, y_target, alpha=1e-3)

    print("  - Facteurs prédictifs de la charge virale résiduelle (Régression Ridge LAPACK) :")
    feature_names = ["Biais", "Nadir CD4 initial", "Log10 charge virale initiale", "Observance (eta)", "Génération thymique (s)"]
    for name, weight in zip(feature_names, w_cholesky):
        print(f"    * {name:<30} : {weight:+.4f}")
    print(f"  - Coefficient de détermination R² : {stats['r2']:.4f} (RMSE: {stats['rmse']:.4f})")

    # Projection ACP clinique
    projected, eigenvalues, components = compute_cohort_pca_lapack(final_state, n_components=2)
    var_exp = (eigenvalues[:2] / np.sum(eigenvalues)) * 100.0
    print(f"  - Variance expliquée par l'ACP LAPACK : Axe 1 = {var_exp[0]:.1f} %, Axe 2 = {var_exp[1]:.1f} %")

    # --- Génération des Figures ---
    print("\n[Graphiques] Sauvegarde des visualisations médicales...")

    # Figure 1 : Trajectoires cliniques moyennes selon le profil
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Évolution des CD4
    if np.any(suppressed_mask):
        axes[0].plot(times, np.mean(trajectory[:, suppressed_mask, 0], axis=1), color="#2b6cb0", lw=2.5, label="Succès virologique")
    if np.any(rebound_mask):
        axes[0].plot(times, np.mean(trajectory[:, rebound_mask, 0], axis=1), color="#c53030", lw=2.5, label="Rebond de résistance")
    if np.any(non_adherent_mask):
        axes[0].plot(times, np.mean(trajectory[:, non_adherent_mask, 0], axis=1), color="#d69e2e", lw=2.5, label="Sous-observance")
    axes[0].set_title("Restauration des Lymphocytes T CD4+", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Temps (jours)")
    axes[0].set_ylabel("CD4+ (cellules / uL)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Évolution de la charge virale sauvage Vs
    if np.any(suppressed_mask):
        axes[1].semilogy(times, np.mean(trajectory[:, suppressed_mask, 3], axis=1), color="#2b6cb0", lw=2.5, label="Sauvage (Succès)")
    if np.any(rebound_mask):
        axes[1].semilogy(times, np.mean(trajectory[:, rebound_mask, 3], axis=1), color="#c53030", lw=2.0, ls="--", label="Sauvage (Résistant)")
        axes[1].semilogy(times, np.mean(trajectory[:, rebound_mask, 4], axis=1), color="#9b2c2c", lw=2.5, label="Mutant V_r (Résistant)")
    axes[1].axhline(50.0, color="gray", linestyle=":", label="Seuil d'indétectabilité (50 cop/mL)")
    axes[1].set_title("Dynamique de Charge Virale (Sauvage vs Résistant)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Temps (jours)")
    axes[1].set_ylabel("Charge virale (virions / mL)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    # Répartition de l'observance thérapeutique
    axes[2].scatter(eta[suppressed_mask], np.log10(V_total_final[suppressed_mask] + 1), color="#2b6cb0", alpha=0.3, s=15, label="Succès")
    if np.any(rebound_mask):
        axes[2].scatter(eta[rebound_mask], np.log10(V_total_final[rebound_mask] + 1), color="#c53030", alpha=0.5, s=20, label="Résistance")
    if np.any(non_adherent_mask):
        axes[2].scatter(eta[non_adherent_mask], np.log10(V_total_final[non_adherent_mask] + 1), color="#d69e2e", alpha=0.5, s=20, label="Sous-observance")
    axes[2].set_title("Observance vs Charge Virale Finale", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Observance thérapeutique (eta)")
    axes[2].set_ylabel("Log10(Charge Virale Finale + 1)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig_path1 = "figures/cohort_trajectories.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {fig_path1}")

    # Figure 2 : Cartographie clinique par ACP LAPACK (DSYEVD)
    plt.figure(figsize=(8, 6))
    plt.scatter(projected[suppressed_mask, 0], projected[suppressed_mask, 1], color="#2b6cb0", alpha=0.4, s=15, label=f"Succès ({rate_suppressed:.1f}%)")
    if np.any(rebound_mask):
        plt.scatter(projected[rebound_mask, 0], projected[rebound_mask, 1], color="#c53030", alpha=0.6, s=25, label=f"Rebond Résistant ({rate_rebound:.1f}%)")
    if np.any(non_adherent_mask):
        plt.scatter(projected[non_adherent_mask, 0], projected[non_adherent_mask, 1], color="#d69e2e", alpha=0.6, s=25, label=f"Sous-observance ({rate_non_adh:.1f}%)")
    plt.title("Cartographie Clinique des Profils de Patients (ACP LAPACK DSYEVD)", fontsize=13, fontweight="bold")
    plt.xlabel(f"Axe Principal 1 ({var_exp[0]:.1f} % variance)")
    plt.ylabel(f"Axe Principal 2 ({var_exp[1]:.1f} % variance)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig_path2 = "figures/clinical_pca_clusters.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {fig_path2}")

    print("\nSimulation et analyse terminées avec succès !")


if __name__ == "__main__":
    main()
