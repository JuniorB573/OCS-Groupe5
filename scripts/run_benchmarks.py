"""
Script d'exécution des benchmarks de performance comparative :
- Méthode lente (Python pur) vs Intel MKL / BLAS 3.
- Inversion explicite naïve vs Cholesky LAPACK (DPOTRF+DPOTRS).
- Passage à l'échelle multithread OpenMP / Intel MKL (Speedup et Efficacité).
Génère les tableaux de métriques et les courbes de comparaison graphique.
"""

import os
import sys
import matplotlib.pyplot as plt

# Permet l'import de src quel que soit le dossier d'exécution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark import (
    get_hpc_system_info,
    benchmark_simulation_naive_vs_mkl,
    benchmark_large_cohort_scaling,
    benchmark_lapack_cholesky_vs_naive,
    benchmark_thread_scalability,
)


def print_table(title: str, headers: list, rows: list):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)
    col_widths = [max(len(h), max((len(f"{r[i]}") for r in rows), default=0)) + 3 for i, h in enumerate(headers)]
    
    header_str = "".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "".join("-" * (w - 1) + " " for w in col_widths)
    print(header_str)
    print(sep_str)
    
    for row in rows:
        row_str = "".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row))
        print(row_str)
    print("=" * 80)


def main():
    os.makedirs("figures", exist_ok=True)
    info = get_hpc_system_info()

    print("\n" + "#" * 80)
    print("  CAMPAGNE DE BENCHMARK HPC : BLAS / LAPACK / INTEL MKL vs MÉTHODES NAÏVES")
    print(f"  Backend détecté : {info['blas_backend']}")
    print(f"  Threads détectés : {info['detected_threads']}")
    print("#" * 80)

    # 1. Benchmark Naïf vs MKL
    print("\n[Benchmark 1/4] Simulation de Cohorte : Python Pur vs BLAS 3 / MKL...")
    cohort_sizes = [50, 100, 200, 400]
    res_naive_mkl = benchmark_simulation_naive_vs_mkl(cohort_sizes=cohort_sizes, days=30.0, dt=0.05, threads=4)

    rows_1 = [
        [
            r["n_patients"],
            f"{r['time_naive_ms']:.1f} ms",
            f"{r['time_mkl_ms']:.2f} ms",
            f"{r['speedup']:.1f} x",
            f"{r['throughput_patients_per_sec']:,.0f} pat/s",
        ]
        for r in res_naive_mkl
    ]
    print_table(
        "Comparatif Simulation : Python Pur (Naïf) vs BLAS 3 / Intel MKL",
        ["Patients (N)", "Temps Naïf", "Temps BLAS 3/MKL", "Facteur Speedup", "Débit MKL"],
        rows_1,
    )

    # 2. Benchmark Montée en charge grandes cohortes & GFLOPS
    print("\n[Benchmark 2/4] Montée en Charge Grandes Cohortes (1 000 à 10 000 patients)...")
    large_cohorts = [1000, 2500, 5000, 10000]
    res_large = benchmark_large_cohort_scaling(cohort_sizes=large_cohorts, days=60.0, dt=0.05, threads=4)

    rows_2 = [
        [
            r["n_patients"],
            f"{r['time_ms']:.1f} ms",
            f"{r['gflops']:.2f} GFLOPS",
            f"{r['throughput_patients_per_sec']:,.0f} pat/s",
        ]
        for r in res_large
    ]
    print_table(
        "Débit GFLOPS et Scalabilité Grandes Cohortes (MKL 4 Threads)",
        ["Patients (N)", "Temps Simulation", "Débit Calcul", "Débit Simulation"],
        rows_2,
    )

    # 3. Benchmark LAPACK Cholesky vs Inversion Naïve
    print("\n[Benchmark 3/4] Analyse Clinique : LAPACK Cholesky (DPOTRF+DPOTRS) vs Inversion...")
    res_lapack = benchmark_lapack_cholesky_vs_naive(cohort_sizes=[1000, 5000, 10000, 25000], n_features=15)

    rows_3 = [
        [
            r["n_patients"],
            f"{r['time_naive_ms']:.3f} ms",
            f"{r['time_lapack_ms']:.3f} ms",
            f"{r['speedup']:.2f} x",
        ]
        for r in res_lapack
    ]
    print_table(
        "Régression Clinique : Inversion Explicite vs LAPACK Cholesky",
        ["Patients (N)", "Inversion Naïve", "LAPACK Cholesky", "Gain Speedup"],
        rows_3,
    )

    # 4. Benchmark Scalabilité Multi-Threads OpenMP / MKL
    print("\n[Benchmark 4/4] Passage à l'Échelle Multithread OpenMP / Intel MKL...")
    thread_counts = [1, 2, 4, 8]
    res_threads = benchmark_thread_scalability(n_patients=10000, thread_counts=thread_counts, days=60.0)

    rows_4 = [
        [
            r["threads"],
            f"{r['time_ms']:.1f} ms",
            f"{r['speedup']:.2f} x",
            f"{r['efficiency_percent']:.1f} %",
        ]
        for r in res_threads
    ]
    print_table(
        "Scalabilité Parallèle Intel MKL sur 10 000 Patients",
        ["Cœurs / Threads", "Temps d'exécution", "Speedup Sp", "Efficacité Ep"],
        rows_4,
    )

    # --- Tracé des Graphiques de Benchmark ---
    print("\n[Graphiques] Enregistrement des courbes de benchmark...")

    # Figure 1 : Speedup Naïf vs MKL
    fig, ax1 = plt.subplots(figsize=(8, 5))
    pts = [r["n_patients"] for r in res_naive_mkl]
    sp = [r["speedup"] for r in res_naive_mkl]
    t_naive = [r["time_naive_ms"] for r in res_naive_mkl]
    t_mkl = [r["time_mkl_ms"] for r in res_naive_mkl]

    ax1.plot(pts, sp, marker="o", lw=2.5, color="#2b6cb0", label="Facteur d'Accélération (Speedup)")
    ax1.set_xlabel("Taille de la cohorte (Nombre de patients)", fontsize=11)
    ax1.set_ylabel("Facteur d'accélération (x plus rapide)", color="#2b6cb0", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accélération de la Simulation : Python Pur vs BLAS 3 / Intel MKL", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig_path1 = "figures/benchmark_naive_vs_mkl.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {fig_path1}")

    # Figure 2 : Scalabilité OpenMP / MKL
    fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(13, 5))
    th = [r["threads"] for r in res_threads]
    speedups = [r["speedup"] for r in res_threads]
    efficiencies = [r["efficiency_percent"] for r in res_threads]

    ax_sp.plot(th, speedups, marker="s", lw=2.5, color="#2c7a7b", label="Mesure MKL")
    ax_sp.plot(th, th, linestyle="--", color="gray", label="Idéal linéaire (Speedup = p)")
    ax_sp.set_title("Speedup Parallèle OpenMP / MKL", fontsize=11, fontweight="bold")
    ax_sp.set_xlabel("Nombre de Threads", fontsize=10)
    ax_sp.set_ylabel("Speedup $S(p) = T_1 / T_p$", fontsize=10)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.legend()

    ax_eff.plot(th, efficiencies, marker="^", lw=2.5, color="#d69e2e", label="Efficacité $E(p)$")
    ax_eff.axhline(100.0, linestyle="--", color="gray", label="Efficacité 100%")
    ax_eff.set_title("Efficacité Parallèle (%)", fontsize=11, fontweight="bold")
    ax_eff.set_xlabel("Nombre de Threads", fontsize=10)
    ax_eff.set_ylabel("Efficacité $E(p) = (S_p / p) \\times 100$", fontsize=10)
    ax_eff.set_ylim(0, 110)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend()

    plt.tight_layout()
    fig_path2 = "figures/benchmark_mkl_threads.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  -> Enregistré : {fig_path2}")

    print("\nCampagne de benchmark achevée avec succès !")


if __name__ == "__main__":
    main()
