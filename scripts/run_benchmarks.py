#!/usr/bin/env python3
"""
Script d'exécution des benchmarks haute performance BLAS/LAPACK/MKL
Projet VIH/SIDA HPC - Groupe 5 (ENSGMM / UNSTIM)
"""

import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ajout du dossier racine au sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.benchmark import (
    get_blas_info,
    matmul_naive,
    benchmark_dgemm,
    benchmark_dgesv,
    benchmark_thread_scaling,
)

FIG_DIR = root_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def print_banner():
    info = get_blas_info()
    print("=" * 80)
    print("      BENCHMARK HAUTE PERFORMANCE : BLAS 3 (DGEMM) & LAPACK (DGESV)")
    print("      Projet de Modélisation VIH/SIDA - Groupe 5 (ENSGMM / UNSTIM)")
    print("=" * 80)
    print(f"[*] Bibliothèque BLAS/LAPACK active en mémoire : {info['backend']}")
    print("=" * 80)


def benchmark_naive_vs_blas():
    print("\n[+] Comparaison : Pur Python Naïf O(n^3) vs BLAS 3 DGEMM vectorisé")
    print("-" * 80)
    print(f"{'Taille N':<10} | {'Python Naïf (s)':<18} | {'BLAS DGEMM (s)':<18} | {'Facteur Accélération':<20}")
    print("-" * 80)
    
    sizes = [50, 100, 200, 300]
    results = []
    
    for n in sizes:
        A_py = [[1.0] * n for _ in range(n)]
        B_py = [[2.0] * n for _ in range(n)]
        
        t0 = time.perf_counter()
        _ = matmul_naive(A_py, B_py)
        t_naive = time.perf_counter() - t0
        
        # Test DGEMM
        res_dgemm = benchmark_dgemm([n], threads=4)[0]
        t_blas = res_dgemm["time_ms"] / 1000.0
        
        speedup = t_naive / t_blas if t_blas > 0 else np.nan
        print(f"{n:<10} | {t_naive:<18.5f} | {t_blas:<18.6f} | x {speedup:<18.1f}")
        results.append((n, t_naive, t_blas, speedup))
    print("-" * 80)
    return results


def benchmark_matrix_scalability():
    print("\n[+] Débit crête et passage à l'échelle en taille de matrice (N = 400 à 3200)")
    print("-" * 80)
    print(f"{'N':<8} | {'DGEMM (ms)':<14} | {'DGEMM (GFLOPS)':<16} | {'DGESV (ms)':<14} | {'DGESV (GFLOPS)':<16}")
    print("-" * 80)
    
    sizes = [400, 800, 1200, 1600, 2400]
    res_gemm = benchmark_dgemm(sizes, threads=4)
    res_gesv = benchmark_dgesv(sizes, threads=4)
    
    for g, s in zip(res_gemm, res_gesv):
        print(f"{g['n']:<8} | {g['time_ms']:<14.2f} | {g['gflops']:<16.2f} | {s['time_ms']:<14.2f} | {s['gflops']:<16.2f}")
    print("-" * 80)
    
    # Génération du graphique de GFLOPS
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot([r['n'] for r in res_gemm], [r['gflops'] for r in res_gemm], 'o-', label="BLAS 3 : DGEMM (Multithread)", color="#2980b9", lw=2.2)
    ax.plot([r['n'] for r in res_gesv], [r['gflops'] for r in res_gesv], 's-', label="LAPACK : DGESV (Solveur LU)", color="#27ae60", lw=2.2)
    ax.set_xlabel("Taille de la matrice $N$")
    ax.set_ylabel("Débit de calcul (GFLOPS)")
    ax.set_title("Performance arithmétique crête des micro-noyaux BLAS/LAPACK")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = FIG_DIR / "blas_lapack_gflops_scaling.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Graphique GFLOPS sauvegardé : {out_path}")


def benchmark_multithreading():
    print("\n[+] Scalabilité OpenMP Multithreading (Taille N = 1600)")
    print("-" * 80)
    print(f"{'Threads':<10} | {'Temps (ms)':<14} | {'Speedup':<12} | {'Efficacité (%)':<16} | {'GFLOPS':<14}")
    print("-" * 80)
    
    threads_list = [1, 2, 4]
    scaling = benchmark_thread_scaling(matrix_size=1600, thread_counts=threads_list)
    
    for s in scaling:
        print(f"{s['threads']:<10} | {s['time_ms']:<14.2f} | {s['speedup']:<12.2f} | {s['efficiency']:<16.1f} | {s['gflops']:<14.2f}")
    print("-" * 80)
    
    # Graphique de speedup
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    threads = [s['threads'] for s in scaling]
    speedups = [s['speedup'] for s in scaling]
    
    ax1.plot(threads, speedups, 'o-', color="#8e44ad", lw=2.5, label="Speedup mesuré")
    ax1.plot(threads, threads, '--', color="#7f8c8d", label="Speedup linéaire idéal")
    ax1.set_xlabel("Nombre de threads OpenMP")
    ax1.set_ylabel("Facteur d'accélération (Speedup)")
    ax1.set_title("Scalabilité parallèle OpenMP sur micro-noyau DGEMM (N = 1600)")
    ax1.set_xticks(threads)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = FIG_DIR / "openmp_thread_speedup.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Graphique Speedup sauvegardé : {out_path}")
    print("\n[V] BENCHMARK TERMINÉ AVEC SUCCÈS.")


if __name__ == "__main__":
    print_banner()
    benchmark_naive_vs_blas()
    benchmark_matrix_scalability()
    benchmark_multithreading()
