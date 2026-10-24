# Modélisation Multi-Patient de l'Infection par le VIH-SIDA et Analyse Pronostique Haute Performance (BLAS, LAPACK & Intel oneMKL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HPC: BLAS/LAPACK](https://img.shields.io/badge/HPC-BLAS%20%7C%20LAPACK%20%7C%20MKL-orange.svg)](#architecture-hpc--formulation-matricielle)
[![Status: Production](https://img.shields.io/badge/build-passing-brightgreen.svg)](#tests-et-validation-numérique)
[![Institution: ENSGMM / UNSTIM](https://img.shields.io/badge/ENSGMM-UNSTIM%20Abomey-red.svg)](https://unstim.bj)

> **Projet de Recherche et d'Ingénierie Numérique HPC**  
> **Institution :** École Nationale Supérieure de Génie Mathématique et Modélisation (ENSGMM)  
> **Université :** Université Nationale des Sciences, Technologies, Ingénierie et Mathématiques (UNSTIM), Abomey, Bénin  
> **Promotion :** ENSGMM 1 (GMM1) — Année Académique 2025–2026  
> **Encadrant Scientifique :** Dr. Carlos AGOSSOU  
> **Auteurs (Groupe 5) :**
> - **ADOGOUN Déo Gracias Mahugnon**
> - **BIAHOU Ahossou Junior Marie-Clémenceau**
> - **SYLLA IMOROU Izou Dine**

---

## Sommaire
1. [Vue d'Ensemble](#vue-densemble)
2. [Modélisation Biomathématique](#modélisation-biomathématique)
   - [Système Différentiel à 5 Compartiments](#1-système-différentiel-à-5-compartiments)
   - [Pression Thérapeutique ART et Échappement Mutant](#2-pression-thérapeutique-art-et-échappement-mutant)
3. [Architecture HPC & Formulation Matricielle](#architecture-hpc--formulation-matricielle)
   - [Hiérarchie BLAS et Mur de la Mémoire](#1-hiérarchie-blas-et-mur-de-la-mémoire)
   - [Formulation Matricielle Globale](#2-formulation-matricielle-globale)
   - [Intégrateur RK4 Vectorisé par Micro-Noyaux BLAS 3 (`DGEMM`)](#3-intégrateur-rk4-vectorisé-par-micro-noyaux-blas-3-dgemm)
   - [Analyse Pronostique par Cholesky LAPACK (`DPOTRF`/`DPOTRS`)](#4-analyse-pronostique-par-cholesky-lapack-dpotrfdpotrs)
   - [Stratification Clinique par ACP LAPACK (`DSYEVD`)](#5-stratification-clinique-par-acp-lapack-dsyevd)
4. [Installation, Configuration et Utilisation des Bibliothèques](#installation-configuration-et-utilisation-des-bibliothèques)
   - [Prérequis Matériels et Logiciels](#1-prérequis-matériels-et-logiciels)
   - [Installation de l'Environnement](#2-installation-de-lenvironnement)
   - [Configuration d'Intel oneMKL et OpenMP](#3-configuration-dintel-onemkl-et-openmp)
5. [Guide d'Exécution](#guide-dexécution)
   - [Lancement de la Simulation de Cohorte](#1-lancement-de-la-simulation-de-cohorte)
   - [Lancement du Banc d'Essai HPC (Benchmarks)](#2-lancement-du-banc-dessai-hpc-benchmarks)
   - [Exécution des Tests Unitaires et Numériques](#3-exécution-des-tests-unitaires-et-numériques)
   - [Compilation des Documents LaTeX (Rapport & Présentation)](#4-compilation-des-documents-latex-rapport--présentation)
6. [Résultats Scientifiques et Débits HPC](#résultats-scientifiques-et-débits-hpc)
7. [Arborescence du Projet](#arborescence-du-projet)
8. [Licence](#licence)

---

## Vue d'Ensemble

Ce dépôt contient le code source complet, les bancs d'essai haute performance et la documentation scientifique d'un simulateur de cohortes cliniques massives pour la dynamique du VIH-1 avec émergence de souches mutantes résistantes.

À l'échelle hospitalière ou dans le cadre d'essais cliniques multicentriques, l'évaluation de stratégies thérapeutiques nécessite de simuler des cohortes de plusieurs milliers de patients ($N = 1\,000$ à $10\,000$) sur des horizons de un à plusieurs années. Alors qu'une approche séquentielle naïve en Python pur s'avère prohibitivement lente, notre implémentation condense l'ensemble de la cohorte dans un tenseur matriciel d'état continu $\mathbf{Y} \in \mathbb{R}^{N \times 5}$ stocké en mémoire Fortran contiguë (zéro-copie), transformant chaque étape d'intégration Runge-Kutta 4 (RK4) en produits matriciels de niveau **BLAS 3** (`DGEMM`).

L'analyse pronostique des biomarqueurs et la prédiction de la charge virale résiduelle s'effectuent via la résolution d'une régression régularisée de Ridge exploitant la factorisation de **Cholesky LAPACK** (`DPOTRF` et `DPOTRS`), complétée par une stratification non-supervisée par Analyse en Composantes Principales (ACP) via l'algorithme spectral Divide & Conquer de LAPACK (`DSYEVD`).

---

## Modélisation Biomathématique

### 1. Système Différentiel à 5 Compartiments

Pour chaque patient $k \in \{1, \dots, N\}$, la dynamique intra-hôte prend en compte la coexistence entre la souche virale sauvage (*wild-type*, sensible) et une quasi-espèce mutante résistante au traitement :

$$\begin{cases}
\dfrac{dT_k}{dt} = s - d\,T_k - (1 - \eta_{s,k})\,\beta_s\,T_k\,V_{s,k} - (1 - \eta_{r,k})\,\beta_r\,T_k\,V_{r,k} \\[8pt]
\dfrac{dI_{s,k}}{dt} = (1 - \mu)\,(1 - \eta_{s,k})\,\beta_s\,T_k\,V_{s,k} - \delta\,I_{s,k} \\[8pt]
\dfrac{dI_{r,k}}{dt} = \mu\,(1 - \eta_{s,k})\,\beta_s\,T_k\,V_{s,k} + (1 - \eta_{r,k})\,\beta_r\,T_k\,V_{r,k} - \delta\,I_{r,k} \\[8pt]
\dfrac{dV_{s,k}}{dt} = p\,I_{s,k} - c\,V_{s,k} \\[8pt]
\dfrac{dV_{r,k}}{dt} = p\,I_{r,k} - c\,V_{r,k}
\end{cases}$$

Où les variables d'état sont :
- $T_k(t)$ : Concentration de lymphocytes T CD4+ sains ($\mu\mathrm{L}^{-1}$).
- $I_{s,k}(t)$ : Cellules CD4+ infectées productrices de virus sauvage ($\mu\mathrm{L}^{-1}$).
- $I_{r,k}(t)$ : Cellules CD4+ infectées productrices de virus résistant ($\mu\mathrm{L}^{-1}$).
- $V_{s,k}(t)$ : Charge virale plasmatique de la souche sensible (virions/$\mu\mathrm{L}$).
- $V_{r,k}(t)$ : Charge virale plasmatique de la souche résistante (virions/$\mu\mathrm{L}$).

### 2. Pression Thérapeutique ART et Échappement Mutant

- **Taux de mutation ponctuelle :** $\mu = 10^{-4}$ par cycle de réplication.
- **Efficacité thérapeutique :** Chaque patient présente un taux d'adhésion individuel $\eta_{s,k} \in [0.50, 0.95]$.
- **Pénalité de résistance (coût de fitness) :** Le virus mutant présente un taux d'infectiosité légèrement inférieur ($\beta_r = 0.85\,\beta_s$), mais échappe en grande partie à l'inhibition antirétrovirale :
$$\eta_{r,k} = \max(0, \eta_{s,k} - 0.35)$$

---

## Architecture HPC & Formulation Matricielle

### 1. Hiérarchie BLAS et Mur de la Mémoire

| Niveau BLAS | Opération Type | Accès Mémoire | Flops | Intensité $q$ | Comportement Matériel |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **BLAS 1** | $\mathbf{y} \leftarrow \alpha\mathbf{x} + \mathbf{y}$ (`DAXPY`) | $3n$ | $2n$ | $\mathcal{O}(1)$ | Goulot d'étranglement bus mémoire RAM |
| **BLAS 2** | $\mathbf{y} \leftarrow \alpha\mathbf{A}\mathbf{x} + \beta\mathbf{y}$ (`DGEMV`) | $n^2$ | $2n^2$ | $\mathcal{O}(1)$ | Saturation rapide du cache |
| **BLAS 3** | $\mathbf{C} \leftarrow \alpha\mathbf{A}\mathbf{B} + \beta\mathbf{C}$ (`DGEMM`) | $3n^2$ | $2n^3$ | $\mathcal{O}(n)$ | **Efficacité crête : réutilisation en registres/cache L1/L2** |

### 2. Formulation Matricielle Globale

En regroupant les $N$ patients dans la matrice d'état $\mathbf{Y} \in \mathbb{R}^{N \times 5}$, l'ensemble du système différentiel s'exprime sous la forme :

$$\mathbf{\dot{Y}} = \mathbf{S} + \mathbf{Y}\,\mathbf{M}^T + \mathbf{\Phi}\,\mathbf{A}^T$$

- **Terme source constant :** $\mathbf{S} = \mathbf{1}_N \otimes (s, 0, 0, 0, 0) \in \mathbb{R}^{N \times 5}$
- **Opérateur linéaire :** $\mathbf{M} \in \mathbb{R}^{5 \times 5}$ encapsulant la mortalité cellulaire et la clairance virale.
- **Flux non-linéaires d'infection :** $\mathbf{\Phi} \in \mathbb{R}^{N \times 2}$ avec $\mathbf{\Phi}_{k,0} = (1-\eta_{s,k})\beta_s T_k V_{s,k}$ et $\mathbf{\Phi}_{k,1} = (1-\eta_{r,k})\beta_r T_k V_{r,k}$.
- **Matrice stœchiométrique de couplage :** $\mathbf{A} \in \mathbb{R}^{5 \times 2}$.

### 3. Intégrateur RK4 Vectorisé par Micro-Noyaux BLAS 3 (`DGEMM`)

L'évaluation de la fonction de second membre $\mathbf{F}(\mathbf{Y})$ calcule simultanément pour les $N$ patients :
$$\mathbf{Y}\,\mathbf{M}^T \quad (\text{appel BLAS 3 \texttt{DGEMM}}), \qquad \mathbf{\Phi}\,\mathbf{A}^T \quad (\text{appel BLAS 3 \texttt{DGEMM}})$$
L'intégrateur Runge-Kutta d'ordre 4 évalue quatre incréments partiels $\mathbf{K}_1, \dots, \mathbf{K}_4$ entièrement sous forme de blocs matriciels continus, éliminant totalement l'interpréteur CPython au cours de l'intégration temporelle.

### 4. Analyse Pronostique par Cholesky LAPACK (`DPOTRF`/`DPOTRS`)

Pour prédire la charge virale à 1 an à partir des biomarqueurs à $t = 30$ jours, la régression Ridge résout :
$$(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})\,\mathbf{w} = \mathbf{X}^T\mathbf{y}$$
1. Calcul du produit symétrique rang-$k$ par **BLAS 3 `DSYRK`**.
2. Factorisation de Cholesky $\mathbf{A}_{\text{ridge}} = \mathbf{L}\mathbf{L}^T$ par **LAPACK `DPOTRF`** ($n^3/3$ flops, deux fois plus rapide que la factorisation LU `DGESV`).
3. Résolution par descente et remontée triangulaires par **LAPACK `DPOTRS`**.

### 5. Stratification Clinique par ACP LAPACK (`DSYEVD`)

La matrice de covariance empirique des profils patients $\mathbf{\Sigma} \in \mathbb{R}^{5 \times 5}$ est diagonalisée par l'algorithme *Divide and Conquer* **LAPACK `DSYEVD`**, garantissant l'orthogonalité exacte des composantes principales et une projection instantanée des $N$ patients par **BLAS 3 `DGEMM`**.

---

## Installation, Configuration et Utilisation des Bibliothèques

### 1. Prérequis Matériels et Logiciels
- **Système d'exploitation :** Windows 10/11, Linux (Ubuntu 20.04+) ou macOS.
- **Python :** Version 3.10 ou supérieure.
- **Processeur :** Architecture x86_64 compatible SIMD (AVX-2 ou AVX-512 recommandé).
- **Compilateur LaTeX (optionnel) :** MiKTeX ou TeX Live pour la génération des rapports.

### 2. Installation de l'Environnement

```bash
# Cloner le dépôt
git clone https://github.com/JuniorB573/OCS-Groupe5.git
cd OCS-Groupe5

# Créer et activer un environnement virtuel Python
python -m venv venv

# Sous Windows (PowerShell) :
.\venv\Scripts\Activate.ps1
# Sous Linux / macOS :
# source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration d'Intel oneMKL et OpenMP

Sous les distributions Python scientifiques modernes (distribution standard ou conda), NumPy et SciPy sont pré-compilés avec **Intel oneMKL** ou **OpenBLAS**.

Pour contrôler explicitement le nombre de threads d'exécution parallèle pour les micro-noyaux BLAS/LAPACK :

```bash
# Sous Windows (PowerShell) :
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

# Sous Linux / macOS (Bash) :
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

Vérification directe de la pile logicielle liée en Python :
```python
import numpy as np
np.show_config()  # Affiche les bibliothèques BLAS/LAPACK liées (mkl_rt, openblas, etc.)
```

---

## Guide d'Exécution

### 1. Lancement de la Simulation de Cohorte
Simule une cohorte de 5 000 patients sur 365 jours, réalise l'analyse pronostique LAPACK et génère les figures cliniques :

```bash
python scripts/run_simulation.py
```
*Sorties générées :*
- `figures/cohort_trajectories.png` : Évolution des compartiments $T$, $V_s$ et $V_r$ (répression vs rebond mutant).
- `figures/clinical_pca_clusters.png` : Stratification des patients dans le plan factoriel ACP.

### 2. Lancement du Banc d'Essai HPC (Benchmarks)
Exécute la suite complète de profilage (Naïf vs MKL, Cholesky vs Inversion, Scalabilité OpenMP) :

```bash
python scripts/run_benchmarks.py
```
*Sorties générées :*
- `figures/benchmark_naive_vs_mkl.png` : Comparatif de temps d'exécution et facteur d'accélération.
- `figures/benchmark_mkl_threads.png` : Scalabilité et speedup multi-cœurs OpenMP.

### 3. Exécution des Tests Unitaires et Numériques
Vérifie l'équivalence numérique absolue entre l'implémentation vectorisée et l'implémentation scalaire naïve ($\epsilon < 10^{-10}$), les invariants biologiques et les solveurs LAPACK :

```bash
pytest tests/test_equivalence.py -v
```

### 4. Compilation des Documents LaTeX (Rapport & Présentation)

```bash
# Compiler le rapport académique officiel (37 pages)
pdflatex -interaction=nonstopmode -output-directory=docs docs/rapport_projet.tex

# Compiler la présentation Beamer
pdflatex -interaction=nonstopmode -output-directory=docs docs/presentation_beamer.tex
```

---

## Résultats Scientifiques et Débits HPC

### 1. Accélération Vectorielle BLAS 3 vs Python Naïf

| Taille Cohorte ($N$) | Temps Python Naïf (s) | Temps BLAS 3 / MKL (s) | Facteur d'Accélération (Speedup) |
| :---: | :---: | :---: | :---: |
| 50 | 0,311 s | 0,042 s | **7,4×** |
| 100 | 0,618 s | 0,078 s | **7,9×** |
| 200 | 1,224 s | 0,154 s | **7,9×** |
| 400 | 2,462 s | 0,299 s | **8,2×** |

### 2. Montée en Charge et Débits à Grande Échelle

| Patients ($N$) | Temps MKL (s) | Débit (patients/s) | Flops Effectués (MFlops) | GFLOPS Effectifs |
| :---: | :---: | :---: | :---: | :---: |
| 500 | 0,336 s | 1 488 | 401,5 | 1,19 GFLOPS |
| 1 000 | 0,596 s | 1 678 | 803,0 | 1,35 GFLOPS |
| 2 000 | 1,162 s | 1 721 | 1 606,0 | 1,38 GFLOPS |
| 5 000 | 2,930 s | 1 706 | 4 015,0 | 1,37 GFLOPS |
| 10 000 | 4,923 s | **2 031** | 8 030,0 | **1,63 GFLOPS** |

### 3. Factorisation de Cholesky LAPACK vs Inversion Naïve

| Algorithme | Routine / Méthode | Temps d'Exécution ($\mu\mathrm{s}$) | Gain Relatif | Stabilité Numérique |
| :--- | :--- | :---: | :---: | :---: |
| Inversion Naïve | `inv(A) @ y` | 254 $\mu\mathrm{s}$ | 1,0× | Sensible au mauvais conditionnement |
| **Cholesky LAPACK** | **`DPOTRF` + `DPOTRS`** | **111 $\mu\mathrm{s}$** | **2,3×** | **Inconditionnellement stable** |

---

## Arborescence du Projet

```text
OCS-Groupe5/
├── .gitignore                      # Fichiers et artefacts exclus du contrôle de version
├── LICENSE                         # Licence open-source MIT
├── pyproject.toml                  # Configuration du projet et de pytest
├── requirements.txt                # Dépendances Python minimales
├── README.md                       # Documentation scientifique et technique complète
│
├── src/                            # Modules Python du simulateur haute performance
│   ├── __init__.py                 # Point d'entrée du package
│   ├── cohort_model.py             # Formulation matricielle et intégrateur RK4 BLAS 3
│   ├── naive_model.py              # Modèle scalaire naïf de référence pour benchmark
│   ├── lapack_analysis.py          # Régression Ridge (Cholesky) et ACP (DSYEVD)
│   └── benchmark.py                # Bancs d'essai comparatifs et mesure GFLOPS
│
├── scripts/                        # Scripts exécutables pour reproduction
│   ├── run_simulation.py           # Simulation d'une cohorte de 5 000 patients
│   └── run_benchmarks.py           # Suite complète de benchmarks et profilage
│
├── tests/                          # Tests unitaires et d'équivalence numérique
│   └── test_equivalence.py         # Tests pytest (5 tests d'invariance et de dérive)
│
├── figures/                        # Graphiques scientifiques et logos institutionnels
│   ├── logo_ensgmm.png             # Logo officiel ENSGMM
│   ├── logo_unstim.png             # Logo officiel UNSTIM
│   ├── cohort_trajectories.png     # Trajectoires cliniques de cohorte sur 365 jours
│   ├── clinical_pca_clusters.png   # Stratification des profils patients par ACP
│   ├── benchmark_naive_vs_mkl.png  # Comparatif temps et speedup Naïf vs MKL
│   └── benchmark_mkl_threads.png   # Scalabilité multi-threads OpenMP
│
└── docs/                           # Documentation LaTeX officielle
    ├── rapport_projet.tex          # Code source LaTeX du rapport académique (37 pages)
    ├── rapport_projet.pdf          # Rapport académique compilé
    ├── Rapport_Projet_VIH_SIDA_Groupe5.pdf # Copie officielle pour archivage
    ├── presentation_beamer.tex     # Code source LaTeX de la soutenance Beamer
    └── Presentation_Beamer_VIH_SIDA_Groupe5.pdf # Support de soutenance compilé
```

---

## Licence

Ce projet est publié sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.  
Développé avec rigueur et passion par l'équipe du **Groupe 5** (Promotion ENSGMM 1, UNSTIM Abomey).
