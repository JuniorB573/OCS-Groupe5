# -*- coding: utf-8 -*-
"""
Regeneration des figures du rapport -- Groupe 5
================================================
Redessine les figures 3 a 14 (modele intra-hote, outils LAPACK, integration numerique)
et 16 a 19 (modele de population et benchmark) a partir du modele de vih_lapack.py.

Usage :
    python generate_figures.py [dossier_de_sortie]

Par defaut, les images sont ecrites dans le dossier "figures_regenerees" (le dossier
"figures" d'origine n'est jamais modifie). Pour remplacer les figures du rapport par
celles-ci, copier les fichiers .png dans "figures/" puis recompiler main.tex.

Dependances : numpy, scipy, matplotlib, vih_lapack.py (meme dossier).
Duree : environ une minute.
"""

import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lapack, blas, expm, hilbert
from scipy.optimize import fsolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vih_lapack as v

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures_regenerees"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})
STYLES = ["-", "--", "-.", ":"]

P = dict(v.PA)
EE = v.ee_h(P)
DFE = v.dfe_h(P)
Y0 = np.array([1000.0, 0.0, 1e-3])


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("ecrit :", path)


def radau(f, t_span, y0, **kw):
    return solve_ivp(f, t_span, y0, method="Radau", rtol=1e-10, atol=1e-14, **kw)


def hom(eta):
    return lambda t, y: v.rhs_h(t, y, P, eta)


# ---------------------------------------------------------------------------
# Fig. 3 : matrices de Hilbert et DGESV
# ---------------------------------------------------------------------------
def fig3():
    kap, err, res = [], [], []
    for n in range(2, 14):
        H = hilbert(n)
        xt = np.ones(n)
        b = H @ xt
        x, rcond, _ = v.lu_solve(H, b)
        kap.append(1.0 / rcond)
        err.append(np.linalg.norm(x - xt) / np.linalg.norm(xt))
        res.append(np.linalg.norm(b - H @ x) / (np.linalg.norm(H, 1) * np.linalg.norm(x)))
    kap = np.array(kap)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(kap, err, "o-", color="black", label="Erreur relative")
    ax.loglog(kap, np.maximum(res, 1e-17), "s--", color="black", label="Résidu relatif")
    ax.loglog(kap, kap * np.finfo(float).eps, ":", color="gray", label=r"$\kappa_1\,\varepsilon_{mach}$")
    ax.set_xlabel(r"Conditionnement $\kappa_1(H_n)$")
    ax.set_ylabel("Erreur")
    ax.legend()
    save(fig, "fig3_hilbert.png")


# ---------------------------------------------------------------------------
# Fig. 4 : conditionnement de la jacobienne en fonction de eta
# ---------------------------------------------------------------------------
def fig4():
    etas = np.linspace(0, 0.999, 400)
    kee, kdfe = [], []
    for e in etas:
        kdfe.append(v.cond1(v.jac_h(DFE, P, e)))
        kee.append(v.cond1(v.jac_h(v.ee_h(P, e), P, e)) if v.R0_h(P, e) > 1.0005 else np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].semilogy(100 * etas, kee, color="black")
    ax[0].set_title("(a) Jacobienne à l'équilibre endémique")
    ax[0].set_ylabel(r"$\kappa_1(J(EE))$")
    ax[1].semilogy(100 * etas, kdfe, color="black")
    ax[1].axvline(75, color="gray", ls=":")
    ax[1].set_title("(b) Jacobienne à l'équilibre sans infection")
    ax[1].set_ylabel(r"$\kappa_1(J(DFE))$")
    for a in ax:
        a.set_xlabel(r"Efficacité $\eta$ (%)")
    save(fig, "fig4_cond_jac.png")


# ---------------------------------------------------------------------------
# Fig. 5 : convergence de Newton
# ---------------------------------------------------------------------------
def fig5():
    starts = {"A": [50, 5, 100], "B": [900, 10, 500], "C": [100, 50, 5000],
              "D": [240, 14, 640], "E": [400, 5, 200]}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (name, y0), ls, mk in zip(starts.items(), ["-", "--", "-.", ":", "-"], "osv^x"):
        y = np.array(y0, float)
        steps = []
        for k in range(12):
            f = v.rhs_h(0, y, P)
            _, _, dy, info = lapack.dgesv(np.asfortranarray(v.jac_h(y, P)), -f)
            y = y + dy
            steps.append(max(np.linalg.norm(dy) / np.linalg.norm(y), 1e-17))
            if steps[-1] < 1e-15:
                break
        ax.semilogy(range(1, len(steps) + 1), steps, ls=ls, marker=mk, color="black", label=f"Départ {name}")
    ax.set_xlabel(r"Itération $k$")
    ax.set_ylabel(r"Pas relatif $\|\delta_k\|/\|y_k\|$")
    ax.legend()
    save(fig, "fig5_newton.png")


# ---------------------------------------------------------------------------
# Schemas d'integration
# ---------------------------------------------------------------------------
def euler_expl(y0, h, tf):
    n = int(round(tf / h))
    y = np.array(y0, float)
    out = [y.copy()]
    with np.errstate(all="ignore"):
        for _ in range(n):
            y = y + h * v.rhs_h(0, y, P)
            out.append(y.copy())
    return np.arange(n + 1) * h, np.array(out)


def euler_impl(y0, h, tf):
    n = int(round(tf / h))
    y = np.array(y0, float)
    out = [y.copy()]
    for _ in range(n):
        M = np.asfortranarray(np.eye(3) - h * v.jac_h(y, P))
        _, _, dy, info = lapack.dgesv(M, h * v.rhs_h(0, y, P))
        y = np.maximum(y + dy, 0.0)
        out.append(y.copy())
    return np.arange(n + 1) * h, np.array(out)


def rk4(y0, h, tf):
    n = int(round(tf / h))
    y = np.array(y0, float)
    f = lambda y_: v.rhs_h(0, y_, P)
    for _ in range(n):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# Fig. 6 : raideur
def fig6():
    ref = radau(hom(None), [0, 30], Y0, dense_output=True)
    tr = np.linspace(0, 30, 600)
    Vr = ref.sol(tr)[2]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for h, ls in zip([0.02, 0.05, 0.08, 0.1], [":", "--", "-.", "-"]):
        t, Y = euler_expl(Y0, h, 30)
        ax[0].semilogy(t, np.abs(Y[:, 2]) + 1e-300, ls=ls, color="black", label=f"$h={h}$ j")
    ax[0].semilogy(tr, Vr, color="gray", lw=3, label="référence")
    ax[0].set_ylim(1e-4, 1e8)
    ax[0].set_title("(a) Euler explicite")
    ax[0].set_ylabel(r"$|V|$ (virions/µL)")
    for h, ls in zip([0.1, 0.5, 1.0], ["-", "--", "-."]):
        t, Y = euler_impl(Y0, h, 30)
        ax[1].semilogy(t, np.maximum(Y[:, 2], 1e-300), ls=ls, color="black", label=f"$h={h}$ j")
    ax[1].semilogy(tr, Vr, color="gray", lw=3, label="référence")
    ax[1].set_ylim(1e-4, 1e5)
    ax[1].set_title("(b) Euler implicite (dgesv)")
    ax[1].set_ylabel(r"$V$ (virions/µL)")
    for a in ax:
        a.set_xlabel("Temps (jours)")
        a.legend(loc="lower right")
    save(fig, "fig6_raideur.png")


# Fig. 7 : ordre de convergence
def fig7():
    ref = radau(hom(None), [0, 10], Y0).y[:, -1]
    hs = np.array([0.04, 0.02, 0.01, 0.005, 0.0025])
    e = {"Euler explicite": [], "Euler implicite": [], "RK4": []}
    for h in hs:
        for name, y in [("Euler explicite", euler_expl(Y0, h, 10)[1][-1]),
                        ("Euler implicite", euler_impl(Y0, h, 10)[1][-1]),
                        ("RK4", rk4(Y0, h, 10))]:
            e[name].append(np.linalg.norm((y - ref) / np.maximum(np.abs(ref), 1e-12)))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (name, vals), ls, mk in zip(e.items(), ["-", "--", "-."], "osv"):
        ax.loglog(hs, vals, ls=ls, marker=mk, color="black", label=name)
    ax.set_xlabel(r"Pas $h$ (jours)")
    ax.set_ylabel(r"Erreur relative à $t=10$ j")
    ax.legend()
    save(fig, "fig7_convergence.png")


# ---------------------------------------------------------------------------
# Fig. 8 : primo-infection ; Fig. 9 : portrait de phase
# ---------------------------------------------------------------------------
def fig8():
    sol = radau(hom(None), [0, 400], Y0, dense_output=True)
    t = np.linspace(0, 400, 4000)
    Y = sol.sol(t)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    labs = [("CD4+ saines $T$ (cellules/µL)", "(a) $T(t)$", False),
            ("Cellules infectées $I$ (cellules/µL)", "(b) $I(t)$", True),
            ("Virus libres $V$ (virions/µL)", "(c) $V(t)$", True)]
    for k, (yl, ti, lg) in enumerate(labs):
        (ax[k].semilogy if lg else ax[k].plot)(t, np.maximum(Y[k], 1e-4), color="black")
        ax[k].axhline(EE[k], color="gray", ls="--")
        ax[k].set_xlabel("Temps (jours)")
        ax[k].set_ylabel(yl)
        ax[k].set_title(ti)
    ax[1].set_ylim(1e-4, 1e3)
    ax[2].set_ylim(1e-3, 1e5)
    save(fig, "fig8_infection_primaire.png")


def fig9():
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    starts = [(1000, 1e-4), (900, 5), (600, 50), (100, 5000), (300, 1)]
    for (T0, V0), ls in zip(starts, ["-", ":", "--", "-.", "-"]):
        y0 = [T0, 0.0, V0]
        s = radau(hom(None), [0, 600], y0, dense_output=True)
        t = np.linspace(0, 600, 6000)
        Y = s.sol(t)
        ax.semilogy(Y[0], np.maximum(Y[2], 1e-4), ls=ls, color="black")
    ax.plot(EE[0], EE[2], "o", color="black", ms=8, label="EE")
    ax.plot(DFE[0], 1e-4, "s", color="black", ms=8, label="DFE")
    ax.set_xlabel(r"$T$ (cellules/µL)")
    ax.set_ylabel(r"$V$ (virions/µL)")
    ax.set_ylim(1e-4, 1e4)
    ax.legend(loc="lower left")
    save(fig, "fig9_portrait_phase.png")


# ---------------------------------------------------------------------------
# Fig. 10 : R0 en fonction de eta ; Fig. 11 : traitement
# ---------------------------------------------------------------------------
def fig10():
    eta = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(100 * eta, [v.R0_h(P, e) for e in eta], color="black")
    ax.axhline(1, color="gray", ls="--")
    ax.axvline(75, color="gray", ls=":")
    ax.annotate(r"$\eta_c=75\,\%$", (75, 1), (45, 2.0), arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel(r"Efficacité du traitement $\eta$ (%)")
    ax.set_ylabel(r"$\mathcal{R}_0(\eta)$")
    save(fig, "fig10_r0_eta.png")


def fig11():
    t = np.linspace(0, 200, 4000)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for eta, ls in zip([0.0, 0.5, 0.7, 0.95], ["-", "--", "-.", ":"]):
        s = solve_ivp(hom(eta), [0, 200], EE, method="Radau", rtol=1e-10, atol=1e-20, dense_output=True)
        Y = s.sol(t)
        ax[0].semilogy(t, np.maximum(Y[2], 1e-4), ls=ls, color="black", label=rf"$\eta={int(100 * eta)}\,\%$")
        ax[1].plot(t, Y[0], ls=ls, color="black", label=rf"$\eta={int(100 * eta)}\,\%$")
    ax[0].axhline(0.05, color="gray", lw=0.8)
    ax[0].text(100, 0.07, "seuil d'indétectabilité (0,05)", fontsize=7, color="gray")
    ax[0].set_ylim(1e-4, 1e4)
    ax[0].set_title("(a) Charge virale")
    ax[0].set_ylabel(r"$V$ (virions/µL)")
    ax[1].set_title("(b) CD4+ saines")
    ax[1].set_ylabel(r"$T$ (cellules/µL)")
    for a in ax:
        a.set_xlabel("Jours depuis l'initiation du traitement")
        a.legend()
    save(fig, "fig11_traitement.png")


# ---------------------------------------------------------------------------
# Fig. 12 : bifurcation transcritique ; Fig. 13 : amplification et rebond
# ---------------------------------------------------------------------------
def fig12():
    R0s = np.linspace(0.2, 10, 200)
    Ts, Is, Vs, mDFE, mEE = [], [], [], [], []
    for R in R0s:
        Q = dict(P)
        Q["beta"] = P["beta"] * R / v.R0_h(P)
        mDFE.append(v.eigvals_lapack(v.jac_h(DFE, Q)).real.max())
        if R > 1:
            e = v.ee_h(Q)
            Ts.append(e[0]); Is.append(e[1]); Vs.append(e[2])
            mEE.append(v.eigvals_lapack(v.jac_h(e, Q)).real.max())
        else:
            Ts.append(1000.0); Is.append(0.0); Vs.append(0.0); mEE.append(np.nan)
    Ts, Is, Vs = map(np.array, (Ts, Is, Vs))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(R0s, Ts, color="black", label=r"$T^\star$")
    ax[0].plot(R0s, 10 * Is, "--", color="black", label=r"$10\,I^\star$")
    ax[0].plot(R0s, Vs / 10, ":", color="black", label=r"$V^\star/10$")
    ax[0].axvline(1, color="gray", lw=0.8)
    ax[0].set_title("(a) Bifurcation transcritique")
    ax[0].set_ylabel("Effectifs (cellules/µL, virions/µL)")
    ax[1].plot(R0s, mDFE, color="black", label="DFE")
    ax[1].plot(R0s, mEE, "--", color="black", label="EE")
    ax[1].axvline(1, color="gray", lw=0.8)
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_ylim(-1, 2.5)
    ax[1].set_title("(b) Échange de stabilité")
    ax[1].set_ylabel(r"$\max\,\mathrm{Re}\,\lambda$ (j$^{-1}$)")
    for a in ax:
        a.set_xlabel(r"$\mathcal{R}_0$")
        a.legend()
    save(fig, "fig12_bifurcation_intra.png")


def fig13():
    J = v.jac_h(EE, P)
    t1 = np.linspace(0, 600, 1200)
    rho = [v.sigma_max(expm(J * t)) for t in t1]
    slow = np.exp(v.eigvals_lapack(J).real.max() * t1)
    eta_fn = lambda t: 0.95 if t < 100 else 0.0
    s1 = solve_ivp(lambda t, y: v.rhs_h(t, y, P, 0.95), [0, 100], EE, method="Radau", rtol=1e-10, atol=1e-14)
    s2 = solve_ivp(lambda t, y: v.rhs_h(t, y, P, 0.0), [100, 600], s1.y[:, -1], method="Radau",
                   rtol=1e-10, atol=1e-14, dense_output=True)
    t2 = np.linspace(100, 600, 5000)
    Y2 = s2.sol(t2)
    ta = np.linspace(0, 100, 500)
    sa = solve_ivp(lambda t, y: v.rhs_h(t, y, P, 0.95), [0, 100], EE, method="Radau",
                   rtol=1e-10, atol=1e-14, dense_output=True)
    Ya = sa.sol(ta)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(t1, rho, color="black")
    ax[0].plot(t1, slow, "--", color="black", lw=0.8)
    ax[0].set_title("(a) Amplification transitoire")
    ax[0].set_ylabel(r"$\rho(t)=\|e^{Jt}\|_2$")
    ax[1].semilogy(np.r_[ta, t2], np.maximum(np.r_[Ya[2], Y2[2]], 1e-3), color="black")
    ax[1].axvline(100, color="gray", ls=":")
    ax[1].axhline(EE[2], color="gray", ls="--")
    ax[1].set_ylim(1e-3, 1e5)
    ax[1].set_title("(b) Rebond après arrêt (jour 100)")
    ax[1].set_ylabel(r"$V$ (virions/µL)")
    ax[2].plot(np.r_[ta, t2], np.r_[Ya[0], Y2[0]], color="black")
    ax[2].axvline(100, color="gray", ls=":")
    ax[2].set_title("(c) CD4+ pendant l'épisode")
    ax[2].set_ylabel(r"$T$ (cellules/µL)")
    for a in ax:
        a.set_xlabel("Temps (jours)")
    save(fig, "fig13_amplification.png")


# ---------------------------------------------------------------------------
# Fig. 14 : sensibilite
# ---------------------------------------------------------------------------
def fig14():
    names = ["s", "d", "beta", "delta", "p", "c"]
    labels = ["$s$", "$d$", r"$\beta$", r"$\delta$", "$p$", "$c$"]

    def idx(f, key, h=1e-6):
        Q = dict(P)
        Q[key] = P[key] * (1 + h)
        return (np.log(f(Q)) - np.log(f(P))) / np.log(1 + h)

    iR = [idx(lambda Q: v.R0_h(Q), k) for k in names]
    iV = [idx(lambda Q: v.ee_h(Q)[2], k) for k in names]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    y = np.arange(len(names))[::-1]
    ax[0].barh(y, iR, color="gray", edgecolor="black")
    ax[0].set_title(r"(a) Sensibilité de $\mathcal{R}_0$")
    ax[0].set_xlabel(r"$\Upsilon_q^{\mathcal{R}_0}$")
    ax[1].barh(y, iV, color="white", edgecolor="black", hatch="///")
    ax[1].set_title(r"(b) Sensibilité de $V^\star$")
    ax[1].set_xlabel(r"$\Upsilon_q^{V^\star}$")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels)
    save(fig, "fig14_sensibilite.png")


# ---------------------------------------------------------------------------
# Modele de population : fig. 16, 17, 18
# ---------------------------------------------------------------------------
PB = dict(v.PB)


def pop_rhs(Q, beta_fn=None, mu2_fn=None):
    def f(t, y):
        R = dict(Q)
        if beta_fn:
            R["beta"] = beta_fn(t)
        if mu2_fn:
            R["mu2"] = mu2_fn(t)
        d = v.f_B(y[:2], R)
        return [d[0], d[1], R["Lam"] - R["mu"] - R["mu2"] * y[1]]   # d(ln N)/dt
    return f


def fig16():
    s = solve_ivp(pop_rhs(PB), [0, 200], [1e-3, 0, 0], rtol=1e-9, atol=1e-12, max_step=0.5, dense_output=True)
    t = np.linspace(0, 200, 2000)
    i1, i2, lnN = s.sol(t)
    sus = 1 - i1 - i2
    prev = i1 + i2
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for y, ls, lab in [(sus, "-", "$s$"), (i1, "--", "$i_1$"), (i2, "-.", "$i_2$")]:
        ax[0].plot(t, 100 * y, ls=ls, color="black", label=lab)
    ax[0].set_title("(a) Proportions")
    ax[0].set_ylabel("% de la population")
    ax[0].legend()
    ax[1].plot(t, 100 * prev, color="black")
    ax[1].axhline(100 * prev[-1], color="gray", ls="--")
    ax[1].set_title("(b) Prévalence totale")
    ax[1].set_ylabel("Prévalence (%)")
    ax[2].plot(t, np.exp(lnN), color="black")
    ax[2].axhline(1, color="gray", ls=":")
    ax[2].set_title("(c) Effectif total")
    ax[2].set_ylabel(r"$N(t)/N(0)$")
    for a in ax:
        a.set_xlabel("Temps (années)")
    save(fig, "fig16_population_dynamique.png")


def fig17():
    base = v.R0_B(PB, 1.0)
    grid = np.linspace(6, 0.4, 150)
    guess = np.array([0.5, 0.1])
    prev, ret, per = {}, {}, {}
    for R in grid:
        Q = dict(PB)
        Q["beta"] = R / base
        if R > 1.0:
            sol, info, ier, _ = fsolve(lambda y: v.f_B(y, Q), guess, full_output=True)
            ok = ier == 1 and sol[0] > 1e-9 and sol[1] > 0
            if ok:
                guess = sol
                w = v.eigvals_lapack(v.J_B(sol, Q))
                prev[R] = 100 * sol.sum()
                ret[R] = -1.0 / w.real.max()
                im = np.abs(w.imag).max()
                per[R] = 2 * np.pi / im if im > 1e-9 else np.nan
                continue
        prev[R] = 0.0
        ret[R] = np.nan
        per[R] = np.nan
    Rs = np.array(sorted(prev))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(Rs, [prev[r] for r in Rs], color="black")
    ax[0].set_title("(a) Bifurcation")
    ax[0].set_ylabel("Prévalence endémique (%)")
    ax[1].semilogy(Rs, [ret[r] for r in Rs], color="black")
    ax[1].set_title("(b) Résilience")
    ax[1].set_ylabel("Temps de retour (années)")
    ax[2].plot(Rs, [per[r] for r in Rs], color="black")
    ax[2].set_title("(c) Oscillations")
    ax[2].set_ylabel("Période (années)")
    for a in ax:
        a.axvline(1, color="gray", lw=0.8)
        a.set_xlabel(r"$\mathcal{R}_0$")
    save(fig, "fig17_bifurcation_pop.png")


def fig18():
    T0 = 30.0
    scen = [("Sans intervention", 1.0, 0.5, "-"),
            (r"Prévention : $\beta\times0{,}5$ dès l'an 30", 0.5, 0.5, "--"),
            (r"Traitement : $\mu_2=0{,}1$ dès l'an 30", 1.0, 0.1, "-."),
            ("Prévention + traitement", 0.5, 0.1, ":")]
    t = np.linspace(0, 200, 2000)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for lab, bf, m2, ls in scen:
        f = pop_rhs(PB, lambda tt, bf=bf: PB["beta"] * (bf if tt >= T0 else 1.0),
                    lambda tt, m2=m2: m2 if tt >= T0 else PB["mu2"])
        s = solve_ivp(f, [0, 200], [1e-3, 0, 0], rtol=1e-9, atol=1e-12, max_step=0.5, dense_output=True)
        i1, i2, lnN = s.sol(t)
        ax[0].plot(t, 100 * (i1 + i2), ls=ls, color="black", label=lab)
        ax[1].plot(t, np.exp(lnN), ls=ls, color="black", label=lab)
    ax[0].set_title("(a) Prévalence")
    ax[0].set_ylabel("Prévalence (%)")
    ax[0].legend()
    ax[1].set_title("(b) Effectif total")
    ax[1].set_ylabel(r"$N(t)/N(0)$")
    for a in ax:
        a.set_xlabel("Temps (années)")
    save(fig, "fig18_interventions.png")


# ---------------------------------------------------------------------------
# Fig. 19 : benchmark BLAS / LAPACK
# ---------------------------------------------------------------------------
def best_time(fn, rep=3):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def fig19():
    rng = np.random.default_rng(2026)
    ns = [100, 200, 400, 800, 1200, 1600]
    tg, ts_ = [], []
    for n in ns:
        A = np.asfortranarray(rng.standard_normal((n, n)))
        B = np.asfortranarray(rng.standard_normal((n, n)))
        b = rng.standard_normal(n)
        tg.append(best_time(lambda: blas.dgemm(1.0, A, B)))
        ts_.append(best_time(lambda: lapack.dgesv(A.copy(order="F"), b)))
    ns = np.array(ns, float)
    tg, ts_ = np.array(tg), np.array(ts_)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].loglog(ns, 1e3 * tg, "o-", color="black", label="DGEMM")
    ax[0].loglog(ns, 1e3 * ts_, "s--", color="black", label="DGESV")
    ax[0].loglog(ns, 1e3 * tg[-1] * (ns / ns[-1]) ** 3, ":", color="gray", label=r"$\propto n^3$")
    ax[0].set_title("(a) Temps d'exécution")
    ax[0].set_ylabel("Temps (ms)")
    ax[1].semilogx(ns, 2 * ns ** 3 / tg / 1e9, "o-", color="black", label="DGEMM")
    ax[1].semilogx(ns, (2 / 3) * ns ** 3 / ts_ / 1e9, "s--", color="black", label="DGESV")
    ax[1].set_title("(b) Débit de calcul")
    ax[1].set_ylabel("Performance (GFLOPS)")
    for a in ax:
        a.set_xlabel(r"Taille $n$")
        a.legend()
    save(fig, "fig19_benchmark_perf.png")


if __name__ == "__main__":
    for f in (fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14,
              fig16, fig17, fig18, fig19):
        f()
    print("\nTerminé : 16 figures dans", OUT)
