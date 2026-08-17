"""
Regenerates the paper's figures using the real-data best-fit configuration
(scripts/fit_desi_planck_sh0es.py, LCDM+confinement, alpha=4, beta=8,
Delta_a=4e-4 fixed at the paper's own fiducial shape, rho0_conf and a_c
fit jointly against DESI DR2 BAO + Planck compressed priors + SH0ES + BBN),
replacing the old figure1.pdf, which was built from the paper's original
purely-internal-consistency parameters (rho0=0.10, a_c=5e-4) that were
never checked against real data.

figure1.pdf: three-panel composite (confinement fraction, equation of
state, expansion enhancement) at the real-data best fit.

figure2.pdf: the Delta_a trade-off scan from scripts/shape_scan.py,
showing that the "resolves the tension" region and the "causal and
energy-bound-safe" region never overlap across the scanned range.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS_DIR = os.path.join(ROOT, "results")

# ---------------------------------------------------------------------
# Figure 1: three-panel composite at the real-data best fit
# ---------------------------------------------------------------------
H0 = 73.09737
Omega_m = 0.02244 / (H0 / 100) ** 2 + 0.13923 / (H0 / 100) ** 2
Omega_r = 9e-5
Omega_L = 1.0 - Omega_m - Omega_r

Omega_b = 0.02244 / (H0 / 100) ** 2
Omega_gamma = 5e-5

alpha, beta = 4.0, 8.0
a_c = 3.4039e-4
Delta_a = 4e-4
rho0_conf = 0.2029


def H_LCDM(a):
    return H0 * np.sqrt(Omega_r / a ** 4 + Omega_m / a ** 3 + Omega_L)


def chi(a):
    x = 0.5 * (1.0 + np.tanh((a_c - a) / Delta_a))
    return np.clip(x, 1e-12, 1.0 - 1e-12)


def shape_max(alpha, beta):
    x_peak = alpha / (alpha + beta)
    return x_peak ** alpha * (1.0 - x_peak) ** beta


_SHAPE_MAX = shape_max(alpha, beta)


def rho_conf_frac(a):
    x = chi(a)
    shape = x ** alpha * (1.0 - x) ** beta
    return rho0_conf * shape / _SHAPE_MAX


def H_with_conf(a):
    frac = rho_conf_frac(a)
    return H_LCDM(a) * np.sqrt(1.0 + frac)


def dlnrho_dla(a):
    x = chi(a)
    t = np.tanh((a_c - a) / Delta_a)
    dx_da = -(1.0 / Delta_a) * (1.0 - t ** 2)
    return (alpha / x - beta / (1.0 - x)) * dx_da * a


def w_conf(a):
    return -1.0 - (1.0 / 3.0) * dlnrho_dla(a)


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

a_vals = np.logspace(-5, -2, 2000)
rho_vals = rho_conf_frac(a_vals)
w_vals = w_conf(a_vals)
H_ratio = H_with_conf(a_vals) / H_LCDM(a_vals)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)

ax = axes[0]
ax.loglog(a_vals, rho_vals, color="black", lw=1.2)
ax.axvline(a_c, color="gray", ls="--", lw=0.9)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$\rho_{\rm conf}/\rho_{\rm tot}$")
ax.set_title(r"(a) Confinement energy fraction")
ax.set_xlim(1e-5, 1e-2)
ax.set_ylim(1e-6, 2.0)

ax = axes[1]
ax.semilogx(a_vals, w_vals, color="black", lw=1.2)
ax.axhline(1.0, color="crimson", ls=":", lw=1.0)
ax.axhline(-1.0, color="gray", ls="--", lw=0.9)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$w_{\rm conf}(a)$")
ax.set_title(r"(b) Equation of state")
ax.set_xlim(1e-5, 1e-2)

ax = axes[2]
ax.semilogx(a_vals, H_ratio, color="black", lw=1.2)
ax.axhline(1.0, color="gray", ls="--", lw=0.9)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$H_{\rm conf}/H_{\Lambda{\rm CDM}}$")
ax.set_title(r"(c) Expansion enhancement")
ax.set_xlim(1e-5, 1e-2)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figure1.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figure1.pdf (real-data best-fit configuration)")

# ---------------------------------------------------------------------
# Figure 2: Delta_a trade-off scan (scripts/shape_scan.py results)
# ---------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "shape_scan.json")) as f:
    rows = json.load(f)

da = np.array([r["delta_a"] for r in rows])
H0s = np.array([r["H0"] for r in rows])
rho0s = np.array([r["rho0_conf"] for r in rows])
wpeaks = np.array([r["w_peak"] for r in rows])

fig, ax1 = plt.subplots(figsize=(6, 4.2))
ax1.semilogx(da, H0s, "o-", color="#2a78d6", label=r"$H_0$ (left axis)")
ax1.axhline(73.04, color="#2a78d6", ls=":", lw=1.0)
ax1.text(da[0], 73.3, "SH0ES", color="#2a78d6", fontsize=8)
ax1.set_xlabel(r"$\Delta a$")
ax1.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", color="#2a78d6")
ax1.tick_params(axis="y", labelcolor="#2a78d6")

ax2 = ax1.twinx()
ax2.semilogx(da, wpeaks, "s-", color="#e34948", label=r"$w_{\rm conf}$ peak (right axis)")
ax2.axhline(1.0, color="#e34948", ls=":", lw=1.0)
ax2.set_ylabel(r"peak $w_{\rm conf}$ (causal bound at $1$)", color="#e34948")
ax2.tick_params(axis="y", labelcolor="#e34948")
ax2.set_yscale("log")

ax1.set_title("Tension relief versus causality across the confinement\n"
              r"profile's transition width $\Delta a$ (real-data joint fit"
              " at each point)")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figure2.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figure2.pdf (Delta_a trade-off scan)")
