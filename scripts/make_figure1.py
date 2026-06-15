import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# ---------------------------------------------------------
# Cosmological parameters (Planck-like)
# ---------------------------------------------------------
H0 = 67.0
Omega_m = 0.315
Omega_r = 9e-5
Omega_L = 1.0 - Omega_m - Omega_r

Omega_b = 0.049
Omega_gamma = 5e-5

# ---------------------------------------------------------
# Confinement parameters (Planck-safe fiducial model)
# ---------------------------------------------------------
alpha = 4.0
beta = 8.0
a_c = 5e-4
Delta_a = 4e-4
rho0_conf = 0.10   # peak fractional energy density

# ---------------------------------------------------------
# Background functions
# ---------------------------------------------------------
def H_LCDM(a):
    return H0 * np.sqrt(Omega_r/a**4 + Omega_m/a**3 + Omega_L)

def chi(a):
    x = 0.5 * (1.0 + np.tanh((a_c - a)/Delta_a))
    return np.clip(x, 1e-12, 1.0 - 1e-12)

def rho_conf_frac(a):
    x = chi(a)
    shape = x**alpha * (1.0 - x)**beta
    shape_max = np.max(shape)
    if shape_max == 0:
        return np.zeros_like(a)
    return rho0_conf * shape / shape_max

def H_with_conf(a):
    frac = rho_conf_frac(a)
    return H_LCDM(a) * np.sqrt(1.0 + frac)

# ---------------------------------------------------------
# Equation of state
# ---------------------------------------------------------
def dlnrho_dla(a):
    x = chi(a)
    t = np.tanh((a_c - a)/Delta_a)
    dx_da = -(1.0/Delta_a) * (1.0 - t**2)
    return (alpha/x - beta/(1.0 - x)) * dx_da * a

def w_conf(a):
    return -1.0 - (1.0/3.0) * dlnrho_dla(a)

# ---------------------------------------------------------
# Sound speed and sound horizon
# ---------------------------------------------------------
def R(a):
    return 3.0 * Omega_b/(4.0 * Omega_gamma) * a

def c_s(a):
    return 1.0 / np.sqrt(3.0 * (1.0 + R(a)))

def sound_horizon(H_func, a_min=1e-6, a_max=1e-3, n=2000):
    a_vals = np.logspace(np.log10(a_min), np.log10(a_max), n)
    integrand = c_s(a_vals) / (a_vals**2 * H_func(a_vals))
    return simpson(integrand, a_vals)

# ---------------------------------------------------------
# Compute key numbers
# ---------------------------------------------------------
a_rec = 1e-3

rs_LCDM = sound_horizon(H_LCDM, a_max=a_rec)
rs_conf = sound_horizon(H_with_conf, a_max=a_rec)
delta_rs = (rs_conf - rs_LCDM) / rs_LCDM * 100.0

frac_rec = rho_conf_frac(np.array([a_rec]))[0]

print("=== Planck-safe fiducial model ===")
print(f"alpha = {alpha}, beta = {beta}")
print(f"a_c = {a_c:.2e}, Delta_a = {Delta_a:.2e}, rho0_conf = {rho0_conf:.2f}")
print()
print(f"Sound horizon (LCDM):        {rs_LCDM:.4e}")
print(f"Sound horizon (with conf.):  {rs_conf:.4e}")
print(f"Fractional change Δr_s/r_s:  {delta_rs:.2f}%")
print()
print(f"Confinement fraction at recombination: {frac_rec:.3f}")

# ---------------------------------------------------------
# Prepare grids for plotting
# ---------------------------------------------------------
a_vals = np.logspace(-5, -2, 2000)

rho_vals = rho_conf_frac(a_vals)
w_vals = w_conf(a_vals)
H_ratio = H_with_conf(a_vals) / H_LCDM(a_vals)

# ---------------------------------------------------------
# Figure 1 — Three-panel composite (PRD style)
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300
})

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)

# Panel (a)
ax = axes[0]
ax.loglog(a_vals, rho_vals, color="black", lw=1.2)
ax.axvline(a_rec, color="gray", ls="--", lw=0.9)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$\rho_{\rm conf}/\rho_{\rm tot}$")
ax.set_title(r"(a) Confinement energy fraction")
ax.set_xlim(1e-5, 1e-2)

# Panel (b)
ax = axes[1]
ax.semilogx(a_vals, w_vals, color="black", lw=1.2)
ax.axhline(-1.0, color="gray", ls="--", lw=0.9)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$w_{\rm conf}(a)$")
ax.set_title(r"(b) Equation of state")
ax.set_xlim(1e-5, 1e-2)

# Panel (c)
ax = axes[2]
ax.semilogx(a_vals, H_ratio, color="black", lw=1.2)
ax.axhline(1.0, color="gray", ls="--", lw=0.9)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$H_{\rm conf}/H_{\Lambda{\rm CDM}}$")
ax.set_title(r"(c) Expansion enhancement")
ax.set_xlim(1e-5, 1e-2)

fig.tight_layout()
fig.savefig("figure1.pdf", bbox_inches="tight")
plt.close()

print("Saved figure1.pdf")
