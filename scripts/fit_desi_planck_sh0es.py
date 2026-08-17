"""
Real-data fit of the confinement-energy Hubble-tension mechanism against
Planck 2018 CMB distance priors, DESI DR2 BAO, and the SH0ES local H0
anchor.

This is the first, direct falsification test flagged in this project's
workspace notes: does letting the confinement amplitude (rho0_conf) and
epoch (a_c) float, fit jointly against real public data, actually relieve
the Hubble tension -- and if so, does it stay inside the "Planck-safe"
rho0_conf <= 0.1 bound this paper itself adopts?

Data sources (all real, public, cited -- not synthetic/toy):
  - DESI DR2 BAO consensus Gaussian likelihood (Abbott et al. 2025,
    "DESI DR2 Results II", arXiv:2503.14738), mean/cov files pulled from
    the community Cobaya likelihood repo
    https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2
    (desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt), copied verbatim into
    data/desi_dr2_bao_ALL_{mean,cov}.txt. 13 measurements (BGS D_V/r_d;
    LRG1/LRG2/LRG3+ELG1/ELG2/QSO D_M/r_d + D_H/r_d; Lya D_H/r_d + D_M/r_d),
    block-diagonal covariance.
  - Planck 2018 compressed CMB distance priors (R, l_A, omega_b h^2) and
    their correlation matrix, TT,TE,EE+lowE column, Table 1 of Chen,
    Huang & Wang 2019 (arXiv:1808.05724): R = 1.7502 +/- 0.0046,
    l_A = 301.471 +/- 0.089, omega_b h^2 = 0.02236 +/- 0.00015,
    corr(R,l_A)=0.46, corr(R,omega_b h^2)=-0.66, corr(l_A,omega_b h^2)=-0.33.
  - SH0ES local H0 = 73.04 +/- 1.04 km/s/Mpc (Riess et al. 2022, ApJL 934,
    L7, arXiv:2112.04510), used as a direct Gaussian prior on H0 -- this is
    NOT a from-scratch Pantheon+SH0ES calibrator-sample refit (that is a
    separate, larger undertaking); it uses the SH0ES team's own headline
    number, standard practice for this kind of mechanism test.

Background cosmology: flat LCDM + radiation (photons + 3.046 effective
neutrino species) + the paper's confinement term, using the same
tanh-triggered generalized-beta profile as scripts/make_figure1.py. Sound
horizon and comoving distances are computed by direct numerical
integration; the redshift of photon decoupling (z*) and the baryon drag
epoch (z_drag) use the standard Hu & Sugiyama (1996) and Eisenstein & Hu
(1998) fitting formulas respectively (verified against arXiv:1808.05724).
No CAMB/CLASS/Cobaya Boltzmann code is run here -- this is the same level
of approximation the rest of this paper's pipeline already uses, applied
to real data instead of an internal-consistency number.

One deliberate fix relative to make_figure1.py: rho_conf_frac here
normalizes by the analytic global peak of x^alpha(1-x)^beta at
x_peak=alpha/(alpha+beta), not by np.max() over whatever array happens to
be passed in. The latter is silently wrong for a single-point call (it
would just return rho0_conf back unchanged) and array-size-dependent in
general; the fit needs a normalization that doesn't depend on the
evaluation grid.
"""
import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
C_LIGHT = 299792.458  # km/s
N_EFF = 3.046
OMEGA_GAMMA_H2 = 2.469e-5  # T_CMB = 2.7255 K

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------
def load_desi_dr2():
    mean_path = os.path.join(DATA_DIR, "desi_dr2_bao_ALL_mean.txt")
    cov_path = os.path.join(DATA_DIR, "desi_dr2_bao_ALL_cov.txt")
    z_list, kind_list, val_list = [], [], []
    with open(mean_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            z_s, val_s, kind_s = line.split()
            z_list.append(float(z_s))
            val_list.append(float(val_s))
            kind_list.append(kind_s)
    cov = np.loadtxt(cov_path)
    z_arr = np.array(z_list)
    val_arr = np.array(val_list)
    assert cov.shape == (len(z_arr), len(z_arr)), "DESI cov/mean size mismatch"
    return z_arr, kind_list, val_arr, cov


PLANCK_R = 1.7502
PLANCK_LA = 301.471
PLANCK_OMBH2 = 0.02236
PLANCK_SIGMA = np.array([0.0046, 0.089, 0.00015])
PLANCK_CORR = np.array([
    [1.00, 0.46, -0.66],
    [0.46, 1.00, -0.33],
    [-0.66, -0.33, 1.00],
])
PLANCK_COV = PLANCK_CORR * np.outer(PLANCK_SIGMA, PLANCK_SIGMA)
PLANCK_COV_INV = np.linalg.inv(PLANCK_COV)
PLANCK_MEAN = np.array([PLANCK_R, PLANCK_LA, PLANCK_OMBH2])

SH0ES_H0 = 73.04
SH0ES_SIGMA = 1.04

# Planck 2018 TT,TE,EE+lowE+lensing+BAO constraint on the effective number
# of relativistic species, N_eff = 2.99 +/- 0.17 (95% CL Delta N_eff < 0.30),
# verified 2026-08-17 (not from memory). This is the real, independent BBN/
# early-radiation-era test flagged the same day: the a_c degeneracy found
# in the point-estimate and MCMC fits pushes rho0_conf/a_c into a regime
# where the confinement component is not a transient spike localized near
# recombination at all, but a large, near-constant extra-radiation-like
# fraction present continuously from long before BBN through matter-
# radiation equality -- exactly the regime this constraint is designed to
# catch, and a regime none of DESI/Planck-distance-priors/SH0ES probes.
PLANCK_NEFF = 2.99
PLANCK_NEFF_SIGMA = 0.17

# Calibration for the Eisenstein & Hu (1998) z_drag fitting formula --
# see the long comment in Cosmology.z_drag() below. z_drag_target /
# z_drag_raw(Planck fiducial ombh2=0.02237, omch2=0.1200).
Z_DRAG_CALIBRATION = 1059.94 / 1020.662522126387


# ---------------------------------------------------------------------
# Confinement model (same shape as scripts/make_figure1.py)
# ---------------------------------------------------------------------
CONF_ALPHA = 4.0
CONF_BETA = 8.0
CONF_DELTA_A = 4e-4


def _shape_max(alpha, beta):
    x_peak = alpha / (alpha + beta)
    return x_peak ** alpha * (1.0 - x_peak) ** beta


_SHAPE_MAX = _shape_max(CONF_ALPHA, CONF_BETA)


def chi_frac(a, a_c, delta_a):
    x = 0.5 * (1.0 + np.tanh((a_c - a) / delta_a))
    return np.clip(x, 1e-12, 1.0 - 1e-12)


def rho_conf_frac(a, rho0_conf, a_c, delta_a=CONF_DELTA_A,
                   alpha=CONF_ALPHA, beta=CONF_BETA):
    """NOTE: normalizes by _shape_max(alpha, beta) computed for *this
    call's* alpha/beta, not a fixed constant - shape_scan.py (2026-08-17)
    varies alpha/beta, and rho0_conf must keep meaning "peak fractional
    energy density" for whichever shape is actually in use, not silently
    assume the paper's fiducial (4, 8). This is a no-op for every prior
    call in this file, which only ever used the fiducial (4, 8) anyway."""
    x = chi_frac(a, a_c, delta_a)
    shape = x ** alpha * (1.0 - x) ** beta
    return rho0_conf * shape / _shape_max(alpha, beta)


# ---------------------------------------------------------------------
# Background cosmology
# ---------------------------------------------------------------------
class Cosmology:
    def __init__(self, H0, ombh2, omch2, rho0_conf=0.0, a_c=5e-4,
                 delta_a=CONF_DELTA_A, alpha=CONF_ALPHA, beta=CONF_BETA):
        self.H0 = H0
        self.h = H0 / 100.0
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.omega_m_h2 = ombh2 + omch2
        self.Omega_b = ombh2 / self.h ** 2
        self.Omega_cdm = omch2 / self.h ** 2
        self.Omega_m = self.Omega_b + self.Omega_cdm
        self.Omega_gamma = OMEGA_GAMMA_H2 / self.h ** 2
        self.Omega_r = self.Omega_gamma * (1.0 + 0.2271 * N_EFF)
        self.Omega_L = 1.0 - self.Omega_m - self.Omega_r
        self.rho0_conf = rho0_conf
        self.a_c = a_c
        self.delta_a = delta_a
        self.alpha = alpha
        self.beta = beta

    def E_lcdm(self, a):
        return np.sqrt(self.Omega_r / a ** 4 + self.Omega_m / a ** 3 + self.Omega_L)

    def conf_frac(self, a):
        if self.rho0_conf == 0.0:
            return np.zeros_like(a)
        return rho_conf_frac(a, self.rho0_conf, self.a_c, self.delta_a,
                              self.alpha, self.beta)

    def H(self, a):
        return self.H0 * self.E_lcdm(a) * np.sqrt(1.0 + self.conf_frac(a))

    def H_of_z(self, z):
        return self.H(1.0 / (1.0 + z))

    def R_baryon(self, a):
        return 3.0 * self.Omega_b / (4.0 * self.Omega_gamma) * a

    def c_s(self, a):
        return 1.0 / np.sqrt(3.0 * (1.0 + self.R_baryon(a)))

    def sound_horizon(self, a_end, a_min=1e-8, n=6000):
        a_vals = np.logspace(np.log10(a_min), np.log10(a_end), n)
        integrand = self.c_s(a_vals) / (a_vals ** 2 * self.H(a_vals))
        return C_LIGHT * np.trapezoid(integrand, a_vals)

    def z_star(self):
        ombh2 = self.ombh2
        omh2 = self.omega_m_h2
        g1 = 0.0738 * ombh2 ** -0.238 / (1.0 + 39.5 * ombh2 ** 0.763)
        g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
        return 1048.0 * (1.0 + 0.00124 * ombh2 ** -0.738) * (1.0 + g1 * omh2 ** g2)

    def z_drag(self):
        # Eisenstein & Hu (1998, ApJ 496, 605, Eq. 4) fitting formula for
        # the baryon-drag redshift, verified against the original paper
        # (arXiv:astro-ph/9709112). At Planck 2018 baseline parameters
        # (ombh2=0.02237, omch2=0.1200) this raw formula gives z_drag =
        # 1020.66, a ~3.85% low bias against Planck's own precise
        # CAMB-derived value z_drag = 1059.94 (and correspondingly r_drag
        # = 150.78 Mpc vs the standard quoted r_drag = 147.09 Mpc) -- a
        # known limitation of this 1998 fitting formula outside the
        # parameter range/recombination code it was calibrated against
        # (in contrast, the Hu & Sugiyama 1996 z_star formula used below
        # reproduces Planck's z_star and r_star to <0.1% with no
        # correction needed; see sanity_check()).
        #
        # Rather than silently use the biased raw value, apply a constant
        # multiplicative correction fixed at the Planck fiducial point so
        # the pipeline's r_drag matches the real, precisely-known value
        # there. This assumes the fitting formula's *response* to
        # parameter changes away from the fiducial point remains
        # accurate even though its absolute normalization was not --
        # untested here, and worth flagging as a residual approximation
        # if this fit is pushed toward parameter values far from Planck's
        # own best fit.
        ombh2 = self.ombh2
        omh2 = self.omega_m_h2
        b1 = 0.313 * omh2 ** -0.419 * (1.0 + 0.607 * omh2 ** 0.674)
        b2 = 0.238 * omh2 ** 0.223
        z_drag_raw = (1291.0 * omh2 ** 0.251 / (1.0 + 0.659 * omh2 ** 0.828)
                      * (1.0 + b1 * ombh2 ** b2))
        return z_drag_raw * Z_DRAG_CALIBRATION

    def comoving_distance_interp(self, z_max, n=3000):
        z_grid = np.linspace(0.0, z_max * 1.001, n)
        a_grid = 1.0 / (1.0 + z_grid)
        inv_H = 1.0 / self.H(a_grid)
        D_C = C_LIGHT * cumulative_trapezoid(inv_H, z_grid, initial=0.0)
        return interp1d(z_grid, D_C, kind="cubic")

    def r_star(self):
        return self.sound_horizon(1.0 / (1.0 + self.z_star()))

    def r_drag(self):
        return self.sound_horizon(1.0 / (1.0 + self.z_drag()))


# ---------------------------------------------------------------------
# chi^2 pieces
# ---------------------------------------------------------------------
DESI_Z, DESI_KIND, DESI_VAL, DESI_COV = load_desi_dr2()
DESI_COV_INV = np.linalg.inv(DESI_COV)
DESI_ZMAX = DESI_Z.max()


def predict_desi(cosmo):
    D_C_interp = cosmo.comoving_distance_interp(DESI_ZMAX)
    r_d = cosmo.r_drag()
    preds = np.empty_like(DESI_VAL)
    for i, (z, kind) in enumerate(zip(DESI_Z, DESI_KIND)):
        D_M = D_C_interp(z)
        D_H = C_LIGHT / cosmo.H_of_z(z)
        if kind == "DM_over_rs":
            preds[i] = D_M / r_d
        elif kind == "DH_over_rs":
            preds[i] = D_H / r_d
        elif kind == "DV_over_rs":
            D_V = (z * D_M ** 2 * D_H) ** (1.0 / 3.0)
            preds[i] = D_V / r_d
        else:
            raise ValueError(kind)
    return preds


def chi2_desi(cosmo):
    resid = predict_desi(cosmo) - DESI_VAL
    return resid @ DESI_COV_INV @ resid


def chi2_planck(cosmo):
    R = cosmo.r_star()
    D_C_interp = cosmo.comoving_distance_interp(cosmo.z_star())
    D_M_star = D_C_interp(cosmo.z_star())
    R_shift = D_M_star * np.sqrt(cosmo.Omega_m) * cosmo.H0 / C_LIGHT
    l_A = np.pi * D_M_star / R
    resid = np.array([R_shift, l_A, cosmo.ombh2]) - PLANCK_MEAN
    return resid @ PLANCK_COV_INV @ resid


def chi2_sh0es(cosmo):
    return ((cosmo.H0 - SH0ES_H0) / SH0ES_SIGMA) ** 2


A_BBN = 1e-9  # T ~ 1 MeV; the plateau value is insensitive to the exact
              # choice as long as a_BBN << a_c and a_BBN << delta_a, true
              # for the entire fitted a_c range (1e-7 to 5e-3 >> 1e-9)


def bbn_neff_effective(cosmo):
    """Effective N_eff implied by the confinement component's energy-
    density fraction during/before BBN (radiation totally dominates the
    background there, so `conf_frac` is -- to extremely good approximation
    at a=1e-9 -- a fraction of the radiation density specifically, not of
    matter+radiation+Lambda as it is by the time of recombination).
    N_eff_eff solves Omega_gamma*(1+0.2271*N_eff_eff)
    = Omega_gamma*(1+0.2271*N_EFF)*(1+f_bbn), i.e. the same total-radiation
    normalization convention already used for Omega_r in Cosmology.__init__.
    """
    f_bbn = cosmo.conf_frac(np.array([A_BBN]))[0]
    base = 1.0 + 0.2271 * N_EFF
    return ((base * (1.0 + f_bbn)) - 1.0) / 0.2271


def chi2_bbn(cosmo):
    neff_eff = bbn_neff_effective(cosmo)
    return ((neff_eff - PLANCK_NEFF) / PLANCK_NEFF_SIGMA) ** 2


def chi2_total(params, use_confinement):
    if use_confinement:
        H0, ombh2, omch2, rho0_conf, log10_ac = params
        a_c = 10 ** log10_ac
    else:
        H0, ombh2, omch2 = params
        rho0_conf, a_c = 0.0, 5e-4
    if not (60.0 < H0 < 82.0 and 0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16):
        return 1e12
    if use_confinement and not (0.0 <= rho0_conf < 0.95 and 1e-7 <= a_c <= 5e-3):
        return 1e12
    cosmo = Cosmology(H0, ombh2, omch2, rho0_conf, a_c)
    if cosmo.Omega_L <= 0.0:
        return 1e12
    return (chi2_desi(cosmo) + chi2_planck(cosmo) + chi2_sh0es(cosmo)
            + chi2_bbn(cosmo))


# ---------------------------------------------------------------------
# Sanity check: does the pipeline recover Planck's own numbers from
# Planck's own best-fit parameters?
# ---------------------------------------------------------------------
def sanity_check():
    print("=== Sanity check: recover Planck 2018 R, l_A from Planck's own ===")
    print("    best-fit parameters (H0=67.36, ombh2=0.02237, omch2=0.1200) ===")
    cosmo = Cosmology(67.36, 0.02237, 0.1200, rho0_conf=0.0)
    z_star = cosmo.z_star()
    z_drag = cosmo.z_drag()
    r_star = cosmo.r_star()
    D_C_interp = cosmo.comoving_distance_interp(z_star)
    D_M_star = D_C_interp(z_star)
    R_shift = D_M_star * np.sqrt(cosmo.Omega_m) * cosmo.H0 / C_LIGHT
    l_A = np.pi * D_M_star / r_star
    print(f"  z_star computed = {z_star:.2f}  (Planck 2018 quotes 1089.80, uncorrected)")
    print(f"  z_drag computed = {z_drag:.2f}  (Planck 2018 quotes 1059.94, "
          f"AFTER the {Z_DRAG_CALIBRATION:.4f}x calibration below)")
    print(f"  r_star computed = {r_star:.2f} Mpc (Planck 2018 quotes 144.39 Mpc, uncorrected)")
    print(f"  R computed  = {R_shift:.4f}   (Planck: {PLANCK_R} +/- {PLANCK_SIGMA[0]})")
    print(f"  l_A computed = {l_A:.3f}   (Planck: {PLANCK_LA} +/- {PLANCK_SIGMA[1]})")
    print(f"  r_drag computed = {cosmo.r_drag():.2f} Mpc "
          f"(Planck 2018 / DESI fiducial: 147.09 Mpc, AFTER calibration)")
    print()


def run_fit(use_confinement, n_restarts=6, seed=0):
    if use_confinement:
        labels = ["H0", "ombh2", "omch2", "rho0_conf", "log10(a_c)"]
        x0_base = np.array([69.0, 0.02236, 0.118, 0.05, np.log10(5e-4)])
    else:
        labels = ["H0", "ombh2", "omch2"]
        x0_base = np.array([67.4, 0.02236, 0.120])

    rng = np.random.default_rng(seed)
    best_res = None
    for i in range(n_restarts):
        if i == 0:
            x0 = x0_base
        else:
            jitter = rng.normal(scale=[2.0, 0.0005, 0.01] +
                                 ([0.05, 1.0] if use_confinement else []))
            x0 = x0_base + jitter
            if use_confinement:
                x0[3] = abs(x0[3])
        res = minimize(chi2_total, x0, args=(use_confinement,), method="Nelder-Mead",
                        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 30000,
                                 "maxfev": 30000})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    return best_res, labels


def report(name, res, labels, use_confinement):
    print(f"=== {name} ===")
    for lab, val in zip(labels, res.x):
        print(f"  {lab:12s} = {val:.5f}")
    if use_confinement:
        H0, ombh2, omch2, rho0_conf, log10_ac = res.x
        a_c = 10 ** log10_ac
    else:
        H0, ombh2, omch2 = res.x
        rho0_conf, a_c = 0.0, 5e-4
    cosmo = Cosmology(H0, ombh2, omch2, rho0_conf, a_c)
    c2_desi = chi2_desi(cosmo)
    c2_planck = chi2_planck(cosmo)
    c2_sh0es = chi2_sh0es(cosmo)
    c2_bbn = chi2_bbn(cosmo)
    n_free = len(labels)
    n_data = len(DESI_VAL) + 3 + 1 + 1
    dof = n_data - n_free
    print(f"  chi2_DESI   = {c2_desi:.2f}  (13 pts)")
    print(f"  chi2_Planck = {c2_planck:.2f}  (3 pts)")
    print(f"  chi2_SH0ES  = {c2_sh0es:.2f}  (1 pt, H0={H0:.2f} vs {SH0ES_H0}+/-{SH0ES_SIGMA})")
    neff_eff = bbn_neff_effective(cosmo)
    print(f"  chi2_BBN    = {c2_bbn:.2f}  (1 pt, N_eff_eff={neff_eff:.3f} vs "
          f"{PLANCK_NEFF}+/-{PLANCK_NEFF_SIGMA})")
    print(f"  chi2_total  = {res.fun:.2f}  / dof={dof} -> chi2/dof = {res.fun/dof:.3f}")
    if use_confinement:
        print(f"  rho0_conf = {rho0_conf:.4f}  (\"Planck-safe\" bound in this paper: <= 0.1)")
        print(f"  a_c       = {a_c:.4e}  (paper fiducial: 5e-4)")
        planck_safe = rho0_conf <= 0.1
        print(f"  Planck-safe bound respected at best fit: {planck_safe}")
    print()
    return cosmo, res.fun


if __name__ == "__main__":
    print(f"Loaded DESI DR2 BAO: {len(DESI_VAL)} data points, z in "
          f"[{DESI_Z.min():.3f}, {DESI_Z.max():.3f}]\n")

    sanity_check()

    res_lcdm, lab_lcdm = run_fit(use_confinement=False)
    cosmo_lcdm, chi2_lcdm = report("Flat LCDM (no confinement), DESI+Planck+SH0ES joint fit",
                                    res_lcdm, lab_lcdm, use_confinement=False)

    res_conf, lab_conf = run_fit(use_confinement=True)
    cosmo_conf, chi2_conf = report("LCDM + confinement (rho0_conf, a_c free), "
                                    "alpha=4,beta=8,Delta_a=4e-4 fixed at paper fiducial",
                                    res_conf, lab_conf, use_confinement=True)

    print("=== Comparison ===")
    print(f"  Delta chi2 (confinement - LCDM) = {chi2_conf - chi2_lcdm:+.2f}")
    print(f"  H0(LCDM best fit)        = {res_lcdm.x[0]:.2f} km/s/Mpc")
    print(f"  H0(confinement best fit) = {res_conf.x[0]:.2f} km/s/Mpc")
    print(f"  SH0ES anchor             = {SH0ES_H0} +/- {SH0ES_SIGMA} km/s/Mpc")
