"""
Does the general "transient pre-recombination confinement-release" idea
survive if the pulse *shape* is allowed to be different from the paper's
specific alpha=4, beta=8, Delta_a=4e-4 fiducial choice - or is the
combination of failures found against that one fiducial point (causality
violation, Planck/BBN energy-fraction bound violations) a structural
property of the whole x^alpha(1-x)^beta confinement-release family, not
just a bad parameter choice?

This does NOT touch main.tex or revisit the paper's own stated numbers -
it is a separate question about the general mechanism class, asked by
scanning the one shape parameter (Delta_a, the transition width) that
most directly trades off two of the three failures against each other:
a narrower pulse (small Delta_a) is more "Planck-safe"/BBN-safe (stays
tightly localized near recombination) but has a faster fractional
turn-on/turn-off, which is exactly what drives the equation-of-state
causality violation (w = -1 - (1/3) dlnrho/dlna diverges near the
profile's power-law edges when the transition happens fast in ln(a)).
A broader pulse relaxes causality but risks bleeding into the low-z BAO
data range or the BBN epoch.

For each Delta_a on a log grid, alpha=4 and beta=8 are held fixed (same
profile *family* as the paper - only the timescale changes), and
(H0, ombh2, omch2, rho0_conf, a_c) are refit against the same real
DESI+Planck+SH0ES+BBN data used in fit_desi_planck_sh0es.py. Reports,
at each Delta_a's best fit: fit quality, rho0_conf vs. both energy bounds,
and the peak/min of w_conf vs. the causal bound.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_desi_planck_sh0es import (  # noqa: E402
    Cosmology, chi2_desi, chi2_planck, chi2_sh0es, chi2_bbn,
    CONF_ALPHA, CONF_BETA, PLANCK_NEFF, PLANCK_NEFF_SIGMA,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def chi2_total_shape(params, delta_a):
    H0, ombh2, omch2, rho0_conf, log10_ac = params
    a_c = 10 ** log10_ac
    if not (60.0 < H0 < 82.0 and 0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16):
        return 1e12
    if not (0.0 <= rho0_conf < 0.95 and 1e-9 <= a_c <= 5e-2):
        return 1e12
    cosmo = Cosmology(H0, ombh2, omch2, rho0_conf, a_c, delta_a=delta_a)
    if cosmo.Omega_L <= 0.0:
        return 1e12
    return (chi2_desi(cosmo) + chi2_planck(cosmo) + chi2_sh0es(cosmo)
            + chi2_bbn(cosmo))


def run_fit_shape(delta_a, n_restarts=6, seed=0):
    x0_base = np.array([70.0, 0.02236, 0.120, 0.05, np.log10(delta_a * 1.2)])
    rng = np.random.default_rng(seed)
    best_res = None
    for i in range(n_restarts):
        if i == 0:
            x0 = x0_base
        else:
            jitter = rng.normal(scale=[2.0, 0.0005, 0.01, 0.05, 1.0])
            x0 = x0_base + jitter
            x0[3] = abs(x0[3])
        res = minimize(chi2_total_shape, x0, args=(delta_a,), method="Nelder-Mead",
                        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 20000,
                                 "maxfev": 20000})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    return best_res


def w_conf_extrema(a_c, delta_a, alpha=CONF_ALPHA, beta=CONF_BETA):
    a_scan = np.logspace(-9, -1, 300000)
    x = 0.5 * (1.0 + np.tanh((a_c - a_scan) / delta_a))
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    t = np.tanh((a_c - a_scan) / delta_a)
    dx_da = -(1.0 / delta_a) * (1.0 - t ** 2)
    dlnrho_dla = (alpha / x - beta / (1.0 - x)) * dx_da * a_scan
    w = -1.0 - (1.0 / 3.0) * dlnrho_dla
    return float(np.nanmax(w)), float(np.nanmin(w))


def main():
    delta_a_grid = np.logspace(-4, -0.3, 14)  # 4e-4 (paper fiducial) up to ~0.5
    print(f"{'Delta_a':>10s} {'H0':>7s} {'rho0':>7s} {'a_c':>10s} "
          f"{'chi2/dof':>9s} {'w_peak':>9s} {'w_min':>8s} "
          f"{'<=0.1':>6s} {'<=0.087':>8s} {'causal':>7s}")
    rows = []
    t0 = time.time()
    for delta_a in delta_a_grid:
        res = run_fit_shape(delta_a)
        H0, ombh2, omch2, rho0_conf, log10_ac = (float(v) for v in res.x)
        a_c = 10 ** log10_ac
        cosmo = Cosmology(H0, ombh2, omch2, rho0_conf, a_c, delta_a=delta_a)
        c2_bbn = float(chi2_bbn(cosmo))
        n_data = 13 + 3 + 1 + 1
        dof = n_data - 5
        w_peak, w_min = w_conf_extrema(a_c, delta_a)
        causal = bool(w_peak <= 1.0 and w_min >= -1.0)
        safe_010 = bool(rho0_conf <= 0.10)
        safe_0087 = bool(rho0_conf <= 0.087)
        rows.append(dict(delta_a=float(delta_a), H0=H0, rho0_conf=rho0_conf,
                          a_c=a_c, chi2_dof=float(res.fun) / dof, chi2_bbn=c2_bbn,
                          w_peak=w_peak, w_min=w_min, causal=causal,
                          safe_010=safe_010, safe_0087=safe_0087))
        print(f"{delta_a:10.2e} {H0:7.2f} {rho0_conf:7.3f} {a_c:10.2e} "
              f"{res.fun/dof:9.3f} {w_peak:9.2f} {w_min:8.2f} "
              f"{'Y' if safe_010 else 'n':>6s} {'Y' if safe_0087 else 'n':>8s} "
              f"{'Y' if causal else 'n':>7s}")
    print(f"\nTotal scan time: {time.time()-t0:.1f}s")

    any_all_pass = any(r["causal"] and r["safe_010"] and r["H0"] > 71.5 for r in rows)
    print(f"\nAny Delta_a with causal=Y, rho0<=0.10=Y, and H0>71.5 (tension "
          f"meaningfully relieved)? {any_all_pass}")

    with open(os.path.join(RESULTS_DIR, "shape_scan.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"Saved to {RESULTS_DIR}/shape_scan.json")


if __name__ == "__main__":
    main()
