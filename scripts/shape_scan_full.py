"""
Extends shape_scan.py's question (does the *general* confinement-release
mechanism survive, not just the paper's specific fiducial parameters?) from
a 1D scan over just the transition width Delta_a to the full shape family:
alpha, beta, Delta_a, a_c, rho0_conf, and the three cosmological parameters,
8 free parameters total, searched jointly rather than one slice at a time.

Two runs:
  (A) UNCONSTRAINED - best joint fit to real data (DESI+Planck+SH0ES+BBN)
      with alpha, beta, Delta_a all free, ignoring causality. Sanity check:
      does letting the *shape itself* vary (not just its timescale) find a
      meaningfully different/better resolution than the paper's fiducial
      (4, 8)? If the optimizer just re-finds something close to (4, 8)
      anyway, that's independent evidence the fiducial shape was already
      close to data-optimal within this profile family.
  (B) CAUSALLY-CONSTRAINED - same 8 parameters, but with a heavy penalty
      added whenever w_conf(a) exits [-1, 1] anywhere in [1e-9, 1e-1].
      This asks the sharper question directly: across the *entire*
      x^alpha(1-x)^beta / tanh-transition family (not just Delta_a at fixed
      alpha=4, beta=8), is there any point that is simultaneously causal
      and actually relieves the Hubble tension? If the best this run can do
      is rho0_conf ~ 0 / H0 ~ 69 (the LCDM-only answer), that confirms
      shape_scan.py's finding generalizes to the full family, not just the
      one slice already checked.

alpha, beta bounds: [0.3, 30], searched in log10 space (large dynamic range,
covers everything from a nearly-symmetric mild bump to a sharply peaked
spike in either direction, well beyond the paper's own (4, 8)).
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_desi_planck_sh0es import (  # noqa: E402
    Cosmology, chi2_desi, chi2_planck, chi2_sh0es, chi2_bbn,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = ["H0", "ombh2", "omch2", "rho0_conf", "log10_ac", "log10_delta_a",
          "log10_alpha", "log10_beta"]


def unpack(params):
    H0, ombh2, omch2, rho0_conf, log10_ac, log10_delta_a, log10_alpha, log10_beta = params
    a_c = 10 ** log10_ac
    delta_a = 10 ** log10_delta_a
    alpha = 10 ** log10_alpha
    beta = 10 ** log10_beta
    return H0, ombh2, omch2, rho0_conf, a_c, delta_a, alpha, beta


def in_bounds(params):
    H0, ombh2, omch2, rho0_conf, log10_ac, log10_delta_a, log10_alpha, log10_beta = params
    return (60.0 < H0 < 82.0 and 0.019 < ombh2 < 0.026 and 0.08 < omch2 < 0.16
            and 0.0 <= rho0_conf < 0.95
            and -9.0 <= log10_ac <= -1.3  # a_c in [1e-9, 5e-2]
            and -5.0 <= log10_delta_a <= -0.3  # delta_a in [1e-5, 0.5]
            and np.log10(0.3) <= log10_alpha <= np.log10(30.0)
            and np.log10(0.3) <= log10_beta <= np.log10(30.0))


def w_conf_extrema(a_c, delta_a, alpha, beta):
    a_scan = np.logspace(-9, -1, 200000)
    x = 0.5 * (1.0 + np.tanh((a_c - a_scan) / delta_a))
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    t = np.tanh((a_c - a_scan) / delta_a)
    dx_da = -(1.0 / delta_a) * (1.0 - t ** 2)
    dlnrho_dla = (alpha / x - beta / (1.0 - x)) * dx_da * a_scan
    w = -1.0 - (1.0 / 3.0) * dlnrho_dla
    return float(np.nanmax(w)), float(np.nanmin(w))


def data_chi2(params):
    H0, ombh2, omch2, rho0_conf, a_c, delta_a, alpha, beta = unpack(params)
    cosmo = Cosmology(H0, ombh2, omch2, rho0_conf, a_c, delta_a, alpha, beta)
    if cosmo.Omega_L <= 0.0:
        return 1e12
    return chi2_desi(cosmo) + chi2_planck(cosmo) + chi2_sh0es(cosmo) + chi2_bbn(cosmo)


def objective_unconstrained(params):
    if not in_bounds(params):
        return 1e12
    return data_chi2(params)


CAUSAL_PENALTY_SCALE = 2e4


def objective_causal(params):
    if not in_bounds(params):
        return 1e12
    H0, ombh2, omch2, rho0_conf, a_c, delta_a, alpha, beta = unpack(params)
    w_peak, w_min = w_conf_extrema(a_c, delta_a, alpha, beta)
    viol = max(0.0, w_peak - 1.0) ** 2 + max(0.0, -1.0 - w_min) ** 2
    return data_chi2(params) + CAUSAL_PENALTY_SCALE * viol


def run_search(objective, n_restarts, seed):
    x0_base = np.array([70.0, 0.02236, 0.120, 0.05,
                         np.log10(5e-4), np.log10(4e-4),
                         np.log10(4.0), np.log10(8.0)])
    rng = np.random.default_rng(seed)
    best_res = None
    for i in range(n_restarts):
        if i == 0:
            x0 = x0_base
        else:
            x0 = np.array([
                rng.uniform(65.0, 78.0),
                rng.normal(0.02236, 0.0005),
                rng.normal(0.120, 0.01),
                rng.uniform(0.0, 0.6),
                rng.uniform(-8.0, -2.0),
                rng.uniform(-4.5, -0.5),
                rng.uniform(np.log10(0.4), np.log10(20.0)),
                rng.uniform(np.log10(0.4), np.log10(20.0)),
            ])
        res = minimize(objective, x0, method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 40000,
                                 "maxfev": 40000})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    return best_res


def report(name, res):
    H0, ombh2, omch2, rho0_conf, a_c, delta_a, alpha, beta = unpack(res.x)
    cosmo = Cosmology(H0, ombh2, omch2, rho0_conf, a_c, delta_a, alpha, beta)
    c2 = data_chi2(res.x)
    w_peak, w_min = w_conf_extrema(a_c, delta_a, alpha, beta)
    dof = 18 - 8
    print(f"=== {name} ===")
    print(f"  H0={H0:.2f}  ombh2={ombh2:.5f}  omch2={omch2:.5f}")
    print(f"  rho0_conf={rho0_conf:.4f}  a_c={a_c:.3e}  delta_a={delta_a:.3e}")
    print(f"  alpha={alpha:.3f}  beta={beta:.3f}")
    print(f"  data_chi2={c2:.2f} / dof={dof} -> {c2/dof:.3f}")
    print(f"  w_conf peak={w_peak:.2f}  min={w_min:.2f}  causal={w_peak<=1.0 and w_min>=-1.0}")
    print(f"  rho0_conf <= 0.10: {rho0_conf<=0.10}   <= 0.087: {rho0_conf<=0.087}")
    print()
    return dict(H0=H0, ombh2=ombh2, omch2=omch2, rho0_conf=rho0_conf, a_c=a_c,
                delta_a=delta_a, alpha=alpha, beta=beta, data_chi2=c2,
                w_peak=w_peak, w_min=w_min)


def main():
    t0 = time.time()
    print("Searching UNCONSTRAINED (alpha, beta, Delta_a, a_c, rho0_conf all free,"
          " no causality penalty) ...")
    res_u = run_search(objective_unconstrained, n_restarts=20, seed=1)
    row_u = report("Unconstrained best fit (real data only)", res_u)
    print(f"  ({time.time()-t0:.1f}s so far)\n")

    print("Searching CAUSALLY-CONSTRAINED (heavy penalty for |w+1|>0 violations) ...")
    res_c = run_search(objective_causal, n_restarts=20, seed=2)
    row_c = report("Causally-constrained best fit", res_c)
    print(f"  (total {time.time()-t0:.1f}s)\n")

    import json
    with open(os.path.join(RESULTS_DIR, "shape_scan_full.json"), "w") as f:
        json.dump({"unconstrained": row_u, "causal": row_c}, f, indent=1)
    print(f"Saved to {RESULTS_DIR}/shape_scan_full.json")


if __name__ == "__main__":
    main()
