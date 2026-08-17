"""
Real posterior (emcee affine-invariant ensemble sampler) for the
confinement-mechanism fit against DESI DR2 BAO + Planck compressed distance
priors + SH0ES H0, replacing the point-estimate chi^2 minimum from
scripts/fit_desi_planck_sh0es.py.

Why this is needed (open item #2 logged 2026-08-16 in NOTES.md): that
script found a_c is genuinely degenerate below ~1e-5 once Delta_a is held
fixed - re-running the minimizer with the lower bound relaxed landed at a
different a_c with identical chi2. A naive covariance/Fisher estimate at a
single best-fit point would misrepresent that flat direction as a normal
(possibly tight) Gaussian uncertainty. A full posterior instead shows the
degenerate direction directly as a broad, prior-bounded marginal.

Priors: flat/uniform over the same physical bounds already used as hard
cutoffs in fit_desi_planck_sh0es.chi2_total (60<H0<82 km/s/Mpc,
0.019<ombh2<0.026, 0.08<omch2<0.16, 0<=rho0_conf<0.95,
1e-7<=a_c<=5e-3, sampled in log10(a_c) since the degenerate direction
spans many decades).

Likelihood: L = exp(-chi2_total/2), i.e. Gaussian in each of the three
real datasets' own quoted uncertainties - identical to what the
point-estimate fit already assumed, just now propagated as a full
posterior instead of a single minimum.
"""
import os
import sys
import time

import numpy as np
import emcee

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_desi_planck_sh0es import chi2_total, run_fit  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = ["H0", "ombh2", "omch2", "rho0_conf", "log10_ac"]
BOUNDS = np.array([
    [60.0, 82.0],
    [0.019, 0.026],
    [0.08, 0.16],
    [0.0, 0.95],
    [np.log10(1e-7), np.log10(5e-3)],
])


def log_prior(theta):
    if np.any(theta < BOUNDS[:, 0]) or np.any(theta > BOUNDS[:, 1]):
        return -np.inf
    return 0.0  # flat


def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    c2 = chi2_total(theta, use_confinement=True)
    if c2 >= 1e11:  # chi2_total's own out-of-bounds/invalid sentinel
        return -np.inf
    return lp - 0.5 * c2


def main(n_walkers=40, n_burn=500, n_prod=2000, seed=0):
    print("Finding starting point (point-estimate best fit) ...")
    best_res, labels = run_fit(use_confinement=True, n_restarts=6)
    x0 = best_res.x
    print(f"  starting from: {dict(zip(labels, np.round(x0, 5)))}")

    ndim = len(x0)
    rng = np.random.default_rng(seed)
    spread = np.array([0.5, 2e-4, 5e-3, 0.02, 0.3])
    p0 = x0[None, :] + spread[None, :] * rng.normal(size=(n_walkers, ndim))
    # Clip initial walkers into the prior box so none start at -inf.
    p0 = np.clip(p0, BOUNDS[:, 0] + 1e-6, BOUNDS[:, 1] - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)

    print(f"Burn-in: {n_burn} steps x {n_walkers} walkers ...")
    t0 = time.time()
    state = sampler.run_mcmc(p0, n_burn, progress=False)
    sampler.reset()
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Production: {n_prod} steps x {n_walkers} walkers ...")
    t0 = time.time()
    sampler.run_mcmc(state, n_prod, progress=False)
    print(f"  done in {time.time()-t0:.1f}s")

    acc = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction: {acc:.3f}")

    try:
        tau = sampler.get_autocorr_time(tol=0)
        print(f"Integrated autocorrelation time per param: {np.round(tau, 1)}")
    except Exception as exc:
        tau = np.full(ndim, np.nan)
        print(f"  autocorr time estimate failed/unreliable: {exc}")

    chain = sampler.get_chain(flat=True)
    logprob = sampler.get_log_prob(flat=True)
    np.savez(os.path.join(RESULTS_DIR, "mcmc_confinement_chain.npz"),
              chain=chain, logprob=logprob, labels=LABELS, tau=tau,
              bounds=BOUNDS, n_walkers=n_walkers, n_burn=n_burn, n_prod=n_prod)
    print(f"Saved chain to {RESULTS_DIR}/mcmc_confinement_chain.npz "
          f"({chain.shape[0]} samples)")

    print("\n=== Marginalized posterior (16 / 50 / 84 percentiles) ===")
    for i, lab in enumerate(LABELS):
        lo, med, hi = np.percentile(chain[:, i], [16, 50, 84])
        print(f"  {lab:10s} = {med:.5f}  (+{hi-med:.5f} / -{med-lo:.5f})")

    rho_frac_gt_010 = np.mean(chain[:, 3] > 0.10)
    rho_frac_gt_0087 = np.mean(chain[:, 3] > 0.087)
    print(f"\nP(rho0_conf > 0.10, this paper's own stated bound)  = {rho_frac_gt_010:.3f}")
    print(f"P(rho0_conf > 0.087, real Planck f_EDE 95%CL analog) = {rho_frac_gt_0087:.3f}")

    ac_lo, ac_med, ac_hi = np.percentile(chain[:, 4], [16, 50, 84])
    ac_span = BOUNDS[4, 1] - BOUNDS[4, 0]
    print(f"\nlog10(a_c) 16/50/84 = {ac_lo:.2f} / {ac_med:.2f} / {ac_hi:.2f} "
          f"(prior range spans {BOUNDS[4,0]:.2f} to {BOUNDS[4,1]:.2f}, "
          f"width {ac_span:.2f} dex)")
    print("If the log10(a_c) posterior width is comparable to the prior "
          "width, that confirms the degeneracy found in the point-estimate "
          "fit directly, rather than inferring it indirectly from repeated "
          "minimizer restarts.")

    try:
        import corner
        fig = corner.corner(chain, labels=LABELS, show_titles=True,
                              title_fmt=".4f", quantiles=[0.16, 0.5, 0.84])
        fig.savefig(os.path.join(RESULTS_DIR, "mcmc_confinement_corner.png"),
                    dpi=150, bbox_inches="tight")
        print(f"\nSaved corner plot to {RESULTS_DIR}/mcmc_confinement_corner.png")
    except ImportError:
        print("\n(corner not installed - skipped corner plot)")

    return sampler, chain


if __name__ == "__main__":
    main()
