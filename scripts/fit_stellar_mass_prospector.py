"""
Proper Bayesian stellar-mass fitting for high-z dropout candidates, using
Prospector (FSPS-based, nested sampling via dynesty) with an explicit
age-of-universe-at-z prior, replacing the earlier EAZY/FSPS-template point
estimates that turned out to be dominated by a single unphysically old/dusty
template (see JWST\Code\searching\photoz_2180_localbg.py,
photoz_34.py, photoz_a2744_97.py for that diagnosis).

Photometry (aperture-corrected, encircled-energy-corrected uJy fluxes) is
taken directly from the existing EAZY .cat files already on disk - not
re-derived here. Redshift is fixed to the previously-established EAZY
best-fit z for each candidate (not re-fit jointly with mass here, to keep
this a direct, interpretable test of "does capping template age at the age
of the universe fix the mass estimate").

Usage: python fit_stellar_mass_prospector.py <candidate>
  where <candidate> is one of: 2180, 34, 97
"""
import sys
import numpy as np

from prospect.models import SpecModel, templates
from prospect.models.sedmodel import SpecModel
from prospect.sources import CSPSpecBasis
from prospect.fitting import fit_model
from prospect.observation import from_oldstyle
from sedpy.observate import load_filters
import astropy.cosmology as apcosmo

COSMO = apcosmo.Planck18

# (candidate id, fixed redshift, bands, fluxes uJy, errors uJy)
CANDIDATES = {
    "2180": (
        8.30,  # eazy_v1.3 best z (the more conservative of the two template-set fits)
        ["jwst_f090w", "jwst_f115w", "jwst_f444w"],
        [-0.005765, 0.012632, 0.251403],
        [0.003305, 0.002846, 0.003828],
    ),
    "34": (
        9.59,
        ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w", "jwst_f277w", "jwst_f356w", "jwst_f444w"],
        [-0.001435, 0.005622, 0.012737, 0.006110, -0.004945, 0.001106, 0.097592],
        [0.003221, 0.002773, 0.002671, 0.002430, 0.001774, 0.001690, 0.002851],
    ),
    "97": (
        8.94,
        ["jwst_f115w", "jwst_f150w", "jwst_f444w"],
        [0.005742, 0.024950, 0.047281],
        [0.003239, 0.002776, 0.003245],
    ),
}


def build_obs(cand_id):
    z, band_names, flux_ujy, err_ujy = CANDIDATES[cand_id]
    filters = load_filters(band_names)
    flux_maggies = np.array(flux_ujy) * 1e-6 / 3631.0
    err_maggies = np.array(err_ujy) * 1e-6 / 3631.0
    obs = dict(
        wavelength=None,
        spectrum=None,
        unc=None,
        redshift=z,
        filters=filters,
        maggies=flux_maggies,
        maggies_unc=err_maggies,
        phot_mask=np.isfinite(flux_maggies),
    )
    observations = from_oldstyle(obs)
    return observations, z


def build_model(z):
    model_params = templates.TemplateLibrary["parametric_sfh"]

    model_params["zred"]["init"] = z
    model_params["zred"]["isfree"] = False

    t_universe_gyr = COSMO.age(z).value
    model_params["tage"]["prior"] = templates.priors.TopHat(mini=0.001, maxi=t_universe_gyr)
    model_params["tage"]["init"] = min(0.1, t_universe_gyr / 2)

    model_params["mass"]["prior"] = templates.priors.LogUniform(mini=1e6, maxi=1e13)
    model_params["mass"]["init"] = 1e9

    model_params["tau"]["prior"] = templates.priors.LogUniform(mini=0.1, maxi=30.0)

    model_params["dust2"]["prior"] = templates.priors.TopHat(mini=0.0, maxi=4.0)
    model_params["dust2"]["init"] = 0.3

    model_params["logzsol"]["prior"] = templates.priors.TopHat(mini=-2.0, maxi=0.2)
    model_params["logzsol"]["isfree"] = False
    model_params["logzsol"]["init"] = -1.0

    return SpecModel(model_params), t_universe_gyr


def build_sps():
    return CSPSpecBasis(zcontinuous=1)


def weighted_percentile(values, weights, percentiles):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(percentiles, cdf, v)


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in CANDIDATES:
        print(f"Usage: python {sys.argv[0]} <candidate> [fast]  (one of {list(CANDIDATES)})")
        sys.exit(1)

    cand_id = sys.argv[1]
    fast = len(sys.argv) == 3 and sys.argv[2] == "fast"

    observations, z = build_obs(cand_id)
    model, t_universe_gyr = build_model(z)
    sps = build_sps()

    print(f"=== Candidate #{cand_id}, fixed z={z:.3f}, age of universe at z = {t_universe_gyr*1000:.1f} Myr ===")
    print(f"tage prior capped at {t_universe_gyr:.4f} Gyr (this is the fix for the earlier unphysical masses)")
    if fast:
        print("FAST mode: reduced live points/calls, for pipeline validation only, not a real posterior")

    # Real dynesty kwargs must match sampler.run_nested()'s own unprefixed
    # parameter names to actually be forwarded (see
    # prospect.fitting.fitting.run_nested_sampler's run_keys filtering).
    # nested_target_n_effective (-> n_effective) is the actual governing
    # stop criterion, NOT maxcall/dlogz_init prefixed with "nested_" (those
    # were silently dropped, which is why the first "fast" attempt ran with
    # full default settings and took ~40min instead of ~2).
    run_params = dict(
        nested_nlive=40 if fast else 400,
        nested_target_n_effective=100 if fast else 2000,
        maxcall=5000 if fast else 500000,
    )

    output = fit_model(observations, model, sps, nested_sampler="dynesty", **run_params)

    result = output["sampling"]
    print("sampling result keys:", list(result.keys()))

    points = np.asarray(result["points"])
    log_weight = np.asarray(result["log_weight"])
    weights = np.exp(log_weight - log_weight.max())

    npz_file = f"prospector_cand{cand_id}{'_fast' if fast else ''}.npz"
    np.savez(npz_file, points=points, log_weight=log_weight,
             log_like=np.asarray(result["log_like"]), theta_labels=np.array(model.theta_labels()))
    print(f"Saved raw posterior samples to {npz_file}")

    labels = model.theta_labels()
    mass_idx = labels.index("mass")
    mass_samples = points[:, mass_idx]

    p16, p50, p84 = weighted_percentile(mass_samples, weights, [0.16, 0.50, 0.84])

    print()
    print(f"=== Posterior stellar mass, candidate #{cand_id} (z fixed at {z:.3f}) ===")
    print(f"log10(M/Msun) = {np.log10(p50):.2f}  (+{np.log10(p84)-np.log10(p50):.2f} / -{np.log10(p50)-np.log10(p16):.2f})")

    for pname in ("tage", "dust2", "tau", "logzsol"):
        if pname in labels:
            idx = labels.index(pname)
            lo, mid, hi = weighted_percentile(points[:, idx], weights, [0.16, 0.50, 0.84])
            print(f"{pname}: {mid:.3f} (+{hi-mid:.3f} / -{mid-lo:.3f})")


if __name__ == "__main__":
    main()
