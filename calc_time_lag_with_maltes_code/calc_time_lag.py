import typing
import numpy as np

from Malte_get_BH_mass import Line
from plot_utils import print_table_for_one_reference, save_centroid_as_txt
from import_data import import_centroid_and_mc_data, load_centroid_data_by_reference
from settings import FWHM_RMS, FWHM_ERR


ArrayLike = typing.Union[typing.Iterable[float], np.ndarray]

MASS_LINES = [
        'HAlpha',
        'HBeta',
        'HGamma',
        "HDelta",
        'HeI5875',
        "HeII1640_not_optical_calibrated",
        'HeII4685',
        'LyAlpha_not_optical_calibrated',
        "CIV1548_not_optical_calibrated",
    ]



def calc_centroid_malte_code(campaign, continuum, lines, include_mass=True, create_tex_file=True):

    correlation_data_dict, mc_data = import_centroid_and_mc_data(campaign, continuum, lines=lines)

    # Erstelle die Line-Objekte
    line_objects = []
    for line in lines:
        if line in correlation_data_dict.keys() and mc_data[line]["centroids"] is not None and mc_data[line]["peaks"] is not None:
            line_obj = Line(
                line,  # Name der Linie
                FWHM_RMS.get(line, 0.0),  # FWHM (rms)
                FWHM_ERR.get(line, 0.0),  # Sigma (rms)
                np.vstack((correlation_data_dict['time shift (tau)'], correlation_data_dict[line])).T,
                mc_data[line]["centroids"],
                mc_data[line]["peaks"],
            )
            line_objects.append(line_obj)

    # Speichere die Ergebnisse mit dynamischem Dateinamen
    if create_tex_file:
        output_filename = f'CCF_lags_{campaign}_{continuum}.tex'
        print(f"✅ Speichere Ergebnisse in Datei: {output_filename}")
        save_centroid_as_txt(line_objects, f"CCF_lags_{campaign}_{continuum}.txt")
        print_table_for_one_reference(output_filename, line_objects, continuum, include_mass=include_mass)
    return line_objects





def weighted_mean_asym_errors(
    m: ArrayLike,
    err_minus: ArrayLike,
    err_plus: ArrayLike,
    symmetrize: str = "mean",   # "mean" or "max"
) -> typing.Dict[str, float]:
    """
    Inverse-variance weighted mean for measurements with asymmetric 1σ errors.

    Parameters
    ----------
    m : array-like
        Measurements M_i (e.g., SMBH masses in units of 1e7 Msun).
    err_minus : array-like
        Lower 1σ uncertainties σ_i^- (positive numbers).
    err_plus : array-like
        Upper 1σ uncertainties σ_i^+ (positive numbers).
    symmetrize : {"mean","max"}
        How to symmetrize asymmetric errors to get σ_i for weights:
        - "mean": σ_i = (σ_i^- + σ_i^+)/2  (common choice)
        - "max" : σ_i = max(σ_i^-, σ_i^+)  (more conservative)
    ddof : int
        Not used in the standard formula; kept for compatibility if you extend this later.

    Returns
    -------
    dict with keys:
        mean  : weighted mean
        err   : 1σ uncertainty of the weighted mean, sqrt(1/sum(w_i))
        weights_sum : sum of weights
    """
    m = np.asarray(m, dtype=float)
    em = np.asarray(err_minus, dtype=float)
    ep = np.asarray(err_plus, dtype=float)

    if not (m.shape == em.shape == ep.shape):
        raise ValueError("m, err_minus, and err_plus must have the same shape.")

    if np.any(em <= 0) or np.any(ep <= 0):
        raise ValueError("All uncertainties must be positive.")

    if symmetrize == "mean":
        sigma = 0.5 * (em + ep)
    elif symmetrize == "max":
        sigma = np.maximum(em, ep)
    else:
        raise ValueError("symmetrize must be 'mean' or 'max'.")

    w = 1.0 / sigma**2
    wsum = np.sum(w)

    mean = np.sum(w * m) / wsum
    err = np.sqrt(1.0 / wsum)

    return {"mean": float(mean), "err": float(err), "weights_sum": float(wsum)}


def rescale_mass_with_shift(M, tau, tau_err_low, tau_err_high, shift=0.5):
    """
    Rescale BH mass when R changes from tau to tau+shift (in light-days).
    Returns: (M_new, M_new_err_low, M_new_err_high)
    Error propagation uses asymmetric tau errors only.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    # central scaling
    fac = (tau + shift) / tau
    M_new = M * fac

    # asymmetric tau bounds
    tau_low = max(tau - tau_err_low, 1e-12)   # avoid <=0
    tau_high = tau + tau_err_high

    # propagate via tau bounds (only through scaling factor)
    M_low_new = M * (tau_low + shift) / tau
    M_high_new = M * (tau_high + shift) / tau

    err_low_new = M_new - M_low_new
    err_high_new = M_high_new - M_new

    return M_new, err_low_new, err_high_new


def mean_bh_mass(reference="UVW2", shift_ld=0.5, mass_lines=None):
    centroid_and_masses = load_centroid_data_by_reference()

    m = []
    err_minus = []
    err_plus = []

    if mass_lines is None:
        mass_lines = MASS_LINES

    reference_data = centroid_and_masses[reference]

    for line in mass_lines:
        data = reference_data[line]


        M = data["M_Mo"]
        tau = data["tau_cent"]
        tau_err_low = data["tau_cent_err_low"]
        tau_err_high = data["tau_cent_err_high"]


        M_new, M_err_low_new, M_err_high_new = rescale_mass_with_shift(
            M, tau, tau_err_low, tau_err_high, shift=shift_ld
        )

        m.append(M_new)
        err_minus.append(M_err_low_new)
        err_plus.append(M_err_high_new)

    res = weighted_mean_asym_errors(m, err_minus, err_plus, symmetrize="max")

    print(f"Weighted mean = {res['mean']:.3f} ± {res['err']:.3f}  (in units of 1e7 Msun)")


def main():

    calc_centroid_malte_code("NGC4593_optical_calibrated", "LyAlpha_not_optical_calibrated", lines=['HAlpha', "HBeta", "OI8446"], include_mass=False, create_tex_file=True)
    calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150_not_optical_calibrated", lines=["UVW2"], include_mass=False, create_tex_file=True)
    calc_centroid_malte_code("NGC4593_optical_calibrated", "HAlpha", lines=["OI8446"], include_mass=False, create_tex_file=True)
    calc_centroid_malte_code("NGC4593_optical_calibrated", "UVW2", lines=['HAlpha', 'HBeta', 'HGamma', "HDelta",'HeI5875', 'HeII4685', 'LyAlpha_not_optical_calibrated', "OI8446"], include_mass=True, create_tex_file=True)
    calc_centroid_malte_code("NGC4593_not_optical_calibrated", "UVW2",
                             lines=["LyAlpha_not_optical_calibrated", "SiIV1393_not_optical_calibrated",  "NV1238_not_optical_calibrated",
                                    "CIV1548_not_optical_calibrated", "HeII1640_not_optical_calibrated"], include_mass=True, create_tex_file=True)
    mean_bh_mass()


if __name__ == "__main__":
    main()