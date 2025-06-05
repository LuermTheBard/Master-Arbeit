from pathlib import Path

import numpy as np
from astropy.constants.codata2018 import c
from scipy.interpolate import interp1d

from import_data import find_prime_data_folder
from settings import F_MEAN, F_SIGMA, CENTRAL_WAVELENGTH


def calculate_standard_error_for_lightcurves(flux, flux_noise_err):
    """
    Calculates the total uncertainty of a lightcurve data point based on:

    1. The relative scatter of the lightcurve (F_SIGMA / F_MEAN),
    2. The individual measurement noise (flux_noise_err).

    Formula:
    total_error = sqrt((F_SIGMA / F_MEAN * flux)^2 + flux_noise_err^2)

    Parameters:
    -----------
    flux : float or np.ndarray
        Observed flux value(s) of the lightcurve.
    flux_noise_err : float or np.ndarray
        Measurement noise or instrumental uncertainty.

    Returns:
    -----------
    float or np.ndarray
        The total error associated with the input flux value(s).
    """

    if F_MEAN == 0:
        raise ValueError("Der Mittelwert des Flusses (F_MEAN) darf nicht 0 sein.")

    total_error = np.sqrt((F_SIGMA / F_MEAN * flux) ** 2 + flux_noise_err ** 2)
    return total_error


# -------------------------------------------------------------------------------------------------------------------

def save_centroid_as_txt(line_objects, output_file="centroids.txt", include_mass=True):
    """
    Saves centroid lags and optionally black hole masses in a machine-readable .txt file.

    Parameters:
    -----------
    line_objects : list
        List of emission line objects with:
        - name
        - tau_cent, tau_cent_err (tuple)
        - M_Mo, M_Mo_err (tuple), if include_mass is True
    output_path : str
        Path to output text file.
    include_mass : bool
        Include black hole mass estimates if True.

    Returns:
    --------
    None
    """
    output_folder = find_prime_data_folder() / "centroids"
    ensure_output_dir(output_folder)


    with open(str(output_folder / output_file), "w") as f:
        if include_mass:
            f.write("name tau_cent tau_cent_err_low tau_cent_err_high M_Mo M_Mo_err_low M_Mo_err_high\n")
        else:
            f.write("name tau_cent tau_cent_err_low tau_cent_err_high\n")

        for line in line_objects:
            tau = line.tau_cent
            err_low = tau - line.tau_cent_err[0]
            err_high = line.tau_cent_err[1] - tau

            if include_mass:
                M = line.M_Mo
                M_err_low = M - line.M_Mo_err[0]
                M_err_high = line.M_Mo_err[1] - M
                f.write(f"{line.name} {tau:.4f} {err_low:.4f} {err_high:.4f} {M:.4f} {M_err_low:.4f} {M_err_high:.4f}\n")
            else:
                f.write(f"{line.name} {tau:.4f} {err_low:.4f} {err_high:.4f}\n")




def print_table_for_one_reference(filename, linelist, continuum, include_mass=True):
    """
    Generates a LaTeX document containing a formatted table with time lag measurements
    (centroid and peak) for a given reference lightcurve and its associated emission lines.

    Parameters:
    -----------
    filename : str or Path
        Output file path where the LaTeX document will be saved.
    linelist : list
        List of objects representing emission lines. Each object must have the attributes:
        - name
        - tau_cent, tau_cent_err (tuple)
        - tau_peak, tau_peak_err (tuple)
        - M_Mo, M_Mo_err (tuple), if include_mass is True
    continuum : str
        Name of the reference continuum lightcurve. Used for labeling.
    include_mass : bool, optional
        Whether to include black hole mass estimates (with errors) in the table. Default is True.

    Returns:
    -----------
    None
    """

    with open(filename, 'w') as outfile:
        # LaTeX Dokument-Kopf
        outfile.write(r'\documentclass{article}' + '\n')
        outfile.write(r'\usepackage{booktabs}' + '\n')  # Schöne Tabellen
        outfile.write(r'\usepackage{siunitx}' + '\n')   # Zahlenformatierung
        outfile.write(r'\usepackage{amsmath}' + '\n')   # _{-x}^{+y}-Notation
        outfile.write(r'\usepackage{graphicx}' + '\n')  # Falls notwendig
        outfile.write(r'\begin{document}' + '\n\n')

        # LaTeX Tabelle mit Continuum-Info
        outfile.write(r'\begin{table}[!htb]' + '\n')
        outfile.write(r'\centering' + '\n')
        escaped_continuum = continuum.replace('_', r'\_')
        outfile.write(fr'\caption{{Centroid and Peak Time Lag for {escaped_continuum}.}}' + '\n')
        outfile.write(fr'\label{{tab:lags_{continuum}}}' + '\n')

        # Tabellenformat abhängig von include_mass
        if include_mass:
            outfile.write(r'\begin{tabular}{l c c c}' + '\n')
        else:
            outfile.write(r'\begin{tabular}{l c c}' + '\n')

        outfile.write(r'\toprule' + '\n')

        # Spaltenüberschriften
        if include_mass:
            outfile.write(r'Name & $\tau_{\text{cent}}$ [d] & $\tau_{\text{peak}}$ [d] & $M_{\text{BH}} [10^7 M_\odot]$ \\' + '\n')
        else:
            outfile.write(r'Name & $\tau_{\text{cent}}$ [d] & $\tau_{\text{peak}}$ [d] \\' + '\n')

        outfile.write(r'\midrule' + '\n')

        # Tabelleninhalt mit Fehlerdarstellung
        for line in linelist:
            tau_cent_str = f"{line.tau_cent:.1f} \\ensuremath{{_{{{round(line.tau_cent_err[0] - line.tau_cent,1)}}}^{{+{round(line.tau_cent_err[1] - line.tau_cent, 1)}}}}}"
            tau_peak_str = f"{line.tau_peak:.1f} \\ensuremath{{_{{{line.tau_peak_err[0] - line.tau_peak:.1f}}}^{{+{line.tau_peak_err[1] - line.tau_peak:.1f}}}}}"
            name = format_label(line.name)

            if include_mass:
                mass_str = f"{line.M_Mo:.1f} \\ensuremath{{_{{{line.M_Mo_err[0] - line.M_Mo:.1f}}}^{{+{line.M_Mo_err[1] - line.M_Mo:.1f}}}}}"
                outfile.write(f"{format_label(name, as_latex=True)} & ${tau_cent_str}$ & ${tau_peak_str}$ & ${mass_str}$ \\\\" + '\n')
            else:
                outfile.write(f"{format_label(name, as_latex=True)} & ${tau_cent_str}$ & ${tau_peak_str}$ \\\\" + '\n')

        # Tabellenabschluss
        outfile.write(r'\bottomrule' + '\n')
        outfile.write(r'\end{tabular}' + '\n')
        outfile.write(r'\end{table}' + '\n\n')

        # LaTeX Dokument-Abschluss
        outfile.write(r'\end{document}' + '\n')


def print_table_for_multiple_reference(filename, reference_light_curve_lines_dict, include_mass=True):
    """
    Generates a LaTeX document containing a formatted table with time lag measurements
    for multiple reference lightcurves, each with its associated emission lines.

    Parameters:
    -----------
    filename : str or Path
        Output file path where the LaTeX document will be saved.
    reference_light_curve_lines_dict : dict
        Dictionary mapping each reference lightcurve name to a list of emission line objects.
        Each object must have the attributes:
        - name
        - tau_cent, tau_cent_err (tuple)
        - tau_peak, tau_peak_err (tuple)
        - M_Mo, M_Mo_err (tuple), if include_mass is True
    include_mass : bool, optional
        Whether to include black hole mass estimates (with errors) in the table. Default is True.

    Returns:
    -----------
    None
    """

    with open(filename, 'w') as outfile:
        # LaTeX Dokument-Kopf
        outfile.write(r'\documentclass{article}' + '\n')
        outfile.write(r'\usepackage{booktabs}' + '\n')  # Schöne Tabellen
        outfile.write(r'\usepackage{siunitx}' + '\n')   # Zahlenformatierung
        outfile.write(r'\usepackage{amsmath}' + '\n')   # _{-x}^{+y}-Notation
        outfile.write(r'\usepackage{graphicx}' + '\n')  # Falls notwendig
        outfile.write(r'\begin{document}' + '\n\n')

        # LaTeX Tabelle mit Continuum-Info
        outfile.write(r'\begin{table}[!htb]' + '\n')
        outfile.write(r'\centering' + '\n')
        outfile.write(fr'\caption{{Centroid and Peak Time Lag for multiple references}}' + '\n')
        outfile.write(fr'\label{{tab:lags_multiple_references}}' + '\n')

        # Tabellenformat abhängig von include_mass
        if include_mass:
            outfile.write(r'\begin{tabular}{l c c c}' + '\n')
        else:
            outfile.write(r'\begin{tabular}{l c c}' + '\n')

        outfile.write(r'\toprule' + '\n')

        # Spaltenüberschriften
        if include_mass:
            outfile.write(r'Name & $\tau_{\text{cent}}$ [d] & $\tau_{\text{peak}}$ [d] & $M_{\text{BH}} [10^7 M_\odot]$ \\' + '\n')
        else:
            outfile.write(r'Name & $\tau_{\text{cent}}$ [d] & $\tau_{\text{peak}}$ [d] \\' + '\n')

        outfile.write(r'\midrule' + '\n')

        # Tabelleninhalt mit Fehlerdarstellung
        for reference, lines in reference_light_curve_lines_dict.items():
            # Zwischenüberschrift für jede Referenz
            num_cols = 4 if include_mass else 3
            escaped_ref = format_label(reference)

            outfile.write(r'\midrule' + '\n')
            outfile.write(fr'\multicolumn{{{num_cols}}}{{l}}{{\textbf{{Reference: {escaped_ref}}}}} \\' + '\n')
            outfile.write(r'\midrule' + '\n')

            for line in lines:
                tau_cent_str = f"{line.tau_cent:.1f} \\ensuremath{{_{{{line.tau_cent_err[0] - line.tau_cent:.1f}}}^{{+{line.tau_cent_err[1] - line.tau_cent:.1f}}}}}"
                tau_peak_str = f"{line.tau_peak:.1f} \\ensuremath{{_{{{line.tau_peak_err[0] - line.tau_peak:.1f}}}^{{+{line.tau_peak_err[1] - line.tau_peak:.1f}}}}}"
                name = format_label(line.name)

                if include_mass:
                    mass_str = f"{line.M_Mo:.1f} \\ensuremath{{_{{{line.M_Mo_err[0] - line.M_Mo:.1f}}}^{{+{line.M_Mo_err[1] - line.M_Mo:.1f}}}}}"
                    outfile.write(f"{name} & ${tau_cent_str}$ & ${tau_peak_str}$ & ${mass_str}$ \\\\" + '\n')
                else:
                    outfile.write(f"{name} & ${tau_cent_str}$ & ${tau_peak_str}$ \\\\" + '\n')

        # Tabellenabschluss
        outfile.write(r'\bottomrule' + '\n')
        outfile.write(r'\end{tabular}' + '\n')
        outfile.write(r'\end{table}' + '\n\n')

        # LaTeX Dokument-Abschluss
        outfile.write(r'\end{document}' + '\n')


def format_label(name, as_latex=True):
    """
    Formats a spectral line or continuum identifier for use in plots or LaTeX documents.

    - Converts internal identifiers (e.g., "HAlpha", "HeII4685", "Cont1") into human-readable labels.
    - Optionally applies LaTeX-specific formatting and escapes underscores.
    - Handles both emission lines and continua.

    Parameters:
    -----------
    name : str
        The original name or identifier of the line or continuum (e.g., "HAlpha", "Cont1").
    as_latex : bool, optional
        If True (default), returns a LaTeX-compatible label (e.g., for tables).
        If False, returns a plain-text version (e.g., for matplotlib plots).

    Returns:
    -----------
    str
        A formatted and optionally LaTeX-escaped label for display.
    """

    original_name = name
    # Escape für LaTeX
    name = name.replace('_', r'\_') if as_latex else name.replace('_', ' ')

    # Continuum?
    if "Cont" in name:
        # is_not_calibrated = ("not\\_optical\\_calibrated" in name if as_latex else "not_optical_calibrated" in original_name)
        num_part = ''.join(filter(str.isdigit, name))
        label = f"Cont. {num_part}" if num_part else name
        # Not needed anymore:
        # if is_not_calibrated:
        #     label += " (not opt. calib.)"
        return label

    # Linie?
    is_not_calibrated = ("not\\_optical\\_calibrated" in name if as_latex else "not_optical_calibrated" in original_name)
    base_name = name.split(r"\_")[0] if as_latex else original_name.split("_")[0]

    replacements_latex = {
        "HAlpha": r"H$\alpha$",
        "HBeta": r"H$\beta$",
        "HGamma": r"H$\gamma$",
        "HDelta": r"H$\delta$",
        "HeI5875": r"He\,\textsc{i}\,5875",
        "HeI7065": r"He\,\textsc{i}\,7065",
        "HeI4471": r"He\,\textsc{i}\,4471",
        "HeI5015": r"He\,\textsc{i}\,5015",
        "HeII4685": r"He\,\textsc{ii}\,4685",
        "OI8446": r"O\,\textsc{i}\,8446",
        "LyAlpha": r"Ly$\alpha$",
        "SiIV1393": r"Si\,\textsc{iv}\,1393",
        "NV1238": r"N\,\textsc{v}\,1238",
        "CIV1548": r"C\,\textsc{iv}\,1548",
        "HeII1640": r"He\,\textsc{ii}\,1640",
    }

    replacements_plot = {
        "UVW2": "UVW2       ",
        "HAlpha": r"Hα            ",
        "HBeta": r"Hβ            ",
        "HGamma": r"Hγ            ",
        "HDelta": r"Hδ            ",
        "HeI5875": r"$\text{He}\thinspace\text{I} \ 5875$  ",
        "HeI7065": r"$\text{He}\thinspace\text{I} \ 7065$  ",
        "HeI4471": r"$\text{He}\thinspace\text{I} \ 4471$  ",
        "HeI5015": r"$\text{He}\thinspace\text{I} \ 5015$  ",
        "HeII4685": r"$\text{He}\thinspace\text{II} \ 4685$ ",
        "OI8446": r"$\text{O}\thinspace\text{I} \ 8446$  ",
        "LyAlpha": r"Lyα           ",
        "SiIV1393": r"$\text{Si}\thinspace\text{IV} \ 1393$",
        "NV1238": r"$\text{N}\thinspace\text{V} \ 1238$  ",
        "CIV1548": r"$\text{C}\thinspace\text{IV} \ 1548$ ",
        "HeII1640": r"$\text{He}\thinspace\text{II} \ 1640$",
    }

    replacements = replacements_latex if as_latex else replacements_plot

    formatted = replacements.get(base_name, base_name)
    # not needed anymore
    # if is_not_calibrated:
    #     formatted += " (not opt. calib.)"

    return formatted

#------------------------------------------------------------------------------------------------------------------------------------


def subtract_continuum(wavelength, intensity, line_wavelength, left_range, right_range):
    """
    Estimates and subtracts the pseudo-continuum around an emission line in a spectrum,
    and normalizes the result to the local maximum near the line.

    A linear continuum is estimated using the mean intensity in two sidebands
    (left and right of the line), and then subtracted from the original spectrum.
    The result is normalized to the peak value within ±50 units of the line center.

    Parameters:
    -----------
    wavelength : array-like
        Array of wavelength values.
    intensity : array-like
        Array of flux or intensity values corresponding to the wavelengths.
    line_wavelength : float
        Central wavelength of the emission line of interest.
    left_range : tuple (float, float)
        Wavelength range used to estimate the left-side continuum (e.g., (start, end)).
    right_range : tuple (float, float)
        Wavelength range used to estimate the right-side continuum.

    Returns:
    -----------
    corrected_intensity : np.ndarray
        Intensity with the pseudo-continuum subtracted and normalized to its local maximum.
    continuum : np.ndarray
        The estimated linear continuum over the full wavelength range.
    """

    left_mask = (wavelength > left_range[0]) & (wavelength < left_range[1])
    right_mask = (wavelength > right_range[0]) & (wavelength < right_range[1])

    left_mean = (np.mean(wavelength[left_mask]), np.mean(intensity[left_mask]))
    right_mean = (np.mean(wavelength[right_mask]), np.mean(intensity[right_mask]))

    continuum_fit = interp1d([left_mean[0], right_mean[0]], [left_mean[1], right_mean[1]], kind="linear",
                             fill_value="extrapolate")
    continuum = continuum_fit(wavelength)

    corrected_intensity = intensity - continuum

    # Bestimme das Maximum innerhalb des Bereichs ±50 um line_wavelength
    norm_range = (line_wavelength - 50, line_wavelength + 50)
    norm_mask = (wavelength > norm_range[0]) & (wavelength < norm_range[1])
    if np.any(norm_mask):
        max_value = np.max(corrected_intensity[norm_mask])
    else:
        max_value = np.max(corrected_intensity)  # Falls der Bereich leer ist, fallback auf das globale Maximum

    corrected_intensity /= max_value

    return corrected_intensity, continuum


def convert_to_velocity(wavelength, line_wavelength):
    """
    Converts wavelength values to velocity space (in km/s), relative to a given line center.

    The velocity is calculated using the non-relativistic Doppler formula:
        v = (λ - λ₀) / λ₀ * c

    Parameters:
    -----------
    wavelength : array-like
        Observed wavelength(s).
    line_wavelength : float
        Central rest-frame wavelength of the emission line.

    Returns:
    -----------
    np.ndarray
        Velocity values in km/s corresponding to the input wavelengths.
    """

    c_km_s = c.to('km/s').value  # Lichtgeschwindigkeit in km/s
    return (wavelength - line_wavelength) / line_wavelength * c_km_s


def transform_wavelength_to_velocity_and_cut(wavelength, intensity, line_name, velocity_range=None, filename=None):
    """
    Converts a spectrum from wavelength to velocity space centered on a given emission line,
    normalizes the flux to its local maximum, and optionally crops the result to a velocity range.

    Parameters:
    -----------
    wavelength : array-like
        Array of observed wavelengths.
    intensity : array-like
        Array of intensity or flux values.
    line_name : str
        Name of the emission line, used to look up its central wavelength.
    velocity_range : tuple (float, float), optional
        Velocity range in km/s to keep around zero. If provided, must be (min < 0, max > 0).
    filename : str or Path, optional
        If provided, the resulting velocity–flux data will be written to this file as text.

    Returns:
    -----------
    velocity : np.ndarray
        Velocity values (in km/s), possibly cropped to the specified range.
    intensity : np.ndarray
        Corresponding normalized intensity values.
    """


    line_wavelength = CENTRAL_WAVELENGTH[line_name]

    # Transformation der Wellenlängen in den Geschwindigkeitsraum
    velocity = convert_to_velocity(wavelength, line_wavelength)

    intensity = normalize_to_maximum(intensity, line_wavelength, wavelength)

    intensity, velocity = cut_normalized_line_out(intensity, velocity, velocity_range)

    # Falls eine Datei angegeben ist, speichern
    if filename is not None:
        with open(str(filename), 'w') as file:
            file.write("# velocity space (km/s) \t normalized flux\n")
            for v, i in zip(velocity, intensity):
                file.write(f"{v}\t{i}\n")

    return velocity, intensity


def normalize_to_maximum(intensity, line_wavelength, wavelength):
    """
    Normalizes the intensity values to the local maximum around the given line center.

    If a peak is found within ±10 units of `line_wavelength`, the intensities are scaled by that value.
    Otherwise, the global maximum is used as a fallback.

    Parameters:
    -----------
    intensity : array-like
        Array of intensity or flux values to be normalized.
    line_wavelength : float
        Central wavelength of the emission line.
    wavelength : array-like
        Corresponding wavelength values.

    Returns:
    -----------
    np.ndarray
        Normalized intensity array.
    """

    # Normalisierung der Intensität auf das Maximum
    if np.max(intensity) != 0:
        # Bestimme das Maximum innerhalb des Bereichs ±10 um line_wavelength
        norm_range = (line_wavelength - 10, line_wavelength + 10)
        norm_mask = (wavelength > norm_range[0]) & (wavelength < norm_range[1])
        if np.any(norm_mask):
            max_value = np.max(intensity[norm_mask])
        else:
            max_value = np.max(intensity)  # Falls der Bereich leer ist, fallback auf das globale Maximum

        intensity /= max_value
    return intensity


def cut_normalized_line_out(intensity, velocity, velocity_range):
    """
    Crops the intensity and velocity arrays to the specified velocity range.

    Ensures that the range is symmetric around 0 by forcing `min_v` to be ≤ 0 and `max_v` ≥ 0.

    Parameters:
    -----------
    intensity : array-like
        Normalized intensity values.
    velocity : array-like
        Velocity values in km/s.
    velocity_range : tuple (float, float)
        Desired velocity window (min, max). Should ideally be symmetric around zero.

    Returns:
    -----------
    intensity : np.ndarray
        Cropped intensity array.
    velocity : np.ndarray
        Cropped velocity array.
    """

    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if velocity_range is not None:
        min_v, max_v = velocity_range
        min_v = min(min_v, -abs(min_v))  # Sicherstellen, dass min negativ ist
        max_v = max(max_v, abs(max_v))  # Sicherstellen, dass max positiv ist
        mask = (velocity >= min_v) & (velocity <= max_v)
        velocity = velocity[mask]
        intensity = intensity[mask]
    return intensity, velocity


def cut_line_out(intensity, wavelength, wavelength_range):
    """
    Crops the intensity and wavelength arrays to a specified wavelength range.

    Parameters:
    -----------
    intensity : array-like
        Intensity values of the spectrum.
    wavelength : array-like
        Wavelength values of the spectrum.
    wavelength_range : tuple (float, float)
        Range of wavelengths to keep (min, max).

    Returns:
    -----------
    intensity : np.ndarray
        Cropped intensity array.
    wavelength : np.ndarray
        Cropped wavelength array.
    """

    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if wavelength_range is not None:
        min_v, max_v = wavelength_range
        mask = (wavelength >= min_v) & (wavelength <= max_v)
        wavelength = wavelength[mask]
        intensity = intensity[mask]
    return intensity, wavelength


def save_velocity_data_to_txt(filename, velocity, intensity):
    """
    Saves velocity and normalized intensity values to a plain text file in tab-separated format.

    The file includes a header and one data row per velocity–intensity pair.

    Parameters:
    -----------
    filename : str or Path
        Name or path of the file to save to.
    velocity : array-like
        Velocity values in km/s.
    intensity : array-like
        Corresponding normalized intensity (flux) values.

    Returns:
    -----------
    None
    """
    with open(filename, 'w') as file:
        file.write("# velocity space (km/s) \t normalized flux\n")
        for v, i in zip(velocity, intensity):
            file.write(f"{v}\t{i}\n")


def ensure_output_dir(output_dir):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    return output_dir_path
