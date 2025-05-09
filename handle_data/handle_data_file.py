import numpy as np

from import_data.import_data import import_line_profile_data
from plot_data.plot_line_profiles import transform_wavelength_to_velocity_and_cut
from settings import F_MEAN, F_SIGMA


def calculate_standard_error_for_lightcurves(flux, flux_noise_err):
    """
    Berechnet den Gesamtfehler einer Lichtkurve basierend auf:

    1. Der relativen Standardabweichung des Flusses (F_SIGMA / F_MEAN).
    2. Dem individuellen Rauschfehler (flux_noise_err).

    Formel:
    Gesamtfehler = sqrt((F_SIGMA / F_MEAN * FLUX)² + flux_noise_err²)

    Parameter:
    - flux (float oder np.array): Der beobachtete Flusswert oder ein Array von Flüssen.
    - flux_noise_err (float): Zusätzlicher Fehler durch Rauschen.

    Rückgabewert:
    - Gesamtfehler als float oder np.array (je nach Eingabetyp von `flux`).
    """

    if F_MEAN == 0:
        raise ValueError("Der Mittelwert des Flusses (F_MEAN) darf nicht 0 sein.")

    total_error = np.sqrt((F_SIGMA / F_MEAN * flux) ** 2 + flux_noise_err ** 2)
    return total_error


# -------------------------------------------------------------------------------------------------------------------


def get_continua_with_highest_corr_coef(line_sorted_data):
    line_best_continua_dict = dict()

    # Schritt 1: Berechnung der maximalen, zweitgrößten und drittgrößten Werte für jede Linie
    for line, cont_data_tuple_list in line_sorted_data.items():
        cont_max_dict = dict()

        for cont_tuple in cont_data_tuple_list:
            # cont_tuple: (continua_name, x_values, y_values)
            cont_max_dict[cont_tuple[0]] = sum(cont_tuple[2])

        # Schritt 2: Sortiere cont_max_dict nach Werten absteigend
        sorted_continua = sorted(cont_max_dict.items(), key=lambda item: item[1], reverse=True)

        # Schritt 3: Extrahiere die drei höchsten Werte (falls vorhanden)
        top_three_continua = {
            "best": sorted_continua[0][0] if len(sorted_continua) > 0 else None,
            "second_best": sorted_continua[1][0] if len(sorted_continua) > 1 else None,
            "third_best": sorted_continua[2][0] if len(sorted_continua) > 2 else None
        }

        # Speichere die Ergebnisse in line_best_continua_dict
        line_best_continua_dict[line] = top_three_continua

    return line_best_continua_dict


def get_weighted_best_continua(line_best_continua_dict, weights=None):
    # Initialisiere ein Wörterbuch zur Gewichtung der continua
    if weights is None:
        weights = {"best": 3, "second_best": 2, "third_best": 1}
    continua_scores = {}

    # Schritt 1: Berechne die gewichteten Punkte
    for line, ranks in line_best_continua_dict.items():
        for rank, continua_name in ranks.items():
            if continua_name is not None:
                # Addiere die gewichteten Punkte für jedes continua
                continua_scores[continua_name] = continua_scores.get(continua_name, 0) + weights[rank]

    # Schritt 2: Sortiere die continua_scores nach den Gesamtpunkten
    sorted_scores = sorted(continua_scores.items(), key=lambda item: item[1], reverse=True)

    # Schritt 3: Extrahiere die besten drei continua
    top_three_overall = {
        "best": sorted_scores[0][0] if len(sorted_scores) > 0 else None,
        "second_best": sorted_scores[1][0] if len(sorted_scores) > 1 else None,
        "third_best": sorted_scores[2][0] if len(sorted_scores) > 2 else None
    }

    return top_three_overall


def sort_1d_corr_data_for_lines(galaxy_campaigns_dict):
    line_sorted_data = dict()

    for campaign, continua_dict in galaxy_campaigns_dict.items():
        line_data_dict = dict()
        for continua, data_dict in continua_dict.items():
            x_values = data_dict["time shift (tau)"]
            for line, data_points in data_dict.items():
                if line != "time shift (tau)":
                    if line in line_data_dict.keys():
                        line_data_dict[line].append((continua, x_values, data_points))
                    else:
                        line_data_dict[line] = [(continua, x_values, data_points)]

        line_sorted_data[campaign] = line_data_dict

    return line_sorted_data


def calc_error_of_function(func, params, errors):
    """
    Calculates the propagated error for a given function and its input parameters.

    Args:
        func (callable): The function to evaluate. It should take parameters as input.
        params (list or np.array): List or array of input parameter values.
        errors (list or np.array): List or array of errors associated with the input parameters.

    Returns:
        float: Propagated error in the result.
    """
    # Convert params and errors to numpy arrays for easy manipulation
    params = np.array(params)
    errors = np.array(errors)

    # Numerical partial derivatives with respect to each parameter
    partial_derivatives = np.zeros_like(params, dtype=float)
    delta = 1e-8  # Small increment for numerical differentiation

    for i in range(len(params)):
        params_step = params.copy()
        params_step[i] += delta
        partial_derivatives[i] = (func(*params_step) - func(*params)) / delta

    # Propagated error calculation
    propagated_error = np.sqrt(np.sum((partial_derivatives * errors) ** 2))
    return propagated_error


def prepare_cut_data(fits_data_H_Alpha, fits_data_H_Beta, output_path):
    H_Alpha_wavelenghts = np.array(fits_data_H_Alpha['avg_HAlpha_Line_Profile.fits']['x_axis'][0])
    H_Alpha_avg_data = np.array(fits_data_H_Alpha['avg_HAlpha_Line_Profile.fits']['data'][0])
    H_Alpha_rms_data = np.array(fits_data_H_Alpha['rms_HAlpha_Line_Profile.fits']['data'][0])
    H_Beta_wavelenghts = np.array(fits_data_H_Beta['avg_HBeta_Line_Profile.fits']['x_axis'][0])
    H_Beta_avg_data = np.array(fits_data_H_Beta['avg_HBeta_Line_Profile.fits']['data'][0])
    H_Beta_rms_data = np.array(fits_data_H_Beta['rms_HBeta_Line_Profile.fits']['data'][0])
    H_Alpha_avg_velocity, H_Alpha_avg_intensity = transform_wavelength_to_velocity_and_cut(H_Alpha_wavelenghts,
                                                                                           H_Alpha_avg_data,
                                                                                           "HAlpha",
                                                                                           (-20000, 20000),
                                                                                           output_path / "H_Alpha_AVG_Line_Profile.txt")
    H_Alpha_rms_velocity, H_Alpha_rms_intensity = transform_wavelength_to_velocity_and_cut(H_Alpha_wavelenghts,
                                                                                           H_Alpha_rms_data, "HAlpha",
                                                                                           (-20000, 20000),
                                                                                           output_path / "H_Alpha_RMS_Line_Profile.txt")
    H_Beta_avg_velocity, H_Beta_avg_intensity = transform_wavelength_to_velocity_and_cut(H_Beta_wavelenghts,
                                                                                         H_Beta_avg_data, "HBeta",
                                                                                         (-20000, 20000),
                                                                                         output_path / "H_Beta_AVG_Line_Profile.txt")
    H_Beta_rms_velocity, H_Beta_rms_intensity = transform_wavelength_to_velocity_and_cut(H_Beta_wavelenghts,
                                                                                         H_Beta_rms_data, "HBeta",
                                                                                         (-20000, 20000),
                                                                                         output_path / "H_Beta_RMS_Line_Profile.txt")
    line_profile_dict = import_line_profile_data(normalized=True)
    line_profile_dict_add = {"avg":
                                 {"HAlpha_substracted_first":
                                      {"data_dict":
                                           {'velocity space (km/s)':
                                                H_Alpha_avg_velocity,
                                            'normalized flux': H_Alpha_avg_intensity}
                                       },
                                  "HBeta_substracted_first":
                                      {"data_dict":
                                           {'velocity space (km/s)':
                                                H_Beta_avg_velocity,
                                            'normalized flux':
                                                H_Beta_avg_intensity}}},
                             "rms":
                                 {"HAlpha_substracted_first":
                                      {"data_dict":
                                           {'velocity space (km/s)':
                                                H_Alpha_rms_velocity,
                                            'normalized flux': H_Alpha_rms_intensity}},
                                  "HBeta_substracted_first":
                                      {"data_dict":
                                           {'velocity space (km/s)':
                                                H_Beta_rms_velocity,
                                            'normalized flux':
                                                H_Beta_rms_intensity}}},
                             }
    merged_dict = merge_dicts(line_profile_dict, line_profile_dict_add)
    return merged_dict


def merge_dicts(d1, d2):
    """
    Rekursive Funktion zum Zusammenführen zweier geschachtelter Dictionaries.
    Falls Werte existieren, werden sie beibehalten oder überschrieben, falls notwendig.
    """
    for key, value in d2.items():
        if key in d1:
            if isinstance(d1[key], dict) and isinstance(value, dict):
                merge_dicts(d1[key], value)  # Rekursiver Aufruf für geschachtelte Dicts
            else:
                d1[key] = value  # Überschreiben, falls kein Dict
        else:
            d1[key] = value  # Falls Key nicht existiert, direkt übernehmen
    return d1


def print_table_for_one_reference(filename, linelist, continuum, include_mass=True):
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
            tau_cent_str = f"{line.tau_cent:.1f} \\ensuremath{{_{{{line.tau_cent_err[0] - line.tau_cent:.2f}}}^{{+{line.tau_cent_err[1] - line.tau_cent:.2f}}}}}"
            tau_peak_str = f"{line.tau_peak:.1f} \\ensuremath{{_{{{line.tau_peak_err[0] - line.tau_peak:.1f}}}^{{+{line.tau_peak_err[1] - line.tau_peak:.1f}}}}}"
            name = line.name.replace('_', r'\_')

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

    original_name = name
    # Escape für LaTeX
    name = name.replace('_', r'\_') if as_latex else name.replace('_', ' ')

    # Continuum?
    if "Cont" in name:
        is_not_calibrated = ("not\\_optical\\_calibrated" in name if as_latex else "not_optical_calibrated" in original_name)
        num_part = ''.join(filter(str.isdigit, name))
        label = f"Cont. {num_part}" if num_part else name
        if is_not_calibrated:
            label += " (not opt. calib.)"
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
        "LyAlpha": r"Lyman $\alpha$",
    }

    replacements_plot = {
        "HAlpha": r"Hα",
        "HBeta": r"Hβ",
        "HGamma": r"Hγ",
        "HDelta": r"Hδ",
        "HeI5875": "He I 5875",
        "HeI7065": "He I 7065",
        "HeI4471": "He I 4471",
        "HeI5015": "He I 5015",
        "HeII4685": "He II 4685",
        "OI8446": "O I 8446",
        "LyAlpha": "Lyman α",
    }

    replacements = replacements_latex if as_latex else replacements_plot

    formatted = replacements.get(base_name, base_name)

    if is_not_calibrated:
        formatted += " (not opt. calib.)"

    return formatted