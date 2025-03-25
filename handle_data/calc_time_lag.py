from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from Malte_get_BH_mass import Line


def plot_ccf_with_centroid(x_values, y_values, x_selected, y_selected, centroid, baseline, threshold, line_name, cont_name):
    """
    Plottet die CCF-Kurve und markiert den Bereich, in dem der Centroid berechnet wurde.

    Parameters:
    x_values (array-like): Die x-Werte der CCF.
    y_values (array-like): Die y-Werte der CCF.
    x_selected (array-like): Die ausgewählten x-Werte für die Centroid-Berechnung.
    y_selected (array-like): Die entsprechenden y-Werte.
    centroid (float): Der berechnete Centroid.
    baseline (float): Die verwendete Baseline.
    threshold (float): Der Schwellenwert für die Centroid-Berechnung.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label='CCF Curve', color='blue')

    # Markiere die nicht ausgewählten Punkte in Schwarz
    x_unselected = np.setdiff1d(x_values, x_selected)
    y_unselected = [y_values[np.where(x_values == x)[0][0]] for x in x_unselected]
    plt.scatter(x_unselected, y_unselected, color='black', label='Unselected Points')

    plt.scatter(x_selected, y_selected, color='red', label='Selected Points')
    plt.axvline(x_selected[0], color='green', linestyle='--', label='Centroid Region')
    plt.axvline(x_selected[-1], color='green', linestyle='--')
    plt.axvline(centroid, color='purple', linestyle='-', label='Centroid')
    plt.axhline(baseline, color='orange', linestyle=':', label='Baseline')
    plt.axhline(threshold, color='magenta', linestyle='-.', label='Threshold')

    # Den Centroid-Wert direkt an der x-Achse anzeigen
    plt.text(centroid, plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
             f'{centroid:.2f}', color='purple', fontsize=10, ha='center')

    plt.xlabel("Time Lag $\\tau$ [d]")
    plt.ylabel("Correlation Coefficient")
    plt.title(f'CCF between {line_name} and {cont_name}')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_time_lag_and_err(x_values, y_values, cont_name, line_name, baseline=None, plot=False):
    """
    Berechnet den Centroid und optional die CCF-Kurve mit markiertem Bereich plotten.

    Parameters:
    x_values (array-like): Die x-Werte.
    y_values (array-like): Die y-Werte.
    baseline (float, optional): Die Baseline für die Berechnung.
    plot (bool, optional): Falls True, wird die CCF-Kurve geplottet.

    Returns:
    dict: Enthält time_lag, time_lag_err, x_selected, y_selected
    """
    centroid, x_selected, y_selected, baseline, threshold = get_centroid_of_peak(x_values, y_values, baseline=baseline)
    #centroid_error = estimate_centroid_error(x_selected, y_selected)

    if plot:
        plot_ccf_with_centroid(x_values, y_values, x_selected, y_selected, centroid, baseline, threshold,line_name, cont_name)

    return {
        "time_lag": centroid,
    #    "time_lag_err": centroid_error,
        "x_selected": x_selected,
        "y_selected": y_selected
    }


def get_time_lags(campaign_ccf_data, baseline=None, selected_continua=None, plot=False):
    campaign_cont_result_dict = dict()

    for campaign, cont_data_dict in campaign_ccf_data.items():
        campaign_cont_result_dict[campaign] = dict()

        for name, cont_data in cont_data_dict.items():
            # Falls eine Liste von Continuen gegeben ist, nur die relevanten Namen speichern
            if selected_continua is None or name in selected_continua:
                result_dict = get_time_lag_from_line(cont_data, name, baseline=baseline, plot=plot)
                campaign_cont_result_dict[campaign][name] = result_dict

    return campaign_cont_result_dict


def get_time_lag_from_line(cont_data, cont_name, baseline=None, plot=False):


    x_values = cont_data.pop('time shift (tau)')

    line_time_lag_dict = dict()

    for line, y_values in cont_data.items():

        result_dict = calculate_time_lag_and_err(x_values, y_values, cont_name, line, baseline=baseline, plot=plot)

        line_time_lag_dict[line] = result_dict

    return line_time_lag_dict



def get_centroid_of_peak(x_values, y_values, baseline=None, threshold=0.8):
    """
    Berechnet den Centroid eines Peaks basierend auf den obersten 'threshold' % der y-Werte.
    Gibt zusätzlich die x- und y-Werte zurück, die für die Berechnung des Centroids genutzt wurden.

    Parameters:
    x_values (array-like): Die x-Werte der Daten.
    y_values (array-like): Die y-Werte der Daten.
    baseline (float, optional): Der Basiswert, von dem aus normalisiert wird. Standard: min(y_values).
    threshold (float, optional): Der Prozentsatz des Peaks, der berücksichtigt wird (0.0 - 1.0). Standard: 0.8.

    Returns:
    tuple: (centroid, x_selected, y_selected)
        - centroid (float): Der berechnete Centroid des Peaks.
        - x_selected (array-like): Die x-Werte oberhalb der Schwelle.
        - y_selected (array-like): Die y-Werte oberhalb der Schwelle.
    """
    y_values_max = max(y_values)

    if baseline is None:
        baseline = min(y_values)

    if baseline == 0:
        y_threshold = y_values_max * threshold
    else:
        y_threshold = baseline + threshold * (y_values_max - baseline)

    # Index des Maximums finden
    max_index = np.argmax(y_values)

    # Suche nach links die erste Stelle, wo y unter die Schwelle fällt
    left_index = max_index
    while left_index > 0 and y_values[left_index] >= y_threshold:
        left_index -= 1

    # Suche nach rechts die erste Stelle, wo y unter die Schwelle fällt
    right_index = max_index
    while right_index < len(y_values) - 1 and y_values[right_index] >= y_threshold:
        right_index += 1

    # Wähle nur die Werte aus, die innerhalb des Bereichs liegen
    x_selected = x_values[left_index + 1:right_index]
    y_selected = y_values[left_index + 1:right_index]

    # Falls keine Werte oberhalb der Schwelle gefunden wurden, gibt NaN zurück
    if len(x_selected) == 0 or np.sum(y_selected) == 0:
        print("Keine Werte oberhalb der Schwelle gefunden.")
        return np.nan, np.array([]), np.array([])

    # Berechne den gewichteten Mittelwert als Centroid
    centroid = np.sum(x_selected * y_selected) / np.sum(y_selected)

    return centroid, x_selected, y_selected, baseline, y_threshold


def printTable(filename, linelist, continuum):
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
        outfile.write(fr'\caption{{Centroid and Peak Time Lag for {continuum}.}}' + '\n')
        outfile.write(fr'\label{{tab:lags_{continuum}}}' + '\n')
        outfile.write(r'\begin{tabular}{l c c c}' + '\n')
        outfile.write(r'\toprule' + '\n')

        # Spaltenüberschriften
        outfile.write(r'Name & $\tau_{\text{cent}}$ [d] & $\tau_{\text{peak}}$ [d] & $M_{\text{BH}} [10^7 M_\odot]$ \\' + '\n')
        outfile.write(r'\midrule' + '\n')

        # Tabelleninhalt mit Fehlerdarstellung
        for line in linelist:
            tau_cent_str = f"{line.tau_cent:.1f} \\ensuremath{{_{{-{abs(line.tau_cent_err[0] - line.tau_cent):.1f}}}^{{+{abs(line.tau_cent_err[1] - line.tau_cent):.1f}}}}}"
            tau_peak_str = f"{line.tau_peak:.1f} \\ensuremath{{_{{-{abs(line.tau_peak_err[0] - line.tau_peak):.1f}}}^{{+{abs(line.tau_peak_err[1] - line.tau_peak):.1f}}}}}"
            mass_str = f"{line.M_Mo:.2f}_{{{line.M_Mo_err[0] - line.M_Mo:.2f}}}^{{+{line.M_Mo_err[1] - line.M_Mo:.2f}}}"

            outfile.write(f"{line.name} & ${tau_cent_str}$ & ${tau_peak_str}$ & ${mass_str}$ \\\\" + '\n')

        # Tabellenabschluss
        outfile.write(r'\bottomrule' + '\n')
        outfile.write(r'\end{tabular}' + '\n')
        outfile.write(r'\end{table}' + '\n\n')

        # LaTeX Dokument-Abschluss
        outfile.write(r'\end{document}' + '\n')




# Definiere die Default-Listen für FWHM (rms) und sigma (rms)
default_fwhm_rms = {
    'HAlpha': 3050.0, 'HBeta': 3160.0, 'HGamma': 3130.0, 'HDelta': 4940.0,
    'HeI5875': 1500.0, 'HeI7065': 1500.0, 'HeI4471': 1500.0, 'HeI5015': 1500.0,
    'HeII4685': 1500.0, 'OI8446': 1500.0
}

default_sigma_rms = {
    'HAlpha': 1180.0, 'HBeta': 1190.0, 'HGamma': 1240.0, 'HDelta': 1560.0,
    'HeI5875': 250.0, 'HeI7065': 250.0, 'HeI4471': 250.0, 'HeI5015': 250.0,
    'HeII4685': 250.0, 'OI8446': 250.0
}

def calc_centroid_malte_code(campaign, continuum):
    base_path = Path(
        rf'C:\Users\lukas\Desktop\Python\Master-Arbeit\data\campaigns\{campaign}\calc_time_lag_ccfs\{continuum}'
    )

    # Definiere die Liniennamen
    lines = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685', 'OI8446']

    # Lade die lineCorrelations-Daten
    line_correlations_path = base_path / "lineCorrelations_ICCF.txt"
    try:
        lineCorrelations = np.loadtxt(line_correlations_path)
    except Exception as e:
        print(f"❌ Fehler beim Laden der Datei {line_correlations_path}: {e}")
        return

    # Lade alle Zentroid- und Peak-Daten in ein Dictionary
    data = {}
    for line in lines:
        try:
            cents_path = base_path / f"calculatedCentroids{continuum}_{line}_ICCF.txt"
            peaks_path = base_path / f"peakDistribution_{continuum}_{line}_ICCF.txt"

            # Falls eine Datei fehlt, wird sie übersprungen
            data[line] = {
                "centroids": np.loadtxt(cents_path) if cents_path.exists() else None,
                "peaks": np.loadtxt(peaks_path) if peaks_path.exists() else None
            }
        except Exception as e:
            print(f"⚠️ Warnung: Fehler beim Laden der Dateien für {line}: {e}")
            continue  # Statt `return`, damit andere Linien geladen werden

    # Index-Mapping für die Spalten von lineCorrelations
    index_map = {
        'HDelta': 1, 'HGamma': 2, 'HeII4685': 3, 'HBeta': 4,
        'HeI5875': 5, 'HAlpha': 6, 'HeI5015': 7, 'OI8446': 8,
        'HeI4471': 9, 'HeI7065': 10, 'OIII5007': 11
    }

    # Erstelle die Line-Objekte
    line_objects = []
    for line in lines:
        if line in index_map and data[line]["centroids"] is not None and data[line]["peaks"] is not None:
            line_obj = Line(
                line,  # Name der Linie
                default_fwhm_rms.get(line, 0.0),  # FWHM (rms)
                default_sigma_rms.get(line, 0.0),  # Sigma (rms)
                np.vstack((lineCorrelations[:, 0], lineCorrelations[:, index_map[line]])).T,
                data[line]["centroids"],
                data[line]["peaks"]
            )
            line_objects.append(line_obj)

    # Speichere die Ergebnisse mit dynamischem Dateinamen
    output_filename = f'CCF_lags_{campaign}_{continuum}.tex'
    print(f"✅ Speichere Ergebnisse in Datei: {output_filename}")
    printTable(output_filename, line_objects, continuum)






#calc_centroid_malte_code("NGC4593_Full_Line", "Cont1150")
#calc_centroid_malte_code("NGC4593_Line_Center", "Cont1150")
#calc_centroid_malte_code("NGC4593_Full_Line", "Cont5100")
#calc_centroid_malte_code("NGC4593_Line_Center", "Cont5100")
