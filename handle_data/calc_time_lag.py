import numpy as np
from matplotlib import pyplot as plt


def plot_ccf_with_centroid(x_values, y_values, x_selected, y_selected, centroid, baseline, threshold):
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

    plt.xlabel('Time Shift (tau)')
    plt.ylabel('CCF Amplitude')
    plt.title('CCF Curve with Centroid Region, Baseline, and Threshold')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_time_lag_and_err(x_values, y_values, baseline=None, plot=False):
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
    centroid_error = estimate_centroid_error(x_selected, y_selected)

    if plot:
        plot_ccf_with_centroid(x_values, y_values, x_selected, y_selected, centroid, baseline, threshold)

    return {
        "time_lag": centroid,
        "time_lag_err": centroid_error,
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
                result_dict = get_time_lag_from_line(cont_data, baseline=baseline, plot=plot)
                campaign_cont_result_dict[campaign][name] = result_dict

    return campaign_cont_result_dict


def get_time_lag_from_line(cont_data, baseline=None, plot=False):


    x_values = cont_data.pop('time shift (tau)')

    line_time_lag_dict = dict()

    for line, y_values in cont_data.items():

        result_dict = calculate_time_lag_and_err(x_values, y_values, baseline=baseline, plot=plot)

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


def estimate_centroid_error(x_selected, y_selected, num_samples=1000, noise_std=0.1):
    """
    Berechnet den Fehler des Centroids mit Monte-Carlo-Simulation.

    Parameters:
    x_selected (array-like): Die x-Werte oberhalb der Schwelle.
    y_selected (array-like): Die y-Werte oberhalb der Schwelle.
    num_samples (int, optional): Anzahl der Monte-Carlo-Simulationen. Standard: 1000.
    noise_std (float, optional): Standardabweichung des zufälligen Rauschens. Standard: 0.1.

    Returns:
    float: Standardabweichung des Centroids aus Monte-Carlo-Simulation.
    """
    if len(x_selected) == 0 or np.sum(y_selected) == 0:
        return np.nan

    centroid_samples = []

    for _ in range(num_samples):
        # Erzeuge gestörte y-Werte durch zufälliges Rauschen
        y_noisy = y_selected + np.random.normal(0, noise_std * np.abs(y_selected), size=len(y_selected))

        # Berechne den Centroid für die gestörten Werte
        noisy_centroid = np.sum(x_selected * y_noisy) / np.sum(y_noisy)
        centroid_samples.append(noisy_centroid)

    # Berechne den Standardfehler des Centroids aus den Monte-Carlo-Simulationen
    centroid_error = np.std(centroid_samples)

    return centroid_error











