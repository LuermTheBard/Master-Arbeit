import sys

import numpy as np
import toml
from scipy.signal import find_peaks


def calc_time_lag_of_line(line_name, continuum_x_y_tuple_list,  window_methode="gradient"):
    """
    Passt eine Gauß-Funktion oder verwendet den Centroid zur Bestimmung des Time Lags.

    Parameters:
        line_name (str): Der Name der Linie.
        continuum_x_y_tuple_list (list): Liste von (continuum, x, y)-Tupeln.
        window_methode (str): Methode zur Auswahl des Fit-Fensters ("minima", "turningpoints" oder "gradient").

    Returns:
        list: Ergebnisse mit Fit-Parametern und zusätzlichen Informationen.
    """
    fit_window_func = get_fit_window_function(window_methode)
    results = []

    print(f"Calculate time Lag for line: {line_name}")

    for continuum, x, y in continuum_x_y_tuple_list:
        x, y = np.asarray(x).flatten(), np.asarray(y).flatten()

        max_index = get_largest_peak_index(y)
        if max_index is None:
            continue

        try:
            left, right, x_fit, y_fit = fit_window_func(max_index, x, y)

            # Fenster extrahieren
            x_fit = x[left:right + 1]
            y_fit = y[left:right + 1]

            result = perform_centroid_calculation(line_name, continuum, x, y, x_fit, y_fit, left, right)
            results.append(result)
        except ValueError as e:
            results.append(create_error_result(line_name, continuum, x, y, str(e)))

    return results


def get_fit_window_function(window_methode):
    """
    Wählt die passende Funktion für die Fit-Fenster-Berechnung aus.

    Parameters: window_methode (str): Methode zur Auswahl des Fit-Fensters ("minima", "turningpoints",
    "gradientchange", "percentage").

    Returns:
        function: Die passende Funktion zur Fensterberechnung.
    """
    if window_methode == "minima":
        return fit_window_with_minima
    elif window_methode == "turningpoints":
        return fit_window_with_turningpoints
    elif window_methode == "gradient":
        return fit_window_with_gradient_change
    elif window_methode == "percentage":
        return fit_window_with_percentage
    else:
        raise ValueError(f"Ungültige Fit-Methode: {window_methode}")


def perform_centroid_calculation(line_name, continuum, x, y, x_fit, y_fit, left, right):
    """Berechnet den Time Lag basierend auf dem Centroid."""
    centroid, centroid_err = calculate_centroid(x_fit, y_fit)
    if centroid is None:
        return create_error_result(line_name, continuum, x, y, "Centroid konnte nicht berechnet werden.")

    print(f"Calculation of time lag for {line_name}/{continuum} successfull: {centroid} +- {centroid_err}")

    return {
        "line_name": line_name,
        "continuum": continuum,
        "time_lag": centroid,
        "time_lag_error": centroid_err,
        "fit_window_start": x[left],
        "fit_window_end": x[right],
        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "fit_function": "Centroid",
        "fit_success": True,
    }


def calculate_centroid(x, y):
    """
    Berechnet den Centroid eines Peaks und schätzt den Fehler des Centroids.

    Parameters:
        x (array-like): Die x-Werte der Daten.
        y (array-like): Die y-Werte der Daten.

    Returns:
        centroid (float or None): Der berechnete Centroid des Peaks.
        centroid_error (float or None): Der geschätzte Fehler des Centroids.
    """
    weighted_sum = np.sum(x * y)
    total_intensity = np.sum(y)

    if total_intensity <= 0:
        return None, None

    centroid = weighted_sum / total_intensity

    # Fehlerberechnung basierend auf der Standardabweichung der gewichteten x-Werte
    variance = np.sum(y * (x - centroid) ** 2) / total_intensity
    centroid_error = np.sqrt(variance / len(x))

    return centroid, centroid_error


def get_largest_peak_index(y):
    """Findet den Index des größten lokalen Maximums."""
    peak_indices, _ = find_peaks(y)
    return peak_indices[np.argmax(y[peak_indices])] if peak_indices.size > 0 else None


def fit_window_with_percentage(max_index, x, y, percentage=20):
    y_max = max(y)
    y_min = min(y)
    peak_percentage = (y_max - y_min) * percentage / 100
    y_threshold = y_max - peak_percentage

    # Finde den Index des y-Maximums
    y_max_index = np.argmax(y)

    # Falls der angegebene max_index anders ist, benutze diesen
    if y_max_index != max_index:
        y_max_index = max_index

    # Suche nach links vom Maximum
    left_index = y_max_index
    while left_index > 0 and y[left_index] > y_threshold:
        left_index -= 1

    # Suche nach rechts vom Maximum
    right_index = y_max_index
    while right_index < len(y) - 1 and y[right_index] > y_threshold:
        right_index += 1

    # Sicherstellen, dass es gültige Indizes gibt
    if left_index < 0:
        left_index = 0
    if right_index >= len(x):
        right_index = len(x) - 1

    # Finde die x-Werte, die links und rechts gefunden wurden
    left = left_index if left_index >= 0 and y[left_index] <= y_threshold else None
    right = right_index if right_index < len(y) and y[right_index] <= y_threshold else None

    # Sicherstellen, dass left und right gültige Indizes haben
    if left is None:
        left = 0
    if right is None:
        right = len(x) - 1

    # Suche nach möglichst gleichen y-Werten für left und right, die links und rechts vom Maximum sind
    best_left = left
    best_right = right
    min_diff = abs(y[left] - y[right])

    # Suche links vom Maximum für den bestmöglichen left Index
    for i in range(left, y_max_index):
        if abs(y[i] - y[right]) < min_diff:
            best_left = i
            min_diff = abs(y[i] - y[right])

    # Suche rechts vom Maximum für den bestmöglichen right Index
    for i in range(y_max_index, right + 1):
        if abs(y[left] - y[i]) < min_diff:
            best_right = i
            min_diff = abs(y[left] - y[i])

    # Aktualisiere die endgültigen Indizes
    left = best_left
    right = best_right

    # Fenster extrahieren
    return extract_fit_window(left, right, x, y)


def fit_window_with_gradient_change(max_index, x, y, threshold=0.07, min_distance=1):
    """
    Bestimmt das Fit-Fenster basierend auf signifikanten positiven Änderungen im Gradienten.

    Parameters:
        max_index (int): Index des Maximums.
        x (array-like): Array der x-Werte.
        y (array-like): Array der y-Werte.
        threshold (float): Schwellenwert für signifikante Änderungen im Gradienten.
        min_distance (int): Mindestabstand zwischen Maximum und gefundenen Punkten.

    Returns:
        tuple: (left, right, x_fit_data, y_fit_data)
    """
    # Berechnung der ersten Ableitung (Gradient)
    dy = np.gradient(y, x)

    # Berechnung der Differenzen des Gradienten
    gradient_change = np.diff(dy)

    # Nur positive Änderungen des Gradienten berücksichtigen
    positive_changes = np.where(gradient_change > threshold)[0] + 1  # +1, da diff die Länge um 1 reduziert

    # Linken Punkt bestimmen
    left_candidates = positive_changes[positive_changes < max_index]
    left = left_candidates[-1] if len(left_candidates) > 0 and max_index - left_candidates[-1] >= min_distance else 0

    # Rechten Punkt bestimmen
    right_candidates = positive_changes[positive_changes > max_index]
    right = right_candidates[0] if len(right_candidates) > 0 and right_candidates[
        0] - max_index >= min_distance else len(x) - 1

    # Fenster extrahieren
    return extract_fit_window(left, right, x, y)


def fit_window_with_minima(max_index, x, y, max_ratio=1.5):
    """Bestimmt das Fit-Fenster basierend auf lokalen Minima."""
    min_indices, _ = find_peaks(-y)

    # Linken Punkt bestimmen
    left_candidates = min_indices[min_indices < max_index]
    left = left_candidates[-1] if len(left_candidates) > 0 else 0

    # Rechten Punkt bestimmen
    right_candidates = min_indices[min_indices > max_index]
    right = right_candidates[0] if len(right_candidates) > 0 else len(x) - 1

    # Bedingung einbauen, wenn einer der beiden Kandidaten deutlich näher am Peak ist als der andere
    left_distance = max_index - left
    right_distance = right - max_index

    if (left_distance > 0 and right_distance > 0 and
            max(left_distance, right_distance) / min(left_distance, right_distance) > max_ratio):
        if left_distance > right_distance:
            # Verwende das nächste Minimum rechts vom ersten Kandidaten
            right = right_candidates[1] if len(right_candidates) > 1 else right
        else:
            # Verwende das nächste Minimum links vom ersten Kandidaten
            left = left_candidates[-2] if len(left_candidates) > 1 else left

    return extract_fit_window(left, right, x, y)


def fit_window_with_turningpoints(max_index, x, y, min_distance=1):
    """
    Bestimmt das Fit-Fenster basierend auf Wendepunkten und stellt sicher,
    dass Wendepunkte nicht zu nah am Maximum liegen.

    Parameters:
        max_index (int): Index des Maximums.
        x (array-like): Array der x-Werte.
        y (array-like): Array der y-Werte.
        min_distance (int): Mindestabstand zwischen Maximum und Wendepunkt.

    Returns:
        tuple: (left, right, x_fit_data, y_fit_data)
    """
    dy, d2y = np.gradient(y, x), np.gradient(np.gradient(y, x), x)
    zero_crossings = np.where(np.diff(np.sign(d2y)))[0]

    # Linken Wendepunkt auswählen
    left_candidates = zero_crossings[zero_crossings < max_index]
    left = left_candidates[-1] if len(left_candidates) > 0 and max_index - left_candidates[-1] >= min_distance else (
        left_candidates[-2] if len(left_candidates) > 1 else 0)

    # Rechten Wendepunkt auswählen
    right_candidates = zero_crossings[zero_crossings > max_index]
    right = right_candidates[0] if len(right_candidates) > 0 and right_candidates[0] - max_index >= min_distance else (
        right_candidates[1] if len(right_candidates) > 1 else len(x) - 1)

    # Fenster extrahieren
    return extract_fit_window(left, right, x, y)


def extract_fit_window(left, right, x, y):
    """Extrahiert das Fit-Fenster zwischen zwei Indizes."""
    fit_mask = (x >= x[left]) & (x <= x[right])
    return left, right, x[fit_mask], y[fit_mask]


def calculate_time_lags_for_continuum(results, continuum_name):
    """
    Gibt die Time Lag Ergebnisse für ein bestimmtes Continuum aus den Ergebnissen zurück.

    Parameters:
        results (list): Die Ergebnisse, die von der calc_time_lag_of_line-Methode zurückgegeben wurden.
        continuum_name (str): Der Name des gewünschten Continuums.

    Returns:
        continuum_results (list): Eine Liste mit den Time Lag Ergebnissen für das angegebene Continuum.
        skipped_result (str): Informationen zu übersprungenen Ergebnissen.
    """
    continuum_results = []
    skipped_result = ""

    for result in results:
        if result.get("continuum") == continuum_name:
            if result.get("fit_success", False):
                time_lag = result.get("time_lag")
                time_lag_error = result.get("time_lag_error")
                if (time_lag is not None and time_lag_error is not None and not np.isnan(time_lag)
                        and not np.isnan(time_lag_error)):
                    continuum_results.append(result)
                else:
                    skipped_result += (f"Result of {result.get('continuum', 'Unknown')} and "
                                       f"{result.get('line_name', 'Unknown')} was skipped, because of invalid values\n")

    if not continuum_results:
        print(f"Keine gültigen Time Lag Ergebnisse für das Continuum '{continuum_name}' gefunden.")
        return None, skipped_result

    return continuum_results, skipped_result


def save_lag_results_to_toml(results, file_path="results.toml"):
    """
    Speichert die Ergebnisse der ersten Methode in einer TOML-Datei.

    Parameters:
        results (list): Die Ergebnisse, die von der calc_time_lag_of_line-Methode zurückgegeben wurden.
        file_path (str): Der Pfad zur TOML-Datei, in die die Ergebnisse gespeichert werden sollen.
    """
    toml_data = {}

    for result in results:
        continuum = result.get("continuum", "N/A")

        # Konvertiere alle np.float Werte zu normalen float Werten
        result = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in result.items()}

        if result.get("fit_success", False):
            toml_data[continuum] = {
                key: result.get(key, "N/A") for key in [
                    "amplitude", "amplitude_error", "time_lag", "time_lag_error",
                    "std_dev", "std_dev_error", "baseline", "baseline_error",
                    "fit_window_start", "fit_window_end", "fit_function"]
            }
            toml_data[continuum]["fit_success"] = True
        else:
            toml_data[continuum] = {
                "error": result.get("error", "N/A"),
                "fit_success": False,
            }

    with open(file_path, "w") as file:
        toml.dump(toml_data, file)

    print(f"Ergebnisse wurden in {file_path} gespeichert.")


def create_error_result(line_name, continuum, x, y, error_message):
    """Erstellt ein Ergebnis bei fehlgeschlagenem Fit."""

    print(f"Calculation of {line_name}/{continuum} was not succesfull: {error_message}", file=sys.stderr)
    return {
        "line_name": line_name,
        "continuum": continuum,
        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "error": error_message,
        "fit_success": False,
    }
