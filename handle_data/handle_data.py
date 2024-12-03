import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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


def gaussian_with_baseline(x, a, x0, sigma, c):
    """Berechnet eine Gauß-Funktion mit Baseline."""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + c


def calc_time_lag_of_line(line_name, continuum_x_y_tuple_list, baseline_tolerance=0.1, window_methode="gradient",
                          lag_method="gaussfit"):
    """
    Passt eine Gauß-Funktion oder verwendet den Centroid zur Bestimmung des Time Lags.

    Parameters:
        line_name (str): Der Name der Linie.
        continuum_x_y_tuple_list (list): Liste von (continuum, x, y)-Tupeln.
        baseline_tolerance (float): Toleranzbereich für die Baseline.
        window_methode (str): Methode zur Auswahl des Fit-Fensters ("minima", "turningpoints" oder "gradient").
        lag_method (str): Methode zur Berechnung des Time Lags ("gaussfit" oder "centroid").

    Returns:
        list: Ergebnisse mit Fit-Parametern und zusätzlichen Informationen.
    """
    fit_window_func = get_fit_window_function(window_methode)
    results = []

    for continuum, x, y in continuum_x_y_tuple_list:
        x, y = np.asarray(x).flatten(), np.asarray(y).flatten()

        max_index = get_largest_peak_index(y)
        if max_index is None:
            results.append(create_error_result(line_name, continuum, x, y, "Keine Maxima gefunden."))
            continue

        try:
            left, right, x_fit, y_fit = fit_window_func(max_index, x, y)

            if lag_method == "gaussfit":
                result = perform_gaussian_fit(line_name, continuum, x, y, x_fit, y_fit, max_index, baseline_tolerance,
                                              left, right)
            elif lag_method == "centroid":
                result = perform_centroid_calculation(line_name, continuum, x, y, x_fit, y_fit, left, right)
            else:
                raise ValueError(f"Ungültige Methode zur Berechnung des Time Lags: {lag_method}")

            results.append(result)
        except ValueError as e:
            results.append(create_error_result(line_name, continuum, x, y, str(e)))

    return results


def get_fit_window_function(window_methode):
    """
    Wählt die passende Funktion für die Fit-Fenster-Berechnung aus.

    Parameters:
        window_methode (str): Methode zur Auswahl des Fit-Fensters ("minima", "turningpoints", "gradientchange").

    Returns:
        function: Die passende Funktion zur Fensterberechnung.
    """
    if window_methode == "minima":
        return fit_window_with_minima
    elif window_methode == "turningpoints":
        return fit_window_with_turningpoints
    elif window_methode == "gradient":
        return fit_window_with_gradient_change
    else:
        raise ValueError(f"Ungültige Fit-Methode: {window_methode}")


def perform_centroid_calculation(line_name, continuum, x, y, x_fit, y_fit, left, right):
    """Berechnet den Time Lag basierend auf dem Centroid."""
    centroid = calculate_centroid(x_fit, y_fit)
    if centroid is None:
        return create_error_result(line_name, continuum, x, y, "Centroid konnte nicht berechnet werden.")

    return {
        "line_name": line_name,
        "continuum": continuum,
        "time_lag": centroid,
        "time_lag_error": None,  # Fehlerabschätzung könnte hier implementiert werden
        "fit_window_start": x[left],
        "fit_window_end": x[right],
        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "fit_function": "Centroid",
        "fit_success": True,
    }


def calculate_centroid(x, y):
    """Berechnet den Centroid eines Peaks."""
    weighted_sum = np.sum(x * y)
    total_intensity = np.sum(y)
    return weighted_sum / total_intensity if total_intensity > 0 else None


def perform_gaussian_fit(line_name, continuum, x, y, x_fit, y_fit, max_index, baseline_tolerance, left, right):
    """Führt den Curve-Fit mit einer Gauß-Funktion durch."""
    baseline = np.min(y)
    c_min, c_max = baseline - baseline_tolerance, baseline + baseline_tolerance
    initial_guess = [y[max_index] - baseline, x[max_index], 1.0, baseline]
    bounds = ([0, x_fit.min(), 0, c_min], [np.inf, x_fit.max(), np.inf, c_max])

    try:
        popt, pcov = curve_fit(gaussian_with_baseline, x_fit, y_fit, p0=initial_guess, bounds=bounds)
        return create_success_result(line_name, continuum, popt, np.sqrt(np.diag(pcov)), x, y, left, right)
    except RuntimeError:
        return create_error_result(line_name, continuum, x, y, "Fit konnte nicht durchgeführt werden.")


def get_largest_peak_index(y):
    """Findet den Index des größten lokalen Maximums."""
    peak_indices, _ = find_peaks(y)
    return peak_indices[np.argmax(y[peak_indices])] if peak_indices.size > 0 else None


def fit_window_with_gradient_change(max_index, x, y, threshold=0.055, min_distance=1):
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


def fit_window_with_minima(max_index, x, y):
    """Bestimmt das Fit-Fenster basierend auf lokalen Minima."""
    min_indices, _ = find_peaks(-y)
    left = min_indices[min_indices < max_index][-1] if any(min_indices < max_index) else 0
    right = min_indices[min_indices > max_index][0] if any(min_indices > max_index) else len(x) - 1
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


def create_success_result(line_name, continuum, popt, errors, x, y, left, right):
    """Erstellt ein Ergebnis bei erfolgreichem Fit."""
    a, x0, sigma, c = popt
    a_err, x0_err, sigma_err, c_err = errors
    return {
        "line_name": line_name,
        "continuum": continuum,
        "amplitude": a,
        "amplitude_error": a_err,
        "time_lag": x0,
        "time_lag_error": x0_err,
        "std_dev": sigma,
        "std_dev_error": sigma_err,
        "baseline": c,
        "baseline_error": c_err,
        "fit_window_start": x[left],
        "fit_window_end": x[right],
        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "fit_function": "a * exp(-((x - x0)^2) / (2 * sigma^2)) + c",
        "fit_success": True,
    }


def create_error_result(line_name, continuum, x, y, error_message):
    """Erstellt ein Ergebnis bei fehlgeschlagenem Fit."""
    return {
        "line_name": line_name,
        "continuum": continuum,
        "x_values": x.tolist(),
        "y_values": y.tolist(),
        "error": error_message,
        "fit_success": False,
    }


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
