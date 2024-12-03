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
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + c


def calc_time_lag_of_line(line_name, continuum_x_y_tuple_list, baseline_tolerance=0.1):
    """
    Passt eine Gauß-Funktion mit einer variablen Baseline (innerhalb eines Bereichs) an das größte Maximum der gegebenen Daten
    für jede Datenreihe an und berechnet Fit-Parameter, Time Lag und Fehler.

    Parameters:
        line_name (str): Der Name der Linie.
        continuum_x_y_tuple_list (list): Eine Liste von Tupeln, wobei jedes Tupel aus
                                         (continuum, x, y) besteht:
                                         - continuum: Name des Kontinuums.
                                         - x: Array der x-Werte.
                                         - y: Array der y-Werte.
        baseline_tolerance (float): Toleranzbereich für die Baseline. Die Baseline kann in
                                    [min(y) - tolerance, min(y) + tolerance] optimiert werden.

    Returns:
        list: Eine Liste von Dictionaries mit Fit-Parametern, Time Lag, Fehlern und weiteren Informationen.
    """


    results = []

    for continuum_x_y_tuple in continuum_x_y_tuple_list:
        continuum, x, y = continuum_x_y_tuple

        # Sicherstellen, dass x und y als 1D-Arrays vorliegen
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Lokale Maxima und Minima finden
        peak_indices, _ = find_peaks(y)
        min_indices, _ = find_peaks(-y)  # Minima finden durch Invertieren von y

        if len(peak_indices) == 0:
            results.append({
                "line_name": line_name,
                "continuum": continuum,
                "x_values": x.tolist(),
                "y_values": y.tolist(),
                "error": "Keine Maxima gefunden.",
                "fit_success": False,
            })
            continue

        # Größtes lokales Maximum auswählen
        max_index = peak_indices[np.argmax(y[peak_indices])]
        max_x = x[max_index]
        max_y = y[max_index]

        #left_min, right_min, x_fit_data, y_fit_data = fit_window_with_minima(max_index, min_indices, x, y)
        left_min, right_min, x_fit_data, y_fit_data = fit_window_with_turningpoints(max_index, x, y)

        # Baseline bestimmen und Toleranzbereich festlegen
        baseline = np.min(y)
        c_min = baseline - baseline_tolerance
        c_max = baseline + baseline_tolerance

        # Initiale Schätzwerte für den Fit
        initial_guess = [max_y - baseline, max_x, 1.0, baseline]  # [Amplitude, Mittelwert, Standardabweichung, Baseline]
        bounds = ([0, x_fit_data.min(), 0, c_min],  # Untergrenzen für die Parameter
                  [np.inf, x_fit_data.max(), np.inf, c_max])  # Obergrenzen für die Parameter

        # Curve-Fitting durchführen
        try:
            popt, pcov = curve_fit(gaussian_with_baseline, x_fit_data, y_fit_data, p0=initial_guess, bounds=bounds)
            a, x0, sigma, c = popt
            errors = np.sqrt(np.diag(pcov))
            a_err, x0_err, sigma_err, c_err = errors

            fit_result = {
                "line_name": line_name,
                "continuum": continuum,
                "amplitude": a,
                "amplitude_error": a_err,
                "time_lag": x0,  # mean als time_lag
                "time_lag_error": x0_err,  # Fehler für time_lag
                "std_dev": sigma,
                "std_dev_error": sigma_err,
                "baseline": c,
                "baseline_error": c_err,
                "fit_window_start": x[left_min],  # Start des Fit-Fensters
                "fit_window_end": x[right_min],  # Ende des Fit-Fensters
                "x_values": x.tolist(),  # Originale x-Werte
                "y_values": y.tolist(),  # Originale y-Werte
                "fit_function": "a * exp(-((x - x0)^2) / (2 * sigma^2)) + c (c eingeschränkt)",
                "fit_success": True,
            }

        except RuntimeError:
            fit_result = {
                "line_name": line_name,
                "continuum": continuum,
                "x_values": x.tolist(),
                "y_values": y.tolist(),
                "error": "Fit konnte nicht durchgeführt werden.",
                "fit_success": False,
            }
        except ValueError as e:
            fit_result = {
                "line_name": line_name,
                "continuum": continuum,
                "x_values": x.tolist(),
                "y_values": y.tolist(),
                "error": f"Ungültige Eingabewerte: {e}",
                "fit_success": False,
            }

        # Ergebnis zur Liste hinzufügen
        results.append(fit_result)

    return results


def fit_window_with_minima(max_index, min_indices, x, y):
    # Linkes und rechtes Minimum finden
    left_min = min_indices[min_indices < max_index][-1] if any(min_indices < max_index) else 0
    right_min = min_indices[min_indices > max_index][0] if any(min_indices > max_index) else len(x) - 1
    # Fenster auf Bereich zwischen linkem und rechtem Minimum setzen
    fit_mask = (x >= x[left_min]) & (x <= x[right_min])
    x_fit_data = x[fit_mask]
    y_fit_data = y[fit_mask]
    return left_min, right_min, x_fit_data, y_fit_data


def fit_window_with_turningpoints(max_index, x, y):
    # Zweite Ableitung berechnen, um Wendepunkte zu finden
    dy = np.gradient(y, x)  # Erste Ableitung
    d2y = np.gradient(dy, x)  # Zweite Ableitung
    # Nullstellen der zweiten Ableitung finden (Wendepunkte)
    zero_crossings = np.where(np.diff(np.sign(d2y)))[0]
    left_wendepunkt = zero_crossings[zero_crossings < max_index][-1] if any(zero_crossings < max_index) else 0
    right_wendepunkt = zero_crossings[zero_crossings > max_index][0] if any(zero_crossings > max_index) else len(x) - 1
    # Fenster auf Bereich zwischen linkem und rechtem Wendepunkt setzen
    fit_mask = (x >= x[left_wendepunkt]) & (x <= x[right_wendepunkt])
    x_fit_data = x[fit_mask]
    y_fit_data = y[fit_mask]
    return left_wendepunkt, right_wendepunkt, x_fit_data, y_fit_data


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
