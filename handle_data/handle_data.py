import numpy as np
from scipy.optimize import curve_fit


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


def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def calc_time_lag_of_line(line_name, continuum_x_y_tuple_list):
    """
    Passt eine Gauß-Funktion an das größte Maximum der gegebenen Daten für jede Datenreihe an
    und berechnet Fit-Parameter, Time Lag und Fehler.

    Parameters:
        line_name (str): Der Name der Linie.
        continuum_x_y_tuple_list (list): Eine Liste von Tupeln, wobei jedes Tupel aus
                                         (continuum, x, y) besteht:
                                         - continuum: Name des Kontinuums.
                                         - x: Array der x-Werte.
                                         - y: Array der y-Werte.

    Returns:
        list: Eine Liste von Dictionaries mit Fit-Parametern, Time Lag, Fehlern und weiteren Informationen.
    """
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


    results = []

    for continuum_x_y_tuple in continuum_x_y_tuple_list:
        continuum, x, y = continuum_x_y_tuple

        # Sicherstellen, dass x und y als 1D-Arrays vorliegen
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Größtes Maximum finden
        max_index = np.argmax(y)
        max_x = x[max_index]
        max_y = y[max_index]

        # Initiale Schätzwerte für den Fit
        initial_guess = [max_y, max_x, 1.0]  # [Amplitude, Mittelwert, Standardabweichung]

        # Curve-Fitting durchführen
        try:
            popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess)
            a, x0, sigma = popt  # Fit-Parameter
            errors = np.sqrt(np.diag(pcov))  # Fehler der Fit-Parameter aus der Kovarianzmatrix
            a_err, x0_err, sigma_err = errors

            fit_result = {
                "line_name": line_name,
                "continuum": continuum,
                "amplitude": a,
                "amplitude_error": a_err,
                "time_lag": x0,  # mean als time_lag
                "time_lag_error": x0_err,  # Fehler für time_lag
                "std_dev": sigma,
                "std_dev_error": sigma_err,
                "x_values": x.tolist(),  # x-Werte hinzufügen
                "y_values": y.tolist(),  # y-Werte hinzufügen
                "fit_function": "a * exp(-((x - x0)^2) / (2 * sigma^2))",
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



