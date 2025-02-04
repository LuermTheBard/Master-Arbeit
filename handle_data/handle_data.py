import numpy as np
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
