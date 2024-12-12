import numpy as np


def get_continua_with_highest_corr_coef(line_sorted_data):
    """
    Bestimmt die Continuen mit den höchsten, zweit- und dritthöchsten Maximalwerten über alle Linien hinweg.

    Args:
        line_sorted_data (dict): Ein Dictionary, bei dem der Schlüssel die Linie und der Wert
                                 eine Liste von Tupeln ist (continua_name, x_values, y_values).

    Returns:
        dict: Dictionary mit den besten Continuen und ihren Maximalwerten:
              {
                  "best_continua": (continua_name, max_value),
                  "second_best_continua": (continua_name, max_value),
                  "third_best_continua": (continua_name, max_value)
              }
    """
    # Dictionary für die besten Continuen pro Linie
    line_best_continua_dict = dict()

    # Schritt 1: Berechnung der maximalen, zweitgrößten und drittgrößten Werte für jede Linie
    for line, cont_data_tuple_list in line_sorted_data.items():
        cont_max_dict = dict()
        cont_second_max_dict = dict()
        cont_third_max_dict = dict()

        for cont_tuple in cont_data_tuple_list:
            # cont_tuple: (continua_name, x_values, y_values)
            sorted_values = sorted(cont_tuple[2], reverse=True)

            cont_max_dict[cont_tuple[0]] = sorted_values[0]

            if len(sorted_values) > 1:
                cont_second_max_dict[cont_tuple[0]] = sorted_values[1]
            else:
                cont_second_max_dict[cont_tuple[0]] = None

            if len(sorted_values) > 2:
                cont_third_max_dict[cont_tuple[0]] = sorted_values[2]
            else:
                cont_third_max_dict[cont_tuple[0]] = None

        line_best_continua_dict[line] = (cont_max_dict, cont_second_max_dict, cont_third_max_dict)

    print()


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
