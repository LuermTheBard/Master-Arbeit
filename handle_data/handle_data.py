from typing import Dict, Tuple


def create_1d_correlation_plot_data(galaxy_campaigns_dict):

    plot_data_dict = dict()

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

        plot_data_dict[campaign] = line_data_dict

    return plot_data_dict


def calc_broad_lines_doppler_shift_with_error(
    delta_lambda_ref: float,
    delta_lambda_ref_err: float,
    lambda_rest_ref: float,
    lambda_rest_ref_err: float,
    ref_name: str,
    line_rest_lambda_dict: Dict[str, float],
    line_rest_lambda_err_dict: Dict[str, float]
) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """
    First calculates the Doppler shift of the reference line in velocity space (delta_v / c)
    and then derives the corresponding Doppler shifts and their errors for the provided lines in a dictionary.

    Args:
        delta_lambda_ref (float): Doppler shift of the reference line (in wavelength).
        delta_lambda_ref_err (float): Error in the Doppler shift of the reference line.
        lambda_rest_ref (float): Rest wavelength of the reference line.
        lambda_rest_ref_err (float): Error in the rest wavelength of the reference line.
        ref_name (str): Name of the reference line.
        line_rest_lambda_dict (Dict[str, float]): Dictionary containing line names and their rest wavelengths.
        line_rest_lambda_err_dict (Dict[str, float]): Dictionary containing line names and their wavelength errors.

    Returns:
        Tuple[Dict[str, Tuple[float, float]], float, float]:
            - A dictionary with the line names as keys and their calculated Doppler shifts (delta_lambda) and errors as values.
            - The Doppler shift of the reference line in velocity space (delta_v / c).
            - The error in the Doppler shift of the reference line in velocity space (delta_v / c).
    """
    # Calculate the Doppler shift in velocity space for the reference line
    delta_v_c = delta_lambda_ref / lambda_rest_ref
    delta_v_c_err = 0.0

    if lambda_rest_ref_err != 0 or delta_lambda_ref_err != 0:
        delta_v_c_err = delta_v_c * (
            ((delta_lambda_ref_err / delta_lambda_ref) if delta_lambda_ref != 0 else 0) ** 2 +
            ((lambda_rest_ref_err / lambda_rest_ref) if lambda_rest_ref != 0 else 0) ** 2
        ) ** 0.5

    print(f"Doppler shift (delta_v / c) for the reference line {ref_name}: {delta_v_c:.6f} ± {delta_v_c_err:.6f}")

    # Initialize the result dictionary
    line_doppler_shifts = {}

    # Calculate Doppler shifts and errors for each line
    for line_name, lambda_rest in line_rest_lambda_dict.items():
        lambda_rest_err = line_rest_lambda_err_dict.get(line_name, 0.0)

        # Calculate the Doppler shift
        delta_lambda_line = delta_v_c * lambda_rest

        # Calculate the error
        delta_lambda_line_err = 0.0
        if lambda_rest_err != 0 or delta_v_c_err != 0:
            delta_lambda_line_err = delta_lambda_line * (
                ((lambda_rest_err / lambda_rest) if lambda_rest != 0 else 0) ** 2 +
                ((delta_v_c_err / delta_v_c) if delta_v_c != 0 else 0) ** 2
            ) ** 0.5

        line_doppler_shifts[line_name] = (delta_lambda_line, delta_lambda_line_err)

        print(f"Line: {line_name}, Rest wavelength: {lambda_rest} ± {lambda_rest_err}, "
              f"Doppler shift (delta_lambda): {delta_lambda_line:.6f} ± {delta_lambda_line_err:.6f}")

    return line_doppler_shifts, delta_v_c, delta_v_c_err

