from pathlib import Path
from typing import Dict, Tuple

from settings import DEFAULT_OUTPUT_DIR

DEFAULT_DOPPLERSHIFT_SAVE_FILENAME = DEFAULT_OUTPUT_DIR / "handle_data" / "doppler_shift_results.txt"


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
    # Calculate the Doppler shift and error for the reference line
    delta_v_c = delta_lambda_ref / lambda_rest_ref
    delta_v_c_err = calc_error(delta_lambda_ref, delta_lambda_ref_err, delta_v_c, lambda_rest_ref, lambda_rest_ref_err)

    print(f"Doppler shift (delta_v / c) for the reference line {ref_name}: {delta_v_c:.6f} ± {delta_v_c_err:.6f}")

    # Calculate Doppler shifts and errors for each line
    line_doppler_shifts = {
        line_name: (
            delta_lambda_line := delta_v_c * lambda_rest,
            calc_error(lambda_rest, line_rest_lambda_err_dict.get(line_name, 0.0), delta_lambda_line, delta_v_c,
                       delta_v_c_err)
        )
        for line_name, lambda_rest in line_rest_lambda_dict.items()
    }

    # Print results
    for line_name, (delta_lambda_line, delta_lambda_line_err) in line_doppler_shifts.items():
        lambda_rest_err = line_rest_lambda_err_dict.get(line_name, 0.0)
        print(f"Line: {line_name}, Rest wavelength: {line_rest_lambda_dict[line_name]} ± {lambda_rest_err}, "
              f"Doppler shift (delta_lambda): {delta_lambda_line:.6f} ± {delta_lambda_line_err:.6f}")

    # Bundle all data into a dictionary
    result_data = {
        "reference": {
            "name": ref_name,
            "delta_lambda_ref": delta_lambda_ref,
            "delta_lambda_ref_err": delta_lambda_ref_err,
            "lambda_rest_ref": lambda_rest_ref,
            "lambda_rest_ref_err": lambda_rest_ref_err,
            "delta_v_c": delta_v_c,
            "delta_v_c_err": delta_v_c_err
        },
        "lines": {
            line_name: {
                "rest_wavelength": line_rest_lambda_dict[line_name],
                "rest_wavelength_err": line_rest_lambda_err_dict.get(line_name, 0.0),
                "doppler_shift": delta_lambda_line,
                "doppler_shift_err": delta_lambda_line_err
            }
            for line_name, (delta_lambda_line, delta_lambda_line_err) in line_doppler_shifts.items()
        }
    }

    # Save results to a file
    _calc_broad_lines_doppler_shift_save_to_txt(result_data)

    return line_doppler_shifts, delta_v_c, delta_v_c_err


def _calc_broad_lines_doppler_shift_save_to_txt(result_data: Dict,
                                                output_file_path: Path = DEFAULT_DOPPLERSHIFT_SAVE_FILENAME):
    """
    Saves the input values and calculated results to a text file.

    Args:
        result_data (Dict): A dictionary containing all the relevant input parameters and results.
        output_file_path (Path): Name of the file to save the results to.
    """

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True)

    file_name = str(output_file_path)

    with open(file_name, "w", encoding="UTF-8") as file:
        file.write("Input Parameters:\n")
        ref_data = result_data["reference"]
        file.write(f"Reference Line: {ref_data['name']}\n")
        file.write(f"Delta Lambda Ref: {ref_data['delta_lambda_ref']:.6f} ± {ref_data['delta_lambda_ref_err']:.6f}\n")
        file.write(f"Lambda Rest Ref: {ref_data['lambda_rest_ref']:.6f} ± {ref_data['lambda_rest_ref_err']:.6f}\n")
        file.write(f"Doppler Shift (delta_v / c): {ref_data['delta_v_c']:.6f} ± {ref_data['delta_v_c_err']:.6f}\n")

        file.write("\nLine Rest Wavelengths and Errors:\n")
        for line_name, line_data in result_data["lines"].items():
            file.write(f"{line_name}: Rest Wavelength = {line_data['rest_wavelength']:.6f} ± {line_data['rest_wavelength_err']:.6f}, "
                       f"Doppler Shift = {line_data['doppler_shift']:.6f} ± {line_data['doppler_shift_err']:.6f}\n")

    print(f"Results saved to {file_name}")


def calc_error(value: float, value_err: float, result: float, ref_value: float, ref_value_err: float) -> float:
    """
    Calculates the propagated error for a given result based on its inputs and their errors.

    Args:
        value (float): First value.
        value_err (float): Error in the first value.
        result (float): Calculated result.
        ref_value (float): Reference value.
        ref_value_err (float): Error in the reference value.

    Returns:
        float: Propagated error in the result.
    """
    return result * (
        ((value_err / value) if value != 0 else 0) ** 2 +
        ((ref_value_err / ref_value) if ref_value != 0 else 0) ** 2
    ) ** 0.5
