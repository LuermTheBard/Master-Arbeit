from handle_data.handle_data import calc_broad_lines_doppler_shift_with_error, calc_error


def test_calc_broad_lines_doppler_shift_with_error():
    # Input parameters
    delta_lambda_ref = 10.0  # Doppler shift of the reference line (H-alpha) in Ångström
    delta_lambda_ref_err = 0.5  # Error in Doppler shift of the reference line
    lambda_rest_ref = 6563.0  # Correct rest wavelength of H-alpha in Ångström
    lambda_rest_ref_err = 1.0  # Error in rest wavelength of H-alpha
    ref_name = "H-alpha"  # Reference line name

    # Dictionary of lines to calculate Doppler shifts for
    line_rest_lambda_dict = {
        "H-beta": 4861.0,  # Rest wavelength of H-beta in Ångström
        "H-gamma": 4340.0,  # Rest wavelength of H-gamma in Ångström
        "H-delta": 4102.0  # Rest wavelength of H-delta in Ångström
    }

    # Corresponding errors in the rest wavelengths
    line_rest_lambda_err_dict = {
        "H-beta": 1.0,  # Error in rest wavelength of H-beta
        "H-gamma": 0.5,  # Error in rest wavelength of H-gamma
        "H-delta": 1.5  # Error in rest wavelength of H-delta
    }

    # Call the function
    doppler_shifts, delta_v_c, delta_v_c_err = calc_broad_lines_doppler_shift_with_error(
        delta_lambda_ref, delta_lambda_ref_err, lambda_rest_ref, lambda_rest_ref_err, ref_name,
        line_rest_lambda_dict, line_rest_lambda_err_dict
    )

    # Expected results
    expected_delta_v_c = delta_lambda_ref / lambda_rest_ref
    expected_delta_v_c_err = expected_delta_v_c * (
        ((delta_lambda_ref_err / delta_lambda_ref) if delta_lambda_ref != 0 else 0) ** 2 +
        ((lambda_rest_ref_err / lambda_rest_ref) if lambda_rest_ref != 0 else 0) ** 2
    ) ** 0.5

    expected_doppler_shifts = {
        "H-beta": (expected_delta_v_c * 4861.0, expected_delta_v_c * 4861.0 * (
            ((1.0 / 4861.0) if 4861.0 != 0 else 0) ** 2 +
            ((expected_delta_v_c_err / expected_delta_v_c) if expected_delta_v_c != 0 else 0) ** 2
        ) ** 0.5),
        "H-gamma": (expected_delta_v_c * 4340.0, expected_delta_v_c * 4340.0 * (
            ((0.5 / 4340.0) if 4340.0 != 0 else 0) ** 2 +
            ((expected_delta_v_c_err / expected_delta_v_c) if expected_delta_v_c != 0 else 0) ** 2
        ) ** 0.5),
        "H-delta": (expected_delta_v_c * 4102.0, expected_delta_v_c * 4102.0 * (
            ((1.5 / 4102.0) if 4102.0 != 0 else 0) ** 2 +
            ((expected_delta_v_c_err / expected_delta_v_c) if expected_delta_v_c != 0 else 0) ** 2
        ) ** 0.5)
    }

    # Assertions
    assert abs(delta_v_c - expected_delta_v_c) < 1e-6, f"Expected delta_v_c: {expected_delta_v_c}, got: {delta_v_c}"
    assert abs(delta_v_c_err - expected_delta_v_c_err) < 1e-6, f"Expected delta_v_c_err: {expected_delta_v_c_err}, got: {delta_v_c_err}"

    for line, (expected_shift, expected_shift_err) in expected_doppler_shifts.items():
        calculated_shift, calculated_shift_err = doppler_shifts[line]
        assert abs(calculated_shift - expected_shift) < 1e-6, (
            f"For line {line}, expected Doppler shift: {expected_shift}, got: {calculated_shift}"
        )
        assert abs(calculated_shift_err - expected_shift_err) < 1e-6, (
            f"For line {line}, expected Doppler shift error: {expected_shift_err}, got: {calculated_shift_err}"
        )

    print("Test passed: test_calc_broad_lines_doppler_shift_with_error()")


def test_calc_error():
    # Input parameters
    delta_lambda_ref = 10.0  # Doppler shift of the reference line
    delta_lambda_ref_err = 0.5  # Error in the Doppler shift of the reference line
    delta_v_c = 0.001524  # Calculated Doppler shift in velocity space
    lambda_rest_ref = 6563.0  # Rest wavelength of the reference line
    lambda_rest_ref_err = 1.0  # Error in the rest wavelength of the reference line

    # Expected result for delta_v_c_err
    expected_delta_v_c_err = delta_v_c * (
        ((delta_lambda_ref_err / delta_lambda_ref) if delta_lambda_ref != 0 else 0) ** 2 +
        ((lambda_rest_ref_err / lambda_rest_ref) if lambda_rest_ref != 0 else 0) ** 2
    ) ** 0.5

    # Call the function
    calculated_delta_v_c_err = calc_error(delta_lambda_ref, delta_lambda_ref_err, delta_v_c, lambda_rest_ref,
                                          lambda_rest_ref_err)

    # Assertion
    assert abs(calculated_delta_v_c_err - expected_delta_v_c_err) < 1e-6, (
        f"Expected delta_v_c_err: {expected_delta_v_c_err}, got: {calculated_delta_v_c_err}"
    )

    print("Test passed: calc_error()")


