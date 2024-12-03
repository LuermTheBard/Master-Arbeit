import numpy as np

from handle_data.handle_data import calc_error_of_function
from handle_data.dopplershift import calc_error


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


def test_simple_multiplication():
    # Testfunktion
    def func(x, y):
        return x * y

    # Parameterwerte und Fehler
    params = [3.0, 2.0]  # x=3.0, y=2.0
    errors = [0.1, 0.2]  # Fehler: x_err=0.1, y_err=0.2

    # Erwarteter Fehler:
    # ∂f/∂x = y, ∂f/∂y = x
    # σ_f = sqrt((∂f/∂x * σ_x)^2 + (∂f/∂y * σ_y)^2)
    expected_error = ((2.0 * 0.1)**2 + (3.0 * 0.2)**2)**0.5

    # Berechnung mit calc_propagated_error
    calculated_error = calc_error_of_function(func, params, errors)

    # Überprüfung
    assert abs(calculated_error - expected_error) < 1e-6, \
        f"Expected {expected_error}, but got {calculated_error}"


def test_quadratic_function():
    # Testfunktion
    def func(x, y):
        return x**2 + y**2

    # Parameterwerte und Fehler
    params = [3.0, 4.0]  # x=3.0, y=4.0
    errors = [0.1, 0.2]  # Fehler: x_err=0.1, y_err=0.2

    # Erwarteter Fehler:
    # ∂f/∂x = 2x, ∂f/∂y = 2y
    # σ_f = sqrt((2x * σ_x)^2 + (2y * σ_y)^2)
    expected_error = ((2 * 3.0 * 0.1)**2 + (2 * 4.0 * 0.2)**2)**0.5

    # Berechnung mit calc_propagated_error
    calculated_error = calc_error_of_function(func, params, errors)

    # Überprüfung
    assert abs(calculated_error - expected_error) < 1e-6, \
        f"Expected {expected_error}, but got {calculated_error}"


def test_complex_function_with_five_parameters():
    # Testfunktion
    # f(a, b, c, d, e) = (a * b + c) / (d - e)
    def func(a, b, c, d, e):
        return (a * b + c) / (d - e)

    # Parameterwerte und Fehler
    params = [2.0, 3.0, 1.0, 5.0, 1.0]  # a=2.0, b=3.0, c=1.0, d=5.0, e=1.0
    errors = [0.1, 0.2, 0.05, 0.1, 0.05]  # Fehler: a_err=0.1, b_err=0.2, etc.

    # Erwartete Ableitungen:
    # ∂f/∂a = b / (d - e)
    # ∂f/∂b = a / (d - e)
    # ∂f/∂c = 1 / (d - e)
    # ∂f/∂d = -(a * b + c) / (d - e)^2
    # ∂f/∂e = (a * b + c) / (d - e)^2

    a, b, c, d, e = params
    ab_plus_c = a * b + c
    denominator = d - e

    partials = [
        b / denominator,                       # ∂f/∂a
        a / denominator,                       # ∂f/∂b
        1 / denominator,                       # ∂f/∂c
        -ab_plus_c / denominator**2,          # ∂f/∂d
        ab_plus_c / denominator**2            # ∂f/∂e
    ]

    # Erwarteter Fehler: σ_f = sqrt((∂f/∂a * σ_a)^2 + (∂f/∂b * σ_b)^2 + ...)
    expected_error = np.sqrt(sum((partials[i] * errors[i])**2 for i in range(5)))

    # Berechnung mit calc_propagated_error
    calculated_error = calc_error_of_function(func, params, errors)

    # Überprüfung
    assert abs(calculated_error - expected_error) < 1e-6, \
        f"Expected {expected_error}, but got {calculated_error}"
