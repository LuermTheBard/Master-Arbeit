import sys
import subprocess
from pathlib import Path

import toml

from handle_data.handle_data import sort_1d_corr_data_for_lines
from handle_data.get_time_lag_from_1D_correlation import calc_time_lag_of_line, calculate_time_lags_for_continuum, \
    save_lag_results_to_toml
from handle_data.dopplershift import calc_broad_lines_doppler_shift_with_error
from import_data.import_data import import_1d_correlation_data, load_dopplershift_data_from_toml, \
    import_1d_lightcurve_data, import_fits_data
from plot_data.plot_1D_correlation_data import process_1d_correlations, compare_plots_across_continua, plot_fit_results
from plot_data.plot_1d_lightcurves_data import plot_1d_lightcurves, plot_1d_lightcurves_with_offset, \
    plot_all_1d_lightcurves_in_groups
from plot_data.plot_fits_data import plot_avg_rms
from settings import DEFAULT_OUTPUT_DIR

# Dictionary to store registered tasks
registered_tasks = {}


# Decorator to register functions as tasks
def task(func):
    registered_tasks[func.__name__] = func
    return func


@task
def dummy_task():
    subprocess.run(["python", "-m", "scrap.scripts.dummy_script", "calc_2_plus_2"], check=True)


@task
def plot_all_1d_corr():
    """
    Plots all 1D correlations.
    """
    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    process_1d_correlations(one_dim_correlation_plot_data)


@task
def save_all_1d_corr(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    process_1d_correlations(one_dim_correlation_plot_data, output_dir=output_dir, save_only=True)


@task
def compare_and_save_all_1d_corr(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    compare_plots_across_continua(one_dim_correlation_plot_data, output_dir=output_dir, save_only=True)


@task
def plot_and_save_1d_lightcurves(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()
    plot_1d_lightcurves(one_dim_lightcurve_data, output_dir)


@task
def plot_and_save_1d_8plots_lightcurves(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()
    plot_all_1d_lightcurves_in_groups(one_dim_lightcurve_data, output_dir)



@task
def save_1d_lightcurves(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()
    plot_1d_lightcurves(one_dim_lightcurve_data, output_dir, save_only=True)


@task
def plot_and_save_1d_lightcurves_with_offset(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()
    plot_1d_lightcurves_with_offset(one_dim_lightcurve_data, output_dir)


@task
def save_1d_lightcurves_with_offset(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()
    plot_1d_lightcurves_with_offset(one_dim_lightcurve_data, output_dir, save_only=True, y_offset=0.15)


@task
def plot_line_1d_corr(line_name=None):
    """
    Plots 1D correlations for a specific line.

    Args:
        line_name (str): The name of the line to plot.

    Raises:
        ValueError: If line_name is not provided.
    """
    # Prüfen, ob line_name definiert ist
    if not line_name:
        raise ValueError("Please specify a line name in the following form: plot_line_1d_corr::line_name")

    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    process_1d_correlations(one_dim_correlation_plot_data, line_name=line_name)


@task
def plot_avg_rms_spec():
    fits_data = import_fits_data()
    plot_avg_rms(fits_data)


@task
def calc_and_plot_time_lag_gauss(output_dir=DEFAULT_OUTPUT_DIR):

    lag_gaus_dir = output_dir / "time_lag_gauss"

    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    for campaign, line_data in sorted_one_dim_correlation_plot_data.items():
        for line, data in line_data.items():
            time_lag_data = calc_time_lag_of_line(line, data, window_methode="percentage", baseline_tolerance=1)
            plot_fit_results(campaign, time_lag_data, output_dir=lag_gaus_dir)


@task
def calc_and_plot_time_lag_centroid(output_dir=DEFAULT_OUTPUT_DIR):

    lag_centroid_dir = output_dir / "time_lag_centroid"

    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    for campaign, line_data in sorted_one_dim_correlation_plot_data.items():
        for line, data in line_data.items():
            time_lag_data = calc_time_lag_of_line(line, data, window_methode="percentage", lag_method="centroid")
            plot_fit_results(campaign, time_lag_data, centroid=True, output_dir=lag_centroid_dir)


@task
def calc_and_save_plot_time_lag_gauss(output_dir=DEFAULT_OUTPUT_DIR):

    lag_gaus_dir = output_dir / "time_lag_gauss"

    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    for campaign, line_data in sorted_one_dim_correlation_plot_data.items():
        for line, data in line_data.items():
            time_lag_data = calc_time_lag_of_line(line, data, window_methode="percentage", baseline_tolerance=1)
            plot_fit_results(campaign, time_lag_data, output_dir=lag_gaus_dir, save_only=True)


@task
def calc_and_save_plot_time_lag_centroid(output_dir=DEFAULT_OUTPUT_DIR):

    lag_centroid_dir = output_dir / "time_lag_centroid"

    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    for campaign, line_data in sorted_one_dim_correlation_plot_data.items():
        for line, data in line_data.items():
            time_lag_data = calc_time_lag_of_line(line, data, window_methode="percentage", lag_method="centroid")
            plot_fit_results(campaign, time_lag_data, centroid=True, output_dir=lag_centroid_dir, save_only=True)


@task
def calc_and_save_time_lag_gauss(output_dir=DEFAULT_OUTPUT_DIR, continuum_name="Cont5100"):

    lag_gaus_dir = output_dir / "time_lag_gauss"

    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)
    for campaign, line_data in sorted_one_dim_correlation_plot_data.items():
        campaign_dir = lag_gaus_dir / campaign
        campaign_dir.mkdir(parents=True, exist_ok=True)
        time_lag_file_path = campaign_dir / f"{campaign}_time_lag.toml"
        line_lag_dict = dict()
        for line, data in line_data.items():
            time_lag_data = calc_time_lag_of_line(line, data, window_methode="percentage", baseline_tolerance=1)
            file_name = campaign_dir / line / f"{line}.toml"
            save_lag_results_to_toml(time_lag_data, file_name)
            overall_results, skipped_result = calculate_time_lags_for_continuum(time_lag_data, continuum_name=continuum_name)
            line_lag_dict[line] = overall_results
            line_lag_dict["skipped_results"] = skipped_result

        sorted_line_lag_dict = dict(sorted(line_lag_dict.items()))

        with open(time_lag_file_path, "w") as file:
            toml.dump(sorted_line_lag_dict, file)


@task
def calc_and_save_time_lag_centroid(output_dir=DEFAULT_OUTPUT_DIR, continuum_name="Cont5100"):

    lag_centroid_dir = output_dir / "time_lag_centroid"

    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)

    for campaign, line_data in sorted_one_dim_correlation_plot_data.items():
        campaign_dir = lag_centroid_dir / campaign
        campaign_dir.mkdir(parents=True, exist_ok=True)
        time_lag_file_path = campaign_dir / f"{campaign}_{continuum_name}_time_lag.toml"
        line_lag_dict = dict()
        for line, data in line_data.items():
            time_lag_data = calc_time_lag_of_line(line, data, window_methode="percentage", lag_method="centroid")
            file_name = campaign_dir / line / f"{line}.toml"
            save_lag_results_to_toml(time_lag_data, file_name)
            overall_results, skipped_result = calculate_time_lags_for_continuum(time_lag_data, continuum_name=continuum_name)
            line_lag_dict[line] = overall_results
            line_lag_dict["skipped_results"] = skipped_result

        sorted_line_lag_dict = dict(sorted(line_lag_dict.items()))

        with open(time_lag_file_path, "w") as file:
            toml.dump(sorted_line_lag_dict, file)



@task
def calc_dopplershift_correction(file_name=None):
    if not file_name:
        raise ValueError("Please specify a file name in the following form: calc_dopplershift_correction::file_name")

    toml_data = load_dopplershift_data_from_toml(file_name)

    ref_name = toml_data["reference_shift"]["name"]

    lambda_rest_ref = toml_data["central_wavelengths"][ref_name]
    lambda_rest_ref_err = toml_data["central_wavelengths_err"][f"{ref_name}_err"]

    delta_lambda_ref = toml_data["reference_shift"]["delta_lambda"]
    delta_lambda_ref_err = toml_data["reference_shift"]["delta_lambda_err"]

    line_rest_lambda_dict = toml_data["central_wavelengths"]
    line_rest_lambda_dict.pop(ref_name)

    line_rest_lambda_err_dict = toml_data["central_wavelengths_err"]
    line_rest_lambda_err_dict.pop(f"{ref_name}_err")

    calc_broad_lines_doppler_shift_with_error(delta_lambda_ref, delta_lambda_ref_err, lambda_rest_ref,
                                              lambda_rest_ref_err, ref_name, line_rest_lambda_dict,
                                              line_rest_lambda_err_dict)


def run_task(commands):
    for command in commands:
        #try:
            # Split command name and additional arguments
            parts = command.split("::")
            name_of_task = parts[0]
            task_args = parts[1:] if len(parts) > 1 else []

            print(f"Running {name_of_task} with arguments {task_args}...")

            # Call the command with unpacked arguments
            registered_tasks[name_of_task](*task_args)
        #except KeyError as k:
            #print(f"Task '{command}' is not available.")
        #except Exception as e:
            #print(f"An error occurred while running '{command}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in registered_tasks:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])
