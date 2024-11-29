import sys
import subprocess
from pathlib import Path

from handle_data.handle_data import create_1d_correlation_plot_data, calc_broad_lines_doppler_shift_with_error
from import_data.import_data import import_1d_correlation_data, load_dopplershift_data_from_toml, \
    import_1d_lightcurve_data, import_fits_data
from plot_data.plot_1D_correlation_data import process_1d_correlations, compare_plots_across_continua
from plot_data.plot_1d_lightcurves_data import plot_1d_lightcurves, plot_1d_lightcurves_with_offset
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
    one_dim_correlation_plot_data = create_1d_correlation_plot_data(one_dim_correlation_data)
    process_1d_correlations(one_dim_correlation_plot_data)


@task
def save_all_1d_corr(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    one_dim_correlation_data = import_1d_correlation_data()
    one_dim_correlation_plot_data = create_1d_correlation_plot_data(one_dim_correlation_data)
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
    one_dim_correlation_plot_data = create_1d_correlation_plot_data(one_dim_correlation_data)
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
    one_dim_correlation_plot_data = create_1d_correlation_plot_data(one_dim_correlation_data)
    process_1d_correlations(one_dim_correlation_plot_data, line_name=line_name)


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
        try:
            # Split command name and additional arguments
            parts = command.split("::")
            name_of_task = parts[0]
            task_args = parts[1:] if len(parts) > 1 else []

            print(f"Running {name_of_task} with arguments {task_args}...")

            # Call the command with unpacked arguments
            registered_tasks[name_of_task](*task_args)
        except KeyError as k:
            print(f"Task '{command}' is not available.")
        except Exception as e:
            print(f"An error occurred while running '{command}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in registered_tasks:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])
