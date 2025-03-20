import subprocess
import sys
from pathlib import Path

import numpy as np
from fontTools.ttLib.tables.G_S_U_B_ import table_G_S_U_B_
from matplotlib import pyplot as plt

from handle_data.calc_time_lag import get_time_lags
from handle_data.handle_data import sort_1d_corr_data_for_lines, get_continua_with_highest_corr_coef, \
    get_weighted_best_continua
from import_data.import_data import import_1d_correlation_data, import_1d_lightcurve_data, import_fits_data, \
    import_line_profile_data
from plot_data.plot_1D_ccfs_in_groups_data import plot_all_1d_ccfs_in_groups_for_cont
from plot_data.plot_1D_correlation_data import process_1d_correlations, compare_plots_across_continua
from plot_data.plot_1D_lightcurves_in_groups_data import plot_all_1d_lightcurves_in_groups
from plot_data.plot_1d_lightcurves_data import plot_1d_lightcurves, plot_1d_lightcurves_with_offset
from plot_data.plot_fits_data import plot_avg_rms
from plot_data.plot_line_profiles import plot_normalized_line_profiles_in_pairs, \
    plot_normalized_line_profiles_together, process_spectrum, plot_normalized_line_profiles_type_together, \
    transform_wavelength_to_velocity_and_cut, save_velocity_data_to_txt, cut_normalized_line_out, cut_line_out
from settings import DEFAULT_OUTPUT_DIR, CENTRAL_WAVELENGTH

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
def calc_time_lag():
    one_dim_correlation_data = import_1d_correlation_data()
    result_dict = get_time_lags(one_dim_correlation_data, baseline=0, selected_continua=["Cont1150", "Cont5100"],
                                plot=True)

    print(result_dict)


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
def plot_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()

    key_order = ["Cont1150", 'HAlpha', 'HBeta', 'HGamma', 'HeI5875', 'HeI7065', 'HeII4685', 'OI8446']
    plot_all_1d_lightcurves_in_groups(one_dim_lightcurve_data, output_dir, compare_cont="Cont1150", key_order=key_order)

    key_order = ["Cont5100", 'HAlpha', 'HBeta', 'HGamma', 'HeI5875', 'HeI7065', 'HeII4685', 'OI8446']
    plot_all_1d_lightcurves_in_groups(one_dim_lightcurve_data, output_dir, compare_cont="Cont5100", key_order=key_order)


@task
def save_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    one_dim_lightcurve_data = import_1d_lightcurve_data()

    key_order = ["Cont1150", 'HAlpha', 'HBeta', 'HGamma', 'HeI5875', 'HeI7065', 'HeII4685', 'OI8446']
    plot_all_1d_lightcurves_in_groups(one_dim_lightcurve_data, output_dir, compare_cont="Cont1150", key_order=key_order,
                                      save_only=True)

    key_order = ["Cont5100", 'HAlpha', 'HBeta', 'HGamma', 'HeI5875', 'HeI7065', 'HeII4685', 'OI8446']
    plot_all_1d_lightcurves_in_groups(one_dim_lightcurve_data, output_dir, compare_cont="Cont5100", key_order=key_order,
                                      save_only=True)


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
def plot_1d_corr_in_groups_for_cont(cont_name=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """

    # Prüfen, ob line_name definiert ist
    if not cont_name:
        raise ValueError("Please specify a line name in the following form: plot_line_1d_corr::line_name")

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    key_order = ["time shift (tau)", 'HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI7065', 'HeI5875', 'HeII4685', 'OI8446']

    one_dim_correlation_data = import_1d_correlation_data()

    plot_all_1d_ccfs_in_groups_for_cont(one_dim_correlation_data, cont_name=cont_name, output_dir=output_dir,
                                        key_order=key_order)


@task
def save_1d_corr_in_groups_for_cont(cont_name=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    save all 1D correlations.
    """

    # Prüfen, ob line_name definiert ist
    if not cont_name:
        raise ValueError("Please specify a line name in the following form: plot_line_1d_corr::line_name")

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    key_order = ["time shift (tau)", 'HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI7065', 'HeI5875', 'HeII4685', 'OI8446']
    # key_order = ["time shift (tau)", 'HeI5015', 'HeI5875', 'HeI4471', 'HeI7065', 'HeII4685','HAlpha', 'HBeta', 'HGamma', 'HDelta',  'OI8446']

    one_dim_correlation_data = import_1d_correlation_data()

    plot_all_1d_ccfs_in_groups_for_cont(one_dim_correlation_data, cont_name=cont_name, output_dir=output_dir,
                                        key_order=key_order, save_only=True)


@task
def plot_avg_rms_spec(file_name="avg_rms_spec", output_dir=DEFAULT_OUTPUT_DIR):
    avg_rms_spec_dir = output_dir / "avg_rms_spec"

    avg_rms_spec_dir.mkdir(parents=True, exist_ok=True)

    avg_rms_spec_file = avg_rms_spec_dir / f"{file_name}.pdf"

    fits_data = import_fits_data()
    plot_avg_rms(fits_data, save_path=avg_rms_spec_file)


@task
def plot_and_save_normalized_line_profiles_in_pairs(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    line_profile_campaign_dict = import_line_profile_data(normalized=True)

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446']

    for campaign, data in line_profile_campaign_dict.items():
        plot_normalized_line_profiles_in_pairs(data, campaign, key_order)


@task
def save_line_normalized_line_profiles_in_pairs(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    line_profile_dict = import_line_profile_data(normalized=True)

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446']

    plot_normalized_line_profiles_in_pairs(line_profile_dict, key_order, save_only=True)


@task
def plot_and_save_normalized_line_profiles_together(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    line_profile_dict = import_line_profile_data(normalized=True)

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446']

    plot_normalized_line_profiles_together(line_profile_dict, key_order)


@task
def plot_and_save_normalized_line_profiles_types_together(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    line_profile_dict = import_line_profile_data(normalized=True)

    key_order = ['HeI5875', 'HeI7065', 'HeII4685', ]

    # key_order = ['HAlpha', 'HBeta', 'HGamma']

    # key_order = ['HAlpha', 'HBeta']

    plot_normalized_line_profiles_type_together(line_profile_dict, key_order)


@task
def save_line_normalized_line_profiles_together(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    line_profile_dict = import_line_profile_data(normalized=True)

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446']

    plot_normalized_line_profiles_together(line_profile_dict, key_order, save_only=True)


@task
def substract_pseudo_continua_from_spectra(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir.mkdir(parents=True)

    fits_data = import_fits_data()

    wavelenghts = np.array(fits_data['NGC4593_avg.fits']['x_axis'][0])
    avg_data = np.array(fits_data['NGC4593_avg.fits']['data'][0])
    rms_data = np.array(fits_data['NGC4593_rms.fits']['data'][0])

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446']

    for line in key_order:
        process_spectrum(wavelenghts, avg_data, line, spec_type="avg", output_dir=output_dir, plot=True)
        process_spectrum(wavelenghts, rms_data, line, spec_type="rms", output_dir=output_dir, plot=True)


@task
def cut_out_and_plot_line_profiles_from_fits(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / "Line_Profiles_intercaly_pseudo_substracted"
    output_path.mkdir(parents=True, exist_ok=True)

    fits_data_H_Alpha = import_fits_data("Halpha_pseudo_cont_substracted")
    fits_data_H_Beta = import_fits_data("Hbeta_pseudo_cont_substracted")

    line_profile_dict = import_line_profile_data(normalized=True)


    H_Alpha_wavelenghts = np.array(fits_data_H_Alpha['avg_HAlpha_Line_Profile.fits']['x_axis'][0])
    H_Alpha_avg_data = np.array(fits_data_H_Alpha['avg_HAlpha_Line_Profile.fits']['data'][0])
    H_Alpha_rms_data = np.array(fits_data_H_Alpha['rms_HAlpha_Line_Profile.fits']['data'][0])

    H_Beta_wavelenghts = np.array(fits_data_H_Beta['avg_HBeta_Line_Profile.fits']['x_axis'][0])
    H_Beta_avg_data = np.array(fits_data_H_Beta['avg_HBeta_Line_Profile.fits']['data'][0])
    H_Beta_rms_data = np.array(fits_data_H_Beta['rms_HBeta_Line_Profile.fits']['data'][0])




    H_Alpha_avg_velocity, H_Alpha_avg_intensity = transform_wavelength_to_velocity_and_cut(H_Alpha_wavelenghts,
                                                                                           H_Alpha_avg_data,
                                                                                           "HAlpha",
                                                                                           (-20000, 20000),
                                                                                           output_path / "H_Alpha_AVG_Line_Profile.txt")

    H_Alpha_rms_velocity, H_Alpha_rms_intensity = transform_wavelength_to_velocity_and_cut(H_Alpha_wavelenghts,
                                                                                           H_Alpha_rms_data, "HAlpha",
                                                                                           (-20000, 20000),
                                                                                           output_path / "H_Alpha_RMS_Line_Profile.txt")

    H_Beta_avg_velocity, H_Beta_avg_intensity = transform_wavelength_to_velocity_and_cut(H_Beta_wavelenghts,
                                                                                         H_Beta_avg_data, "HBeta",
                                                                                         (-20000, 20000),
                                                                                         output_path / "H_Beta_AVG_Line_Profile.txt")

    H_Beta_rms_velocity, H_Beta_rms_intensity = transform_wavelength_to_velocity_and_cut(H_Beta_wavelenghts,
                                                                                         H_Beta_rms_data, "HBeta",
                                                                                         (-20000, 20000),
                                                                                         output_path / "H_Beta_RMS_Line_Profile.txt")

    line_profile_dict = import_line_profile_data(normalized=True)

    line_profile_dict_add = {"avg":
                             {"HAlpha_substracted_first":
                                  {"data_dict":
                                       {'velocity space (km/s)':
                                            H_Alpha_avg_velocity,
                                        'normalized flux': H_Alpha_avg_intensity}
                                   },
                              "HBeta_substracted_first":
                                  {"data_dict":
                                       {'velocity space (km/s)':
                                            H_Beta_avg_velocity,
                                        'normalized flux':
                                            H_Beta_avg_intensity}}},
                         "rms":
                             {"HAlpha_substracted_first":
                                  {"data_dict":
                                       {'velocity space (km/s)':
                                            H_Alpha_rms_velocity,
                                        'normalized flux': H_Alpha_rms_intensity}},
                              "HBeta_substracted_first":
                                  {"data_dict":
                                       {'velocity space (km/s)':
                                            H_Beta_rms_velocity,
                                        'normalized flux':
                                            H_Beta_rms_intensity}}},
                         }

    merged_dict = merge_dicts(line_profile_dict, line_profile_dict_add)

    key_order = ['HAlpha', 'HAlpha_substracted_first']

    plot_normalized_line_profiles_type_together(merged_dict, key_order)

    key_order = ['HBeta', 'HBeta_substracted_first']

    plot_normalized_line_profiles_type_together(merged_dict, key_order)

    key_order = ['HAlpha_substracted_first', 'HBeta_substracted_first']

    plot_normalized_line_profiles_type_together(merged_dict, key_order)


def merge_dicts(d1, d2):
    """
    Rekursive Funktion zum Zusammenführen zweier geschachtelter Dictionaries.
    Falls Werte existieren, werden sie beibehalten oder überschrieben, falls notwendig.
    """
    for key, value in d2.items():
        if key in d1:
            if isinstance(d1[key], dict) and isinstance(value, dict):
                merge_dicts(d1[key], value)  # Rekursiver Aufruf für geschachtelte Dicts
            else:
                d1[key] = value  # Überschreiben, falls kein Dict
        else:
            d1[key] = value  # Falls Key nicht existiert, direkt übernehmen
    return d1


@task
def cut_normalized_line_profile(line_name, range=(-10000, 10000),  plot=False, output_dir=DEFAULT_OUTPUT_DIR):
    if isinstance(range, str):
        range = tuple(map(int, range.strip("() ").split(',')))

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / "Line_Profiles_cut_out"
    output_path.mkdir(parents=True, exist_ok=True)

    line_profile_dict = import_line_profile_data(normalized=True)

    line_data_avg = line_profile_dict["avg"][line_name]["data_dict"]
    line_data_rms = line_profile_dict["rms"][line_name]["data_dict"]

    velocitys = line_data_avg["velocity space (km/s)"]
    intensities_avg = line_data_avg["normalized flux"]
    intensities_rms = line_data_rms["normalized flux"]

    intensity_avg, velocity_avg = cut_normalized_line_out(intensities_avg, velocitys, range)
    intensity_rms, velocity_rms = cut_normalized_line_out(intensities_rms, velocitys, range)

    range_str = f"{range[0]}_{range[1]}"

    save_velocity_data_to_txt(output_path / f"{line_name}_norm_avg_profile_{range_str}.txt", velocity_avg, intensity_avg)
    save_velocity_data_to_txt(output_path / f"{line_name}_norm_rms_profile_{range_str}.txt", velocity_rms, intensity_rms)

    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        axs[0].plot(velocity_avg, intensity_avg, linestyle='-', marker='o', color='b')
        axs[0].set_ylabel("Avg Profile")
        axs[0].grid(True)

        axs[1].plot(velocity_rms, intensity_rms, linestyle='--', marker='s', color='r')
        axs[1].set_xlabel("Velocity (km/s)")
        axs[1].set_ylabel("RMS Profile")
        axs[1].grid(True)

        plt.suptitle(f"Line Profile: {line_name}")
        plt.show()


@task
def cut_line_profile(line_name, cut_out_range=(-100, 100),  plot=False, output_dir=DEFAULT_OUTPUT_DIR):
    if isinstance(cut_out_range, str):
        cut_out_range = tuple(map(int, cut_out_range.strip("() ").split(',')))

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / "Line_Profiles_cut_out"
    output_path.mkdir(parents=True, exist_ok=True)

    central_wave_length = CENTRAL_WAVELENGTH[line_name]

    cut_out_range = (central_wave_length + cut_out_range[0], central_wave_length + cut_out_range[1])

    line_profile_dict = import_line_profile_data()

    line_data_avg = line_profile_dict["avg"][line_name]["data_dict"]
    line_data_rms = line_profile_dict["rms"][line_name]["data_dict"]

    wavelengths = line_data_avg["velocity space (km/s)"]
    intensities_avg = line_data_avg['flux ergs/s/cm2/A']
    intensities_rms = line_data_rms['flux ergs/s/cm2/A']

    intensity_avg, velocity_avg = cut_line_out(intensities_avg, wavelengths, cut_out_range)
    intensity_rms, velocity_rms = cut_line_out(intensities_rms, wavelengths, cut_out_range)

    range_str = f"{cut_out_range[0]}_{cut_out_range[1]}"

    save_velocity_data_to_txt(output_path / f"{line_name}_avg_profile_{range_str}.txt", velocity_avg, intensity_avg)
    save_velocity_data_to_txt(output_path / f"{line_name}_rms_profile_{range_str}.txt", velocity_rms, intensity_rms)

    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        axs[0].plot(velocity_avg, intensity_avg, linestyle='-', marker='o', color='b')
        axs[0].set_ylabel("Avg Profile")
        axs[0].grid(True)

        axs[1].plot(velocity_rms, intensity_rms, linestyle='--', marker='s', color='r')
        axs[1].set_xlabel("Velocity (km/s)")
        axs[1].set_ylabel("RMS Profile")
        axs[1].grid(True)

        plt.suptitle(f"Line Profile: {line_name}")
        plt.show()



@task
def highest_corr_coef():
    one_dim_correlation_data = import_1d_correlation_data()
    sorted_one_dim_correlation_plot_data = sort_1d_corr_data_for_lines(one_dim_correlation_data)

    campaign_result_dict = dict()
    for campaign, lines_data in sorted_one_dim_correlation_plot_data.items():
        campaign_result_dict[campaign] = get_weighted_best_continua(get_continua_with_highest_corr_coef(lines_data))

    print()


def run_task(commands):
    for command in commands:
        # try:
        # Split command name and additional arguments
        parts = command.split("::")
        name_of_task = parts[0]
        task_args = parts[1:] if len(parts) > 1 else []

        print(f"Running {name_of_task} with arguments {task_args}...")

        # Call the command with unpacked arguments
        registered_tasks[name_of_task](*task_args)
    # except KeyError as k:
    # print(f"Task '{command}' is not available.")
    # except Exception as e:
    # print(f"An error occurred while running '{command}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in registered_tasks:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])
