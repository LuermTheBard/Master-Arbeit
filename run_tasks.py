import sys
from pathlib import Path

import numpy as np

from import_data.import_data import import_1d_correlation_data, import_1d_lightcurve_data, import_fits_data, \
    import_line_profile_data
from plot_data.plot_1D_ccfs_and_reference_lightcurves import plot_1d_corr_and_lightcurves_in_groups
from plot_data.plot_1D_ccfs_in_groups_data import plot_all_1d_ccfs_in_groups_for_cont
from plot_data.plot_1D_lightcurves_in_groups_data import plot_all_1d_lightcurves_in_groups
from plot_data.plot_fits_data import plot_avg_rms
from plot_data.plot_line_profiles_in_groups import plot_normalized_line_profiles_in_groups, process_spectrum, \
    plot_cut_out_line_profile, cut_normalized_line_out, cut_line_out
from settings import DEFAULT_OUTPUT_DIR, CENTRAL_WAVELENGTH

# Dictionary to store registered tasks
registered_tasks = {}


# Decorator to register functions as tasks
def task(func):
    registered_tasks[func.__name__] = func
    return func


# Reusable helper to ensure directory exists

def ensure_output_dir(output_dir):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    return output_dir_path


@task
def plot_avg_rms_spec(output_dir=DEFAULT_OUTPUT_DIR):
    avg_rms_spec_dir = output_dir / "avg_rms_spec"

    avg_rms_spec_dir.mkdir(parents=True, exist_ok=True)

    avg_rms_spec_file = avg_rms_spec_dir

    fits_data = import_fits_data()
    plot_avg_rms(fits_data, save_path=avg_rms_spec_file)


def run_1d_lightcurves_groups(output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    ensure_output_dir(output_dir)
    data = import_1d_lightcurve_data()
    for cont in ["Cont1150_not_optical_calibrated", "Cont5100"]:
        key_order_lines = [cont, 'HAlpha', 'HBeta', 'HGamma', 'LyAlpha_not_optical_calibrated', 'HeI5875', 'HeII4685', 'OI8446']
        key_order_conts = ["Cont1150_not_optical_calibrated", "Cont4010", "Cont4440", "Cont5100", "Cont6110", "Cont6880", "Cont8015", "Cont8900"]
        for campaign, data_dict in data.items():
            plot_all_1d_lightcurves_in_groups(data_dict, campaign, output_dir, compare_cont=cont, key_order_lines=key_order_lines, key_order_conts=key_order_conts, save_only=save_only)



@task
def plot_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    run_1d_lightcurves_groups(output_dir)


@task
def save_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    run_1d_lightcurves_groups(output_dir, save_only=True)



@task
def plot_1d_corr_in_groups_for_cont_optical_calibrated(cont_name=None, output_dir=DEFAULT_OUTPUT_DIR):
    
    # Prüfen, ob cont_name definiert ist
    if not cont_name:
        raise ValueError("Please specify a cont_name in the following form: plot_line_1d_corr::cont_name")

    ensure_output_dir(output_dir)

    key_order = ["time shift (tau)", 'HAlpha', 'HBeta', 'HGamma', 'HDelta', "LyAlpha_not_optical_calibrated", 'HeI5875',  'HeII4685', 'OI8446']
    one_dim_correlation_data = import_1d_correlation_data()

    plot_all_1d_ccfs_in_groups_for_cont(one_dim_correlation_data["NGC4593_optical_calibrated"], "NGC4593_optical_calibrated", cont_name=cont_name, output_dir=output_dir,
                                            key_order=key_order)


@task
def save_1d_corr_in_groups_for_cont_optical_calibrated(cont_name=None, output_dir=DEFAULT_OUTPUT_DIR):
    

    # Prüfen, ob cont_name definiert ist
    if not cont_name:
        raise ValueError("Please specify a cont_name in the following form: plot_line_1d_corr::cont_name")

    ensure_output_dir(output_dir)

    key_order = ["time shift (tau)", 'HAlpha', 'HBeta', 'HGamma', 'HDelta', "LyAlpha_not_optical_calibrated", 'HeI5875', 'HeII4685', 'OI8446']
    # key_order = ["time shift (tau)", 'HeI5015', 'HeI5875', 'HeI4471', 'HeI7065', 'HeII4685','HAlpha', 'HBeta', 'HGamma', 'HDelta',  'OI8446']
    one_dim_correlation_data = import_1d_correlation_data()

    plot_all_1d_ccfs_in_groups_for_cont(one_dim_correlation_data["NGC4593_optical_calibrated"],
                                        "NGC4593_optical_calibrated", cont_name=cont_name, output_dir=output_dir,
                                        key_order=key_order, save_only=True)


@task
def save_1d_corr_in_groups_bowen_fluorescence_for_cont(output_dir=DEFAULT_OUTPUT_DIR):

    ensure_output_dir(output_dir)

    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    key_order_cont1150 = ["time shift (tau)", 'HAlpha', 'HBeta', "LyAlpha_not_optical_calibrated", 'OI8446']
    key_order_cont1460 = ["time shift (tau)", 'HAlpha', 'HBeta', "LyAlpha_not_optical_calibrated", 'OI8446']
    key_order_lyalpha = ["time shift (tau)", 'HAlpha', 'HBeta', 'OI8446']
    key_order_halpha = ["time shift (tau)", 'OI8446']
    key_order_hbeta= ["time shift (tau)", 'OI8446']

    keyorders = {"Cont1150_not_optical_calibrated": key_order_cont1150,
                 "Cont1460_not_optical_calibrated": key_order_cont1460,
                 "LyAlpha_not_optical_calibrated": key_order_lyalpha,
                 "HAlpha": key_order_halpha,
                 "HBeta": key_order_hbeta}

    one_dim_correlation_data = import_1d_correlation_data()
    for campaign, data_dict in one_dim_correlation_data.items():
        for reference_lightcurve, key_order in keyorders.items():
            plot_all_1d_ccfs_in_groups_for_cont(data_dict, campaign, cont_name=reference_lightcurve, output_dir=output_dir,
                                                key_order=key_order, save_only=True, file_name=f"{reference_lightcurve}_bowen_fluorescence_ccfs", only_key_order=True)



# methodes to plot lightcurves and ccfs together
@task
def save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines(output_dir=DEFAULT_OUTPUT_DIR):

    ensure_output_dir(output_dir)

    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    key_order_cont1150 = ["time shift (tau)",
                          'HAlpha',
                          'HBeta',
                          "LyAlpha_not_optical_calibrated",
                          'OI8446']
    # key_order_cont1460 = ["time shift (tau)", 'HAlpha', 'HBeta', "LyAlpha_not_optical_calibrated", 'OI8446']
    key_order_lyalpha = ["time shift (tau)", 'HAlpha', 'HBeta', 'OI8446']
    key_order_halpha = ["time shift (tau)", 'OI8446']
    key_order_hbeta= ["time shift (tau)", 'OI8446']

    keyorders = {"Cont1150_not_optical_calibrated": key_order_cont1150,
                 # "Cont1460_not_optical_calibrated": key_order_cont1460,
                 "LyAlpha_not_optical_calibrated": key_order_lyalpha,
                 "HAlpha": key_order_halpha,
                 "HBeta": key_order_hbeta}

    final_sorted_keys = ["time shift (tau)", 'OI8446', "LyAlpha_not_optical_calibrated", 'HBeta', 'HAlpha']

    one_dim_correlation_data = import_1d_correlation_data()
    lightcurves_data = import_1d_lightcurve_data()
    for campaign, data_dict in one_dim_correlation_data.items():
        lightcurves_ccfs_dict = {"lightcurves": lightcurves_data[campaign], "ccfs": data_dict}
        plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccfs_dict, campaign, output_dir, keyorders, file_name="ccfs_and_reference_lightcurves", final_key_order=final_sorted_keys)


@task
def save_1d_corr_and_lightcurves_in_groups_for_UVW2(output_dir=DEFAULT_OUTPUT_DIR):

    ensure_output_dir(output_dir)

    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    key_order_UVW2_optical  = ["time shift (tau)", 'HAlpha', 'HBeta', 'HGamma', 'HDelta', "LyAlpha_not_optical_calibrated", 'HeI5875', 'HeII4685', 'OI8446']

    key_order_UVW2_UV = ["time shift (tau)", "SiIV1393_not_optical_calibrated", "NV1238_not_optical_calibrated", "CIV1548_not_optical_calibrated", "HeII1640_not_optical_calibrated", "OIII]1660_not_optical_calibrated"]



    keyorders_optical = {"UVW2": key_order_UVW2_optical,}


    keyorders_UV = {"UVW2": key_order_UVW2_UV, }

    keyorders_dict = {"NGC4593_optical_calibrated": keyorders_optical, "NGC4593_not_optical_calibrated": keyorders_UV}


    one_dim_correlation_data = import_1d_correlation_data()
    lightcurves_data = import_1d_lightcurve_data()
    for campaign, data_dict in one_dim_correlation_data.items():
        lightcurves_ccfs_dict = {"lightcurves": lightcurves_data[campaign], "ccfs": data_dict}
        plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccfs_dict, campaign, output_dir, keyorders_dict[campaign], file_name="ccfs_and_reference_lightcurves", final_key_order=keyorders_dict[campaign])


@task
def save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha(output_dir=DEFAULT_OUTPUT_DIR):

    ensure_output_dir(output_dir)

    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)


    key_order_UVW2 = ["time shift (tau)",
                      "LyAlpha_not_optical_calibrated",
                      "NV1238_not_optical_calibrated",
                      "SiIV1393_not_optical_calibrated",
                      "CIV1548_not_optical_calibrated",
                      "HeII1640_not_optical_calibrated",
                      "HDelta",
                      "HGamma",
                      'HeII4685',
                      "HBeta",
                      'HeI5875',
                      "HAlpha"
                      ]

    keyorders= {"UVW2": key_order_UVW2, }


    one_dim_correlation_data = import_1d_correlation_data()
    lightcurves_data = import_1d_lightcurve_data()

    lightcurves_ccfs_dict_combined = {
        "lightcurves": {
            "lines": {
                **lightcurves_data["NGC4593_optical_calibrated"]["lines"],
                **lightcurves_data["NGC4593_not_optical_calibrated"]["lines"]
            },
            "continua": {
                **lightcurves_data["NGC4593_optical_calibrated"]["continua"],
                **lightcurves_data["NGC4593_not_optical_calibrated"]["continua"]
            }
        },
        "ccfs": {"UVW2":{**one_dim_correlation_data["NGC4593_optical_calibrated"]["UVW2"],
                 **one_dim_correlation_data["NGC4593_not_optical_calibrated"]["UVW2"]}}
    }



    plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccfs_dict_combined, "NGC4593_Combined", output_dir, keyorders,
                                           file_name="ccfs_and_reference_lightcurves", final_key_order=keyorders,
                                           rows=11, cols=2, only_one_label = True)

@task
def run_normalized_profiles_together_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    ensure_output_dir(output_dir)

    profile_data = import_line_profile_data(normalized=True)

    # key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', "LyAlpha_not_optical_calibrated", 'HeI5875', 'HeII4685', 'OI8446']
    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeII4685',
                 'OI8446']

    plot_normalized_line_profiles_in_groups(profile_data, key_order=key_order)


@task
def substract_pseudo_continua_from_spectra(plot=False, output_dir=DEFAULT_OUTPUT_DIR):
    ensure_output_dir(output_dir)

    fits_data = import_fits_data()

    wavelenghts = np.array(fits_data['NGC4593_avg.fits']['x_axis'][0])
    avg_data = np.array(fits_data['NGC4593_avg.fits']['data'][0])
    rms_data = np.array(fits_data['NGC4593_rms.fits']['data'][0])

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446', "OIII5007"]

    for line in key_order:
        process_spectrum(wavelenghts, avg_data, line, spec_type="avg", output_dir=output_dir, plot=plot)
        process_spectrum(wavelenghts, rms_data, line, spec_type="rms", output_dir=output_dir, plot=plot)


@task
def cut_line_profile(
    line_name,
    cut_out_range=(-1000, 1000),
    normalized=True,
    plot=False,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    def str_to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return False  # fallback

    normalized = str_to_bool(normalized)
    plot = str_to_bool(plot)

    if isinstance(cut_out_range, str):
        cut_out_range = tuple(map(int, cut_out_range.strip("() ").split(',')))

    # Sicherstellen, dass Hauptausgabeverzeichnis existiert
    output_dir_path = ensure_output_dir(output_dir)

    # Sicherstellen, dass Unterordner existiert
    output_path = output_dir_path / "Line_Profiles_cut_out"
    output_path.mkdir(parents=True, exist_ok=True)

    # Daten importieren
    line_profile_dict = import_line_profile_data(normalized=normalized)
    line_data_avg = line_profile_dict["avg"][line_name]["data_dict"]
    line_data_rms = line_profile_dict["rms"][line_name]["data_dict"]

    if normalized is True:
        velocity = line_data_avg.get("velocity space (km/s)")
        intensities_avg = line_data_avg.get("normalized flux")
        intensities_rms = line_data_rms.get("normalized flux")
        intensity_avg, velocity_avg = cut_normalized_line_out(intensities_avg, velocity, cut_out_range)
        intensity_rms, velocity_rms = cut_normalized_line_out(intensities_rms, velocity, cut_out_range)
    else:
        central_wave_length = CENTRAL_WAVELENGTH[line_name]
        cut_range_absolute = (
            central_wave_length + cut_out_range[0],
            central_wave_length + cut_out_range[1]
        )
        wavelengths = line_data_avg.get("velocity space (km/s)")
        intensities_avg = line_data_avg.get('flux ergs/s/cm2/A')
        intensities_rms = line_data_rms.get('flux ergs/s/cm2/A')
        intensity_avg, velocity_avg = cut_line_out(intensities_avg, wavelengths, cut_range_absolute)
        intensity_rms, velocity_rms = cut_line_out(intensities_rms, wavelengths, cut_range_absolute)

    # Plotten/Speichern
    plot_cut_out_line_profile(
        cut_out_range, intensity_avg, intensity_rms, line_name,
        output_path, plot, velocity_avg, velocity_rms
    )

def run_task(commands):
    for command in commands:

        parts = command.split("::")
        name_of_task = parts[0]
        task_args = parts[1:] if len(parts) > 1 else []

        print(f"Running {name_of_task} with arguments {task_args}...")

        registered_tasks[name_of_task](*task_args)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in registered_tasks:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])
