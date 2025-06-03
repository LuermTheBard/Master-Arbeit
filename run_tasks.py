import sys

from plot_data.plot_fits_data import plot_avg_rms_spec
from plot_data.plot_1D_lightcurves_in_groups_data import (
    plot_1d_lightcurves_in_groups,
    save_1d_lightcurves_in_groups
)
from plot_data.plot_1D_ccfs_in_groups_data import (
    plot_1d_corr_in_groups_for_cont_optical_calibrated,
    save_1d_corr_in_groups_for_cont_optical_calibrated,
    save_1d_corr_in_groups_bowen_fluorescence_for_cont
)
from plot_data.plot_1D_ccfs_and_reference_lightcurves import (
    save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines,
    save_1d_corr_and_lightcurves_in_groups_for_UVW2,
    save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha
)
from plot_data.plot_line_profiles_in_groups import (
    run_normalized_profiles_together_in_groups,
    substract_pseudo_continua_from_spectra,
    cut_line_profile
)

# Central registry of task names (strings)
TASK_LIST = [
    'plot_avg_rms_spec',
    # plot lightcurves
    'plot_1d_lightcurves_in_groups',
    'save_1d_lightcurves_in_groups',
    # plot/save ccfs
    'plot_1d_corr_in_groups_for_cont_optical_calibrated',
    'save_1d_corr_in_groups_for_cont_optical_calibrated',
    # save ccfs for bowen fluorescence
    'save_1d_corr_in_groups_bowen_fluorescence_for_cont',
    # save lightcurves and ccfs in one plot
    'save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines',
    'save_1d_corr_and_lightcurves_in_groups_for_UVW2',
    'save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha',
    # plot line profiles
    'run_normalized_profiles_together_in_groups',
    # extract line profiles
    'substract_pseudo_continua_from_spectra',
    'cut_line_profile'
]

# Lookup table for function names → function objects
task_lookup = {
    'plot_avg_rms_spec': plot_avg_rms_spec,
    'plot_1d_lightcurves_in_groups': plot_1d_lightcurves_in_groups,
    'save_1d_lightcurves_in_groups': save_1d_lightcurves_in_groups,
    'plot_1d_corr_in_groups_for_cont_optical_calibrated': plot_1d_corr_in_groups_for_cont_optical_calibrated,
    'save_1d_corr_in_groups_for_cont_optical_calibrated': save_1d_corr_in_groups_for_cont_optical_calibrated,
    'save_1d_corr_in_groups_bowen_fluorescence_for_cont': save_1d_corr_in_groups_bowen_fluorescence_for_cont,
    'save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines': save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines,
    'save_1d_corr_and_lightcurves_in_groups_for_UVW2': save_1d_corr_and_lightcurves_in_groups_for_UVW2,
    'save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha': save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha,
    'run_normalized_profiles_together_in_groups': run_normalized_profiles_together_in_groups,
    'substract_pseudo_continua_from_spectra': substract_pseudo_continua_from_spectra,
    'cut_line_profile': cut_line_profile
}


def run_task(commands):
    for command in commands:
        parts = command.split("::")
        name_of_task = parts[0]
        task_args = parts[1:] if len(parts) > 1 else []

        print(f"Running '{name_of_task}' with arguments {task_args}...")

        if name_of_task not in task_lookup:
            print(f"[ERROR] Task '{name_of_task}' is not defined.")
            continue

        func = task_lookup[name_of_task]
        func(*task_args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in TASK_LIST:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])

