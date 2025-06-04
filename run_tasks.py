import sys
from inspect import signature

from plot_avg_rms import plot_avg_rms_spec
from plot_1D_lightcurves_in_groups import (
    plot_1d_lightcurves_in_groups,
    save_1d_lightcurves_in_groups
)
from plot_1D_ccfs_in_groups import (
    plot_1d_corr_in_groups, save_1d_corr_in_groups,
    save_1d_corr_in_groups_bowen_fluorescence_for_cont
)
from plot_1D_ccfs_and_reference_lightcurves import (
    save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines,
    save_1d_corr_and_lightcurves_in_groups_for_UVW2,
    save_1d_corr_and_lightcurves_in_groups_UVW2_from_UV_Lines_to_HAlpha
)
from plot_line_profiles_in_groups import (
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
    'plot_1d_corr_in_groups': plot_1d_corr_in_groups,
    'save_1d_corr_in_groups': save_1d_corr_in_groups,
    'save_1d_corr_in_groups_bowen_fluorescence_for_cont': save_1d_corr_in_groups_bowen_fluorescence_for_cont,
    'save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines': save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines,
    'save_1d_corr_and_lightcurves_in_groups_for_UVW2': save_1d_corr_and_lightcurves_in_groups_for_UVW2,
    'save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha': save_1d_corr_and_lightcurves_in_groups_UVW2_from_UV_Lines_to_HAlpha,
    'run_normalized_profiles_together_in_groups': run_normalized_profiles_together_in_groups,
    'substract_pseudo_continua_from_spectra': substract_pseudo_continua_from_spectra,
    'cut_line_profile': cut_line_profile
}


def run_task(argv):
    task_name = argv[0]
    args = argv[1:]

    if task_name not in task_lookup:
        print(f"Task '{task_name}' not found.")
        return

    task_func = task_lookup[task_name]
    sig = signature(task_func)

    required_params = [
        p for p in sig.parameters.values()
        if p.default == p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    if len(args) < len(required_params):
        print(f"Task '{task_name}' requires at least {len(required_params)} argument(s), but {len(args)} were given.")

        # Format usage string with required and optional parameters
        usage_parts = []
        for p in sig.parameters.values():
            if p.default == p.empty:
                usage_parts.append(f"<{p.name}>")
            else:
                usage_parts.append(f"[{p.name}]")

        usage_string = f"Usage: python script.py {task_name} {' '.join(usage_parts)}"
        print(usage_string)
        return

    try:
        task_func(*args)
    except Exception as e:
        print(f"Error running task '{task_name}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify at least one task name, or use 'list_tasks' to see all available tasks.")
    elif sys.argv[1] == "list_tasks":
        print("Available tasks:")
        for task_name in TASK_LIST:
            print(f"- {task_name}")
    else:
        run_task(sys.argv[1:])
