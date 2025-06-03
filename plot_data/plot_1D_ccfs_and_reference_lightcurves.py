import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from import_data.import_data import import_1d_correlation_data, import_1d_lightcurve_data
from plot_utils import format_label, calculate_standard_error_for_lightcurves, ensure_output_dir
from plot_data.general_plot import finalize_figure, format_yaxis, format_month_day
from settings import BASE_MJD, DEFAULT_OUTPUT_DIR

matplotlib.use('Qt5Agg')


def plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccf_data_dict, campaign, output_dir, key_orders, save_only=False, file_name=None, final_key_order=None, rows=4, cols=2, only_one_label=False):
    """
    Organizes and plots CFFs and their corresponding lightcurves
    in subplot groups, based on specified key orders.

    Parameters:
    -----------
    lightcurves_ccf_data_dict : dict
        Dictionary containing CCF and lightcurve data structured as:
        {"ccfs": ..., "lightcurves": {"continua": ..., "lines": ...}}.
    campaign : str
        Campaign identifier used for folder structure.
    output_dir : str or pathlib.Path
        Base directory for saving plots.
    key_orders : dict
        Dictionary mapping each reference lightcurve to a list specifying the order of keys to plot.
    save_only : bool, optional
        If True, the plots are saved to disk without displaying them. Default is False.
    file_name : str, optional
        Optional base name for output files. If None, a default name is generated from the title.
    final_key_order : list of str, optional
        Final desired order of plotted keys (e.g., lines), used for sorting.
    rows : int, optional
        Number of subplot rows per figure. Default is 4.
    cols : int, optional
        Number of subplot columns per figure. Default is 2.
    only_one_label : bool, optional
        If True, only one y-axis label per figure is shown. Default is False.

    Returns:
    -----------
    None
    """

    base_mjd = BASE_MJD
    xlabel_ccfs = "Time Lag $\\tau$ [d]"
    ylabel_ccfs = "Correlation Coefficient"

    xlabel_lightcurves = f"MJD - {base_mjd:.2f}"
    ylabel_cont_lightcurves = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                   r"\mathrm{\AA}^{-1}]$")
    ylabel_line_lightcurves = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1}]$")
    yerr_name_lightcurves = 'fluxerrs [ergs/s/cm2/A]'




    save_folder = output_dir / campaign / "plot_1d_ccfs" / "corr_and_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    all_sorted_data_dict = dict()

    first = True
    for reference_lightcurve, key_order in key_orders.items():

        def sort_keys(key):
            for idx, prefix in enumerate(key_order):
                if key == prefix:
                    return idx
            return len(key_order)

        try:
            sorted_data_dict = dict(
                sorted(lightcurves_ccf_data_dict["ccfs"][reference_lightcurve].items(),
                       key=lambda item: sort_keys(item[0])))


            keys_to_keep = key_order[1:]
            if first:
                keys_to_keep = ["time shift (tau)"] + keys_to_keep
                first = False

            sorted_data_dict = {
                (k if k == "time shift (tau)" else k + "_ref_" + reference_lightcurve): v
                for k, v in sorted_data_dict.items()
                if k in keys_to_keep
            }

        except KeyError:
            print(f"[Warning] Continuum name '{reference_lightcurve}' not found in campaign '{campaign}'. Skipping.")
            continue

        all_sorted_data_dict.update(sorted_data_dict)

    final_sorted_data_dict=dict()

    for key, value in all_sorted_data_dict.items():

        if key == "time shift (tau)":
            final_sorted_data_dict.update({key:value})
            continue

        line, reference = key.split("_ref_")

        if "Cont" in reference:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][reference]
        else:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][reference]

        if "Cont" in line:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][line]
        else:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][line]

        final_sorted_data_dict.update({key:{"ccfs":value,"lightcurves":lightcurve_data,"lightcurves_ref":lightcurve_reference_data}})

    def final_sort_keys(key):
        for idx, prefix in enumerate(final_key_order):
            if prefix == key.split("_ref_")[0]:
                return idx
        return len(key_order)

    final_sorted_data_dict = dict(
        sorted(final_sorted_data_dict.items(),
               key=lambda item: final_sort_keys(item[0])))


    plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs, xlabel_lightcurves,
                                                  title=f"CCFs and reference lightcurves",
                                                  save_only=save_only, output_dir=save_folder, shared_y=False, file_name=file_name, rows=rows, cols=cols, only_one_label = only_one_label)


def plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs,
                                                  xlabel_lightcurves, title, save_only, output_dir, shared_y,
                                                  file_name, rows=4, cols=2, only_one_label=False):
    """
    Plots CCFs and their associated normalized lightcurves
    in a side-by-side subplot layout.

    Parameters:
    -----------
    final_sorted_data_dict : dict
        Dictionary containing the CCFs and lightcurve pairs. Must include a key
        'time shift (tau)' for the CCF x-axis.
    xlabel_ccfs : str
        X-axis label for CCF subplots (right column).
    ylabel_ccfs : str
        Y-axis label for CCF subplots.
    xlabel_lightcurves : str
        X-axis label for lightcurve subplots (left column).
    title : str
        Title for the entire figure or plot group.
    save_only : bool
        If True, plots are saved to disk only; if False, they are also displayed.
    output_dir : str or pathlib.Path
        Directory to save the output files.
    shared_y : bool
        Whether subplots should share the same y-axis limits.
    file_name : str
        Optional base name for the saved files (PDF and PNG).
    rows : int, optional
        Number of subplot rows per figure. Default is 4.
    cols : int, optional
        Number of subplot columns per figure. Default is 2.
    only_one_label : bool, optional
        If True, only one label per y-axis (left/right) is shown to avoid clutter.

    Returns:
    -----------
    None
    """
    x_values_ccfs = final_sorted_data_dict['time shift (tau)']
    final_sorted_data_dict.pop('time shift (tau)')


    for current_data, group_index in prepare_ccfs_references_data(final_sorted_data_dict, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=False, sharey=shared_y,
                                 gridspec_kw={'width_ratios': [4, 1]})  # 2/3 : 1/3 Verhältnis
        fig.subplots_adjust(hspace=0, wspace=0)
        #fig.tight_layout()
        if only_one_label is True:
            # Linke Seite oben (y-Achse): "Normalized Lightcurves"
            fig.text(0.02, 0.5, "Normalized Lightcurves", va='center', ha='left', rotation='vertical', fontsize=12)

            # Rechte Seite unten (y-Achse): ylabel_ccfs
            fig.text(0.98, 0.5, ylabel_ccfs, va='center', ha='right', rotation='vertical', fontsize=12)

        for i, (line_name, line_data) in enumerate(current_data):

            row, col = divmod(i, cols)

            ax = axes[row, col]

            if line_data is not None:
                if i % 2 == 0:
                    color = ("blue", "orange")
                    yerr = True
                else:
                    color = "black"
                    yerr = None
            else:
                yerr = None
                line_data = np.array([])
                color = "black"

            configure_ccfs_and_reference_axis(ax, row, col, ylabel_ccfs, color, x_values_ccfs, line_data, yerr, line_name, only_one_label = only_one_label)


        finalize_figure(fig, axes, x_label=(xlabel_lightcurves, xlabel_ccfs), title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir, file_name=file_name)


def prepare_ccfs_references_data(data, rows, cols):
    """
    Prepares and groups the input data into (rows x cols) chunks for subplot grids.
    Each item is duplicated so that both a lightcurve and a CCF version can be plotted.
    Incomplete groups are padded with placeholder entries.

    Parameters:
    -----------
    data : dict
        Dictionary containing CCF/lightcurve data. Keys are line-reference combinations.
    rows : int
        Number of rows in each subplot group.
    cols : int
        Number of columns in each subplot group.

    Yields:
    -----------
    current_data : list of tuples
        Subset of (key, value) pairs for one subplot group.
    group_index : int
        Index of the current group (starting from 0).
    """

    # Jedes Item zweimal hintereinander einfügen
    data_items = []
    for key, value in data.items():
        data_items.append((key, value))
        data_items.append((key, value))  # direkt danach nochmal

    total_plots = len(data_items)
    plots_per_group = rows * cols
    num_groups = (total_plots + plots_per_group - 1) // plots_per_group

    for group_index in range(num_groups):
        start_index = group_index * plots_per_group
        end_index = min(start_index + plots_per_group, total_plots)
        current_data = data_items[start_index:end_index]

        # Mit Platzhaltern auffüllen, falls unvollständig
        while len(current_data) < plots_per_group:
            current_data.append((
                f'Empty {len(current_data) + 1}',
                None
            ))

        yield current_data, group_index



def configure_ccfs_and_reference_axis(ax, row, col, ylabel_ccfs, color, x_values_ccfs, line_data, yerr,
                                      line_name_and_ref_name, only_one_label=False):
    """
    Configures a single subplot axis to display either a normalized lightcurve pair
    or a CCF, depending on the data provided.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The subplot axis to configure.
    row : int
        Row index within the subplot grid.
    col : int
        Column index within the subplot grid.
    ylabel_ccfs : str
        Y-axis label for the CCFs (used for the right column).
    color : tuple or str
        Tuple of two colors (for lightcurve plots) or single color (for CCFs).
    x_values_ccfs : array-like
        X-axis values for the CCF plot.
    line_data : dict or np.ndarray
        Dictionary with lightcurve and CCF data or an empty array for placeholders.
    yerr : bool or None
        Whether to include error bars (used for lightcurve plots).
    line_name_and_ref_name : str
        Name of the line and its reference, formatted as "<line>_ref_<reference>".
    only_one_label : bool, optional
        If True, y-axis labels are minimized to reduce clutter.

    Returns:
    -----------
    None
    """

    if "_ref_" in line_name_and_ref_name:

        line_name, reference_name = line_name_and_ref_name.split("_ref_")
    else:
        line_name = None
        reference_name = None

    if len(line_data) == 0:
        return

    if yerr is not None:
        x_key = 'timestamps [MJD]'
        y_key = 'fluxes [ergs/s/cm2/A]'
        yerr_key = 'fluxerrs [ergs/s/cm2/A]'

        y_norm, yerr_norm = normalize_lightcurve(line_data["lightcurves"][y_key], line_data["lightcurves"][yerr_key])
        y_ref_norm, yerr_ref_norm = normalize_lightcurve(line_data["lightcurves_ref"][y_key],
                                                          line_data["lightcurves_ref"][yerr_key])

        plot_normalized_lightcurve(ax, line_data["lightcurves"][x_key], y_norm, yerr_norm,
                                   format_label(line_name, as_latex=False), color[0])
        plot_normalized_lightcurve(ax, line_data["lightcurves_ref"][x_key], y_ref_norm, yerr_ref_norm,
                                   format_label(reference_name, as_latex=False), color[1])

        configure_axes_for_lightcurves(ax, row, col, only_one_label)
        ax.legend(fontsize=8)
    else:
        ax.plot(x_values_ccfs, line_data["ccfs"], color=color)
        ax.text(9.5,0.95, format_label(line_name, as_latex=False),  ha='right', va='top', fontsize=8)
        configure_axes_for_ccfs(ax, row, col, ylabel_ccfs, only_one_label)


def normalize_lightcurve(y, yerr_vals):
    yerr_vals = calculate_standard_error_for_lightcurves(y, yerr_vals)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    yerr_norm = yerr_vals / y_std
    return y_norm, yerr_norm


def plot_normalized_lightcurve(ax, x, y_norm, yerr_norm, label, color):
    ax.errorbar(x, y_norm, yerr=yerr_norm, fmt='.:', capsize=3, markersize=4, label=label, color=color)


def configure_axes_for_lightcurves(ax, row, col, only_one_label=False):
    """
    Configures axis formatting and ticks for a normalized lightcurve subplot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to configure.
    row : int
        Row index in the subplot grid.
    col : int
        Column index in the subplot grid.
    only_one_label : bool, optional
        If True, hides redundant y-axis labels for cleaner layout.

    Returns:
    -----------
    None
    """

    if col == 0:
        if only_one_label is False:
            ax.set_ylabel("Normalized Lightcurves", fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(-2.99, 4.499)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=10)


def configure_axes_for_ccfs(ax, row, col, ylabel_ccfs, only_one_label=False):
    """
    Configures axis formatting and ticks for a cross-correlation function (CCF) subplot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to configure.
    row : int
        Row index in the subplot grid.
    col : int
        Column index in the subplot grid.
    ylabel_ccfs : str
        Y-axis label for the CCFs (applied to right column).
    only_one_label : bool, optional
        If True, limits y-axis labeling to reduce clutter.

    Returns:
    -----------
    None
    """

    ax.axvline(x=0, color='black', linestyle=':', linewidth=0.5)
    ax.set_xlim(-4.999, 9.999)
    ax.set_ylim(0, 0.999)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    if col == 0:
        if only_one_label is False:
            ax.set_ylabel("Lightcurves", fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    else:
        ax.yaxis.tick_right()
        if only_one_label is False:
            ax.set_ylabel(ylabel_ccfs, fontsize=12)
            ax.yaxis.set_label_position("right")
        ax_right = ax.secondary_yaxis('right')
        ax_right.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_right.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    if row < 3:
        ax.set_xticklabels([])

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(2))
        ax_top.tick_params(axis='x')




# methods to run:

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
    lightcurves_ccfs_dict = {"lightcurves": lightcurves_data["NGC4593_optical_calibrated"], "ccfs": one_dim_correlation_data["NGC4593_optical_calibrated"]}
    plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccfs_dict, "NGC4593_optical_calibrated", output_dir, keyorders, file_name="ccfs_and_reference_lightcurves", final_key_order=final_sorted_keys)


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



# save_1d_corr_and_lightcurves_in_groups_for_UVW2()
# save_1d_corr_and_lightcurves_in_groups_UVW2_form_UV_Lines_to_HAlpha()
save_1d_corr_and_lightcurves_in_groups_for_bowen_fluorescence_lines()

