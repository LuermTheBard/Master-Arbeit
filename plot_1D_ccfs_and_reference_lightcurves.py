import datetime
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from import_data import import_1d_correlation_data, import_1d_lightcurve_data, load_centroid_data_by_reference, \
    import_centroid_and_mc_data
from plot_utils import format_label, calculate_standard_error_for_lightcurves, ensure_output_dir
from settings import SYMBOLES_AND_COLORS_FOR_LIGHTCURVES, NUMBER_MAPPING, ERR_CORRECTION, ERR_SET

# =======================
#   KONSTANTEN & EINSTELLUNGEN
# =======================

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "default_output"
matplotlib.use('Qt5Agg')


# =======================
#   DATENIMPORT & VORBEREITUNG
# =======================

def deep_merge(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def save_1d_corr_and_lightcurves_general(
        campaign_keys,
        keyorders_dict,
        output_dir=DEFAULT_OUTPUT_DIR,
        file_name="ccfs_and_reference_lightcurves",
        final_key_order=None,
        rows=8,
        cols=2,
        figsize=None,
        combine_data=False,
        campaign_label=None,
        show_reference_label=False,
        for_paper=False,
        extra_data_name=None,
        show_histogram=None):

    ensure_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_dim_correlation_data = import_1d_correlation_data()
    centroid_data = load_centroid_data_by_reference()
    lightcurves_data = import_1d_lightcurve_data()

    if combine_data:
        lightcurves_ccfs_dict = {
            "lightcurves": {
                "lines":
                    deep_merge(lightcurves_data["NGC4593_optical_calibrated"]["lines"],
                               lightcurves_data["NGC4593_not_optical_calibrated"]["lines"])
                ,
                "continua":
                    deep_merge(lightcurves_data["NGC4593_optical_calibrated"]["continua"],
                               lightcurves_data["NGC4593_not_optical_calibrated"]["continua"])

            },
            "ccfs": deep_merge(one_dim_correlation_data["NGC4593_optical_calibrated"],
                               one_dim_correlation_data["NGC4593_not_optical_calibrated"])
        }



        plot_1d_corr_and_lightcurves_in_groups(
            lightcurves_ccfs_dict,
            campaign_label or "Combined",
            output_dir,
            keyorders_dict,
            file_name=file_name,
            final_key_order=final_key_order or list(keyorders_dict["UVW2"]),
            rows=rows,
            cols=cols,
            figsize=figsize,
            centroid_data=centroid_data,
            only_one_label=True,
            show_reference_label=show_reference_label,
            for_paper=for_paper,
            extra_data_name=extra_data_name,
            show_histogram=show_histogram
        )

    else:
        for campaign in campaign_keys:
            lightcurves_ccfs_dict = {
                "lightcurves": lightcurves_data[campaign],
                "ccfs": one_dim_correlation_data[campaign]
            }

            plot_1d_corr_and_lightcurves_in_groups(
                lightcurves_ccfs_dict,
                campaign,
                output_dir,
                keyorders_dict[campaign],
                file_name=file_name,
                final_key_order=final_key_order or keyorders_dict[campaign],
                rows=rows,
                cols=cols,
                figsize=figsize,
                centroid_data=centroid_data,
                only_one_label=True,
                show_reference_label = show_reference_label,
                show_histogram=show_histogram

            )


# =======================
#   DATENVERARBEITUNG / SORTIERUNG
# =======================


def plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccf_data_dict, campaign, output_dir, key_orders, save_only=False, file_name=None, final_key_order=None, rows=4, cols=2, figsize=None, only_one_label=False, centroid_data=None, show_reference_label=False,
                                           for_paper=False, extra_data_name=None, show_histogram=None):
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

    xlabel_ccfs = "Time Lag $\\tau$ [d]"
    ylabel_ccfs = "Correlation Coefficient"

    xlabel_lightcurves = f"MJD"


    save_folder = output_dir / "corr_and_lightcurves"
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

    if for_paper and extra_data_name:

        line, reference = extra_data_name.split("_ref_")

        if "Cont" in reference:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][reference]
        else:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][reference]

        if "Cont" in line:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][line]
        else:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][line]

        extra_data = {extra_data_name:{"ccfs":lightcurves_ccf_data_dict["ccfs"][reference][line],"lightcurves":lightcurve_data,"lightcurves_ref":lightcurve_reference_data}}
        final_sorted_data_dict.update(extra_data)


    plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict,
                                                  xlabel_ccfs,
                                                  ylabel_ccfs,
                                                  xlabel_lightcurves,
                                                  centroid_data=centroid_data,
                                                  save_only=save_only,
                                                  output_dir=save_folder,
                                                  shared_y=False,
                                                  file_name=file_name + " " + campaign,
                                                  rows=rows,
                                                  cols=cols,
                                                  figsize=figsize,
                                                  only_one_label=only_one_label,
                                                  show_reference_label=show_reference_label,
                                                  for_paper=for_paper,
                                                  show_histogram=show_histogram)


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



def normalize_lightcurve(y, yerr_vals, err_correction=None, err_set=None):
    yerr_vals = calculate_standard_error_for_lightcurves(y, yerr_vals, err_correction=err_correction, err_set=err_set)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    yerr_norm = yerr_vals / y_std
    return y_norm, yerr_norm

# =======================
#   PLOT-ERSTELLUNG
# =======================

def plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs,
                                                  xlabel_lightcurves, save_only, output_dir, shared_y,
                                                  file_name, centroid_data=None, rows=4, cols=2, figsize=None, only_one_label=False, show_reference_label=False,
                                                  for_paper=False, show_histogram=None):
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
    if figsize is None:
        figsize = (5, 12)


    x_values_ccfs = final_sorted_data_dict['time shift (tau)']
    final_sorted_data_dict.pop('time shift (tau)')

    for current_data, group_index in prepare_ccfs_references_data(final_sorted_data_dict, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=shared_y,
                                 gridspec_kw={'width_ratios': [4, 1]})

        # Spaltenweise X-Achse sharen
        # Linke Spalte (Spalte 0)
        for i in range(1, rows):
            axes[i, 0].sharex(axes[0, 0])

        # Rechte Spalte (Spalte 1)
        for i in range(1, rows):
            axes[i, 1].sharex(axes[0, 1])

        fig.subplots_adjust(hspace=0, wspace=0)
        #fig.tight_layout()
        if only_one_label is True:
            # Linke Seite oben (y-Achse): "Normalized Lightcurves"
            fig.text(0.06, 0.5, "Normalized Flux", va='center', ha='left', rotation='vertical', fontsize=12)

            if not for_paper:
                # Rechte Seite unten (y-Achse): ylabel_ccfs
                fig.text(1.0, 0.5, ylabel_ccfs, va='center', ha='right', rotation='vertical', fontsize=12)

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


            configure_ccfs_and_reference_axis(ax,
                                              row,
                                              rows,
                                              col,
                                              ylabel_ccfs,
                                              color,
                                              x_values_ccfs,
                                              line_data,
                                              yerr,
                                              line_name_and_ref_name=line_name,
                                              centroid_data=centroid_data,
                                              only_one_label=only_one_label,
                                              show_reference_label=show_reference_label,
                                              for_paper=for_paper,
                                              show_histogram=show_histogram)

        check_for_empty_rows_ccfs_and_reference(axes, fig, x_label=(xlabel_lightcurves, xlabel_ccfs), for_paper=for_paper)

        finalize_figure_ccfs_and_reference(fig, file_name, save_only=save_only, output_dir=output_dir)



def configure_ccfs_and_reference_axis(ax, row, rows, col, ylabel_ccfs, color, x_values_ccfs, line_data, yerr,
                                      line_name_and_ref_name, centroid_data=None, only_one_label=False, show_reference_label=False, for_paper=False,
                                      show_histogram=None):
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

        err_corr = ERR_CORRECTION.get(line_name, None)
        ref_err_corr = ERR_CORRECTION.get(reference_name, None)
        err_set = ERR_SET.get(line_name, None)
        ref_err_set = ERR_SET.get(reference_name, None)

    else:
        line_name = None
        reference_name = None
        err_corr = None
        ref_err_corr = None
        err_set = None
        ref_err_set = None

    if len(line_data) == 0:
        return

    if yerr is not None:
        x_key = 'timestamps [MJD]'
        y_key = 'fluxes [ergs/s/cm2/A]'
        yerr_key = 'fluxerrs [ergs/s/cm2/A]'

        y_norm, yerr_norm = normalize_lightcurve(line_data["lightcurves"][y_key],
                                                 line_data["lightcurves"][yerr_key],
                                                 err_correction=err_corr,
                                                 err_set=err_set)
        y_ref_norm, yerr_ref_norm = normalize_lightcurve(line_data["lightcurves_ref"][y_key],
                                                          line_data["lightcurves_ref"][yerr_key],
                                                         err_correction=ref_err_corr,
                                                         err_set=ref_err_set)

        ax.text(
            57582, 2.5,  # Position relativ zur Achse (x, y)
            f"{NUMBER_MAPPING[row + 1]})",  # Dein Label
            ha='right', va='top',
            fontsize=9,
            fontweight='bold'  # macht den Text fett
        )
        if line_name != "UVW2":
            if line_name in SYMBOLES_AND_COLORS_FOR_LIGHTCURVES.keys():
                line_color = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]["color"]
                fmt = f"{SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]['symbole']}-"
                markersize = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]["markersize"]
                alpha = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name].get("alpha", 1.0)
            else:
                line_color = color[0]
                fmt = ".-"
                markersize = 3
                alpha = 1.0

            if reference_name in SYMBOLES_AND_COLORS_FOR_LIGHTCURVES.keys():
                ref_line_color = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name]["color"]
                ref_fmt = f"{SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name]['symbole']}-"
                ref_markersize = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name]["markersize"]
                ref_alpha = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name].get("alpha", 1.0)
            else:
                ref_line_color = color[0]
                ref_fmt = ".-"
                ref_markersize = 3
                ref_alpha = 1.0



            ax.errorbar(line_data["lightcurves"][x_key], y_norm, yerr=yerr_norm,
                        label=format_label(line_name, as_latex=False, for_paper=for_paper), color=line_color, fmt=fmt, capsize=2,
                        markersize=markersize, alpha=alpha, linewidth=0.5, elinewidth=0.5)
            if show_reference_label:

                ax.errorbar(line_data["lightcurves_ref"][x_key], y_ref_norm, yerr=yerr_ref_norm, label=format_label(reference_name, as_latex=False, for_paper=for_paper), color=ref_line_color, fmt=ref_fmt, capsize=2, markersize=ref_markersize, alpha=ref_alpha, linewidth=0.5, elinewidth=0.5)
            else:
               ax.errorbar(line_data["lightcurves_ref"][x_key], y_ref_norm, yerr=yerr_ref_norm, color=ref_line_color, fmt=ref_fmt, capsize=2,
                           markersize=ref_markersize, alpha=ref_alpha, linewidth=0.5, elinewidth=0.5)

        else:
            ax.errorbar(line_data["lightcurves"][x_key], y_norm, yerr=yerr_norm,
                        label=format_label(line_name, as_latex=False, for_paper=for_paper), color=SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]["color"], fmt=f"{SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]['symbole']}-", alpha=0.8, capsize=2,  markersize=3, linewidth=0.5, elinewidth=0.5)

        configure_axes_for_lightcurves(ax, row, only_one_label, for_paper=for_paper)
        ax.legend(fontsize=7, loc="upper right", frameon=False, markerfirst=False)
    else:
        ax.plot(x_values_ccfs, line_data["ccfs"], color=color)

        try:
            _, mc_correlation_data_optical_calibrated = import_centroid_and_mc_data("NGC4593_optical_calibrated", reference_name, [line_name])
            merged_mc_correlation_data = mc_correlation_data_optical_calibrated
        except Exception as e:
            print(f"{e}")

            try:
                _, mc_correlation_data_not_optical_calibrated = import_centroid_and_mc_data("NGC4593_not_optical_calibrated",
                                                                                        reference_name, [line_name])
                merged_mc_correlation_data = mc_correlation_data_not_optical_calibrated
            except Exception as e:
                print(f"{e}")
                merged_mc_correlation_data = {}


        if centroid_data:


            tau = math.ceil(abs(centroid_data[reference_name][line_name]["tau_cent"])*10)/10
            err_h = math.ceil(abs(centroid_data[reference_name][line_name]["tau_cent_err_high"])*10)/10
            err_l = math.ceil(abs(centroid_data[reference_name][line_name]["tau_cent_err_low"])*10)/10

            ax.text(
                9, 0.95,
                fr"$\mathbf{{\tau_\mathrm{{cent}} = {tau:.1f}^{{+{err_h:.1f}}}_{{-{err_l:.1f}}}}}$",
                ha='right',
                va='top',
                fontsize=7.5
            )
            try:

                ax.axvline(tau, color="grey", linestyle="--", linewidth=1)
                if show_histogram:
                    ax.hist(merged_mc_correlation_data[line_name]["centroids"], bins=50, density=True, alpha=0.7, color="grey")
            except KeyError:
                print(f"No centroid data found for line {line_name}")





        ccfs_labels = format_label(line_name, as_latex=False).split(" ")[0]
        if "$" in ccfs_labels:
            ccfs_labels = ccfs_labels + "$"
        if not for_paper:
            ax.text(9, 0.90, ccfs_labels, ha='right', va='top', fontsize=7)
        configure_axes_for_ccfs(ax, row, rows, ylabel_ccfs, only_one_label, for_paper=for_paper)


def configure_axes_for_lightcurves(ax, row, only_one_label=False, for_paper=False):
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

    if only_one_label is False:
        ax.set_ylabel("Normalized Flux", fontsize=12)
    ax.yaxis.set_label_coords(0, 0.5)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='major', direction='in', length=4)
    ax.tick_params(axis='y', which='minor', direction='in', length=2)
    ax.set_ylim(-3, 3)
    if not for_paper:
        ax.set_yticklabels([])


    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='x', which='major', direction='inout', length=4)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', direction='in', length=2)



    ax_top = ax.secondary_xaxis('top')
    ax_top.xaxis.set_major_locator(MultipleLocator(5))
    ax_top.tick_params(axis='x', which='major', direction='in')
    ax_top.xaxis.set_minor_locator(MultipleLocator(1))
    ax_top.tick_params(axis='x', which='minor', direction='in', length=2)

    ax_top.set_xticklabels([])

    if row == 0 and not for_paper:
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        plt.setp(ax_top.get_xticklabels(), rotation=45, ha='right')



def configure_axes_for_ccfs(ax, row, nrows, ylabel_ccfs, only_one_label=False, for_paper=False):
    ax.axvline(x=0, color='black', linestyle=':', linewidth=0.5)
    ax.set_xlim(-5, 10)
    ax.set_ylim(0, 1)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='y', which='major', direction='in', length=4)
    ax.tick_params(axis='y', which='minor', direction='in', length=2)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    if not only_one_label:
        ax.set_ylabel(ylabel_ccfs, fontsize=12)

    # --- Tick-Labels steuern: 1 überall, 0 nur unten & vorletzte ---
    is_bottom = (row == nrows - 1)
    is_penultimate = (row == nrows - 2)

    def _yfmt(y, pos):
        if np.isclose(y, 0.0):
            return "0" if (is_bottom or is_penultimate) else ""
        if np.isclose(y, 1.0):
            return "1"                     # <-- 1 überall
        return f"{y:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(_yfmt))
    # ---------------------------------------------------------------

    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='x', which='major', direction='inout', length=4)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', direction='in', length=2)

    ax_top = ax.secondary_xaxis('top')
    ax_top.xaxis.set_major_locator(MultipleLocator(5))
    ax_top.tick_params(axis='x', which='both', direction='in')
    ax_top.xaxis.set_minor_locator(MultipleLocator(1))
    ax_top.tick_params(axis='x', which='minor', direction='in', length=2)
    ax_top.set_xticklabels([])

    if row == 0 and not for_paper:
        ax_top.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))



def check_for_empty_rows_ccfs_and_reference(
    axes, fig, x_label,
    for_paper=False,
    paper_gap_inch=-0.2,   # Reduzierter Abstand zur vorletzten Reihe (in Zoll)
    xlabel_pad=2
):
    def _is_valid_axis(ax):
        return (ax in fig.axes)

    n_rows = axes.shape[0]

    # 1) Leere Reihen entfernen
    for r in range(n_rows):
        if all(not _is_valid_axis(axes[r, c]) for c in range(2)):
            for c in range(2):
                if axes[r, c] in fig.axes:
                    fig.delaxes(axes[r, c])

    # 2) Verbleibende Reihen finden
    remaining = [r for r in range(n_rows) if any(axes[r, c] in fig.axes for c in range(2))]
    if not remaining:
        return
    lowest_row = max(remaining)
    penultimate_row = max([r for r in remaining if r < lowest_row], default=None)

    # 3) Formatter/Lokatoren für alle sichtbaren Achsen
    for r in remaining:
        for c in range(2):
            ax = axes[r, c]
            if ax not in fig.axes:
                continue
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    # 4) Ticklabel-Sichtbarkeit + x-Label unten an letzte Reihe
    for r in remaining:
        for c in range(2):
            ax = axes[r, c]
            if ax not in fig.axes:
                continue
            ax.tick_params(axis='x', which='both', direction='in',
                           labelbottom=(r == lowest_row))
    for c in range(2):
        axl = axes[lowest_row, c]
        if axl in fig.axes:
            lbl = x_label[c] if isinstance(x_label, tuple) else x_label
            axl.set_xlabel(lbl, labelpad=xlabel_pad)

    # 5) Im Paper-Modus: letzte Reihe nach oben schieben (kleinerer Abstand zur vorletzten)
    if for_paper and penultimate_row is not None:
        fig_w, fig_h = fig.get_size_inches()
        dy_rel = paper_gap_inch / fig_h  # Zoll -> Figure-Fraction
        for c in range(2):
            ax_last = axes[lowest_row, c]
            if ax_last not in fig.axes:
                continue
            bbox = ax_last.get_position()
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            w, h = (x1 - x0), (y1 - y0)
            # verschiebe leicht nach oben (verkleinert Abstand nach oben)
            ax_last.set_position([x0, y0 + dy_rel, w, h])


def finalize_figure_ccfs_and_reference(fig, filename, save_only, output_dir):
    """
    Finalizes the layout of a Matplotlib figure: removes empty subplot rows, sets title,
    saves the figure as PDF and PNG, and optionally displays it.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to finalize.
    axes : numpy.ndarray
        2D array of Matplotlib Axes objects.
    title : str
        Title of the figure.
    group_index : int
        Index of the current group (used in filenames).
    save_only : bool
        If True, the figure will only be saved (not shown).
    output_dir : str or pathlib.Path
        Directory where the figure will be saved.
    x_label : str or tuple of str
        Label(s) for the x-axis. Tuple assigns different labels per column.

    Returns:
    -----------
    None
    """

    save_path = output_dir / f"{filename}.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    save_path = output_dir / f"{filename}.png"
    plt.savefig(save_path, bbox_inches='tight')

    print(f"Figure saved to {save_path}")

    if not save_only:
        plt.show()
    plt.close(fig)

# =======================
#   HILFSFUNKTIONEN
# =======================

def is_valid_axis(ax, fig):
    """
    Checks whether a given axis is part of the figure and contains visible content.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to check.
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the axis.

    Returns:
    -----------
    bool
        True if the axis is in the figure and contains lines, containers, or images.
    """
    return ax in fig.axes and (len(ax.lines) > 0 or len(ax.containers) > 0 or len(ax.images) > 0)


def format_month_day(mjd, pos):
    """
    Formatter function for Matplotlib that converts MJD to 'Month Day' format (e.g., 'Aug 01').

    Parameters:
    -----------
    mjd : float
        Modified Julian Date.
    pos : int
        Axis position (passed by Matplotlib, not used here).

    Returns:
    -----------
    str
        Date string in the format '%b %d', e.g., 'Aug 01'.
    """

    date = mjd_to_date(mjd)
    return date.strftime('%b %d')


def mjd_to_date(mjd):
    """
    Converts a Modified Julian Date (MJD) to a Gregorian calendar date.

    Parameters:
    -----------
    mjd : float
        The Modified Julian Date value.

    Returns:
    -----------
    datetime.datetime
        The corresponding calendar date based on the MJD epoch (1858-11-17).
    """

    mjd_start_date = datetime.datetime(1858, 11, 17)  # MJD Startdatum
    return mjd_start_date + datetime.timedelta(days=mjd)

# =======================
#   METHODENAUFRUFE
# =======================

"""
uv_to_halpha_keyorder = ["time shift (tau)",
                         "UVW2",
                         "LyAlpha_not_optical_calibrated",
                         "NV1238_not_optical_calibrated",
                         "SiIV1393_not_optical_calibrated",
                         "CIV1548_not_optical_calibrated",
                         "HeII1640_not_optical_calibrated",
                         "HDelta",
                         "HGamma",
                         "HeII4685",
                         "HBeta",
                         "HeI5875",
                         "HAlpha"]





save_1d_corr_and_lightcurves_general(
    campaign_keys=[],  # Nicht erforderlich bei combine_data=True
    keyorders_dict={"UVW2": uv_to_halpha_keyorder},
    file_name="UV_to_HAlpha_ccfs_and_reference_lightcurves",
    final_key_order=uv_to_halpha_keyorder,
    rows=len(uv_to_halpha_keyorder) - 1,
    cols=2,
    combine_data=True,
    campaign_label="NGC4593_Combined"
)



uvw2_keyorders_optical = {"UVW2":
                              ["time shift (tau)",
                               "HAlpha",
                               "HBeta",
                               "HGamma",
                               "HDelta",
                               "LyAlpha_not_optical_calibrated",
                               "HeI5875",
                               "HeII4685",
                               "OI8446"]}

uvw2_keyorders_not_optical = { "UVW2": ["time shift (tau)",
                                        "SiIV1393_not_optical_calibrated",
                                        "NV1238_not_optical_calibrated",
                                        "CIV1548_not_optical_calibrated",
                                        "HeII1640_not_optical_calibrated"]}


save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=uvw2_keyorders_optical,
    file_name="UVW2_ccfs_and_reference_lightcurves_optical",
    combine_data=True,
    rows=8,
    figsize=(5, 8)
)

save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=uvw2_keyorders_not_optical,
    file_name="UVW2_ccfs_and_reference_lightcurves_not_optical",
    combine_data=True,
    rows=4,
    figsize=(5, 4)
)

OI_keyorder = { "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446"],
                "HAlpha": ["time shift (tau)", "OI8446"],
                "UVW2": ["time shift (tau)", "HAlpha", "OI8446"],}

save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=OI_keyorder,
    file_name="OI_ccfs_and_reference_lightcurves",
    final_key_order=["time shift (tau)", "OI8446", "HAlpha"],
    combine_data=True,
    rows=4,
    figsize=(6, 4),
    show_reference_label=True
)
"""
OI_paper_keyorder_HAlpha = { "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446","HAlpha"],
                      "UVW2": ["time shift (tau)", "HAlpha", "OI8446"],
                      #"HAlpha": ["time shift (tau)", "LyAlpha_not_optical_calibrated"],
                      }

OI_paper_HST_UV_keyorder_HAlpha = { "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446","HAlpha"],
                      "Cont1150_not_optical_calibrated": ["time shift (tau)", "HAlpha", "OI8446"],
                      #"HAlpha": ["time shift (tau)", "LyAlpha_not_optical_calibrated"],
                      }

OI_paper_keyorder_HBeta = { "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446","HBeta"],
                      "UVW2": ["time shift (tau)", "HBeta", "OI8446"],
                      #"HAlpha": ["time shift (tau)", "LyAlpha_not_optical_calibrated"],
                      }

OI_paper_HST_UV_keyorder_HBeta = { "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446", "HBeta"],
                      "Cont1150_not_optical_calibrated": ["time shift (tau)", "HBeta", "OI8446"],
                      #"HAlpha": ["time shift (tau)", "LyAlpha_not_optical_calibrated"],
                      }

save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=OI_paper_keyorder_HAlpha,
    file_name="OI_ccfs_and_reference_lightcurves_paper HAlpha",
    final_key_order=["time shift (tau)", "OI8446", "HAlpha"],
    combine_data=True,
    rows=5,
    figsize=(6, 8),
    show_reference_label=True,
    for_paper=True,
    extra_data_name="OI8446_ref_HAlpha",
    show_histogram=False
)

"""
save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=OI_paper_HST_UV_keyorder_HAlpha,
    file_name="OI_ccfs_and_reference_lightcurves_HST_UV_paper HAlpha",
    final_key_order=["time shift (tau)", "OI8446", "HAlpha"],
    combine_data=True,
    rows=5,
    figsize=(6, 8),
    show_reference_label=True,
    for_paper=True,
    extra_data_name="OI8446_ref_HAlpha"
)

save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=OI_paper_keyorder_HBeta,
    file_name="OI_ccfs_and_reference_lightcurves_paper HBeta",
    final_key_order=["time shift (tau)", "OI8446", "HBeta"],
    combine_data=True,
    rows=5,
    figsize=(6, 8),
    show_reference_label=True,
    for_paper=True,
    extra_data_name="OI8446_ref_HBeta"
)


save_1d_corr_and_lightcurves_general(
    campaign_keys=[],
    keyorders_dict=OI_paper_HST_UV_keyorder_HBeta,
    file_name="OI_ccfs_and_reference_lightcurves_HST_UV_paper HBeta",
    final_key_order=["time shift (tau)", "OI8446", "HBeta"],
    combine_data=True,
    rows=5,
    figsize=(6, 8),
    show_reference_label=True,
    for_paper=True,
    extra_data_name="OI8446_ref_HBeta"
)
"""


"""

bowen_keyorders = {
    "NGC4593_optical_calibrated": {
        "Cont1150_not_optical_calibrated": ["time shift (tau)", "HAlpha", "HBeta", "LyAlpha_not_optical_calibrated", "OI8446"],
        "LyAlpha_not_optical_calibrated": ["time shift (tau)", "HAlpha", "HBeta", "OI8446"],
        "HAlpha": ["time shift (tau)", "OI8446"],
        "HBeta": ["time shift (tau)", "OI8446"]
    }
}



save_1d_corr_and_lightcurves_general(
    campaign_keys=["NGC4593_optical_calibrated"],
    keyorders_dict=bowen_keyorders,
    file_name="bowen_fluorescence_ccfs_and_reference_lightcurves",
    final_key_order=["time shift (tau)", "OI8446", "LyAlpha_not_optical_calibrated", "HBeta", "HAlpha"],
    rows=9,
    figsize=(6, 11),
    show_reference_label=True
)

"""