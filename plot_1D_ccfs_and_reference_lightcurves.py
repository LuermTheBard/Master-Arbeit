import datetime
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from import_data import import_1d_correlation_data, import_1d_lightcurve_data, load_centroid_data_by_reference, \
    import_centroid_and_mc_data
from plot_utils import format_label, calculate_standard_error_for_lightcurves, ensure_output_dir
from settings import SYMBOLES_AND_COLORS_FOR_LIGHTCURVES, NUMBER_MAPPING, ERR_CORRECTION, ERR_SET

# =======================
#   CONSTANTS & SETTINGS
# =======================

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "default_output"
matplotlib.use('Qt5Agg')


# =======================
#   PLOT CONFIGURATION (PlotConfig)
# =======================

@dataclass
class PlotConfig:
    """
    Controls the layout and appearance of CCF/lightcurve figures.

    Fields
    ------
    rows : int
        Number of subplot rows per figure.
    cols : int
        Number of subplot columns (default: 2 = lightcurve + CCF).
    figsize : tuple or None
        Figure size in inches (width, height). None = calculated automatically.
    combine_data : bool
        If True, merge optically and non-optically calibrated datasets.
    show_reference_label : bool
        If True, the reference lightcurve is labelled in the legend.
    format_labels_as_paper : bool
        If True, use LaTeX formatting for axis labels (for publications).
    layout_show_right_ccf_ylabel : bool
        If True, show the y-axis label on the right side of the CCF panels.
    layout_show_top_secondary_labels : bool
        If True, show date labels on the secondary top x-axis.
    lightcurve_hide_yticklabels : bool
        If True, hide y-axis tick labels on lightcurve panels.
    ccf_show_inline_label_text : bool
        If True, show the line/reference name as inline text inside the CCF panel.
    adjust_last_row_gap_inch : float
        Vertical offset (in inches) of the last row. A negative value reduces
        the gap to the second-to-last row (e.g. -0.2 for paper layout).
    include_extra_data : bool
        If True, an additional dataset (extra_data_name) is appended as the last row to the plot.
    extra_data_name : str or None
        Key of the extra dataset, formatted as "<line>_ref_<reference>".
    show_histogram : bool or None
        Controls histogram display inside the CCF panel (True / False / None).
    show_subfigure_labels : bool
        If True, sub-figure labels (a), b), ...) are shown.
    row_spacing : float or None
        Vertical spacing between rows (hspace). None = no spacing.
    line_style : str
        Matplotlib line style, e.g. "-" (solid) or ":" (dotted).
    grid : tuple or None
        Grid settings as (show_minor, alpha, linewidth, linestyle),
        e.g. (True, 0.12, 0.3, ':'). None = no grid.
    """
    rows: int = 8
    cols: int = 2
    figsize: Optional[Tuple] = None
    combine_data: bool = True
    show_reference_label: bool = False
    format_labels_as_paper: bool = False
    layout_show_right_ccf_ylabel: bool = True
    layout_show_top_secondary_labels: bool = True
    lightcurve_hide_yticklabels: bool = True
    ccf_show_inline_label_text: bool = True
    adjust_last_row_gap_inch: float = 0.0
    include_extra_data: bool = False
    extra_data_name: Optional[str] = None
    show_histogram: Optional[bool] = None
    show_subfigure_labels: bool = True
    row_spacing: Optional[float] = None
    line_style: str = "-"
    grid: Optional[Tuple] = None

# -------------------------------------------------------------------
# Predefined configurations for common use cases
# -------------------------------------------------------------------

# Exploration: all layout helpers enabled, no paper mode
EXPLORE_CONFIG = PlotConfig()

# Publication layout: clean appearance for papers / thesis
PAPER_CONFIG = PlotConfig(
    show_reference_label=True,
    format_labels_as_paper=True,
    layout_show_right_ccf_ylabel=False,
    layout_show_top_secondary_labels=False,
    lightcurve_hide_yticklabels=False,
    ccf_show_inline_label_text=False,
    adjust_last_row_gap_inch=-0.2,
    show_histogram=False,
    show_subfigure_labels=False,
    line_style="-",
    grid=(True, 0.12, 0.3, ':'),  # (show_minor, alpha, linewidth, linestyle)
)

# =======================
#   DATA IMPORT & PREPARATION
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
        campaign_label=None,
        config: PlotConfig = None,
        save_only: bool = True,
):

    if config is None:
        config = PlotConfig()

    ensure_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_dim_correlation_data = import_1d_correlation_data()
    centroid_data = load_centroid_data_by_reference()
    lightcurves_data = import_1d_lightcurve_data()

    if config.combine_data:
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
            centroid_data=centroid_data,
            only_one_label=True,
            config=config,
            save_only=save_only,
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
                centroid_data=centroid_data,
                only_one_label=True,
                config=config,
                save_only=save_only,
            )




# =======================
#   DATA PROCESSING / SORTING
# =======================


def plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccf_data_dict,
                                           campaign,
                                           output_dir,
                                           key_orders,
                                           save_only=False,
                                           file_name=None,
                                           final_key_order=None,
                                           only_one_label=False,
                                           centroid_data=None,
                                           config: PlotConfig = None,
                                           ):
    """
    Organizes and plots CCFs and their corresponding lightcurves
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
    only_one_label : bool, optional
        If True, only one y-axis label per figure is shown. Default is False.
    config : PlotConfig, optional
        Layout and appearance settings. Defaults to PlotConfig() if None.

    Returns:
    -----------
    None
    """

    if config is None:
        config = PlotConfig()

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

    if config.include_extra_data and config.extra_data_name:

        line, reference = config.extra_data_name.split("_ref_")

        if "Cont" in reference:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][reference]
        else:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][reference]

        if "Cont" in line:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][line]
        else:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][line]

        extra_data = {config.extra_data_name:{"ccfs":lightcurves_ccf_data_dict["ccfs"][reference][line],"lightcurves":lightcurve_data,"lightcurves_ref":lightcurve_reference_data}}
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
                                                  only_one_label=only_one_label,
                                                  config=config,
                                                  )


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

    # Insert each item twice in a row (once for lightcurve, once for CCF)
    data_items = []
    for key, value in data.items():
        data_items.append((key, value))
        data_items.append((key, value))  # repeated immediately after

    total_plots = len(data_items)
    plots_per_group = rows * cols
    num_groups = (total_plots + plots_per_group - 1) // plots_per_group

    for group_index in range(num_groups):
        start_index = group_index * plots_per_group
        end_index = min(start_index + plots_per_group, total_plots)
        current_data = data_items[start_index:end_index]

        # Pad with placeholders if the group is incomplete
        while len(current_data) < plots_per_group:
            current_data.append((
                f'Empty {len(current_data) + 1}',
                None
            ))

        yield current_data, group_index

def print_lightcurves_with_final_errors(line_name, x, y, yerr_vals, err_correction=None, err_set=None):

    # Calculate errors
    yerr_vals = calculate_standard_error_for_lightcurves(
        y, yerr_vals, err_correction=err_correction, err_set=err_set
    )

    # Output filename (e.g. OI8446.txt)
    filename = DEFAULT_OUTPUT_DIR / f"{line_name}_final_errors.txt"

    # Build header
    header = (
        f"# {line_name} timestamps [MJD], fluxes [ergs/s/cm2/A], "
        f"fluxerrs [ergs/s/cm2/A]\n"
    )

    # Write file
    with open(filename, "w") as f:
        f.write(header)
        for xi, yi, ei in zip(x, y, yerr_vals):
            f.write(f"{xi:.15e} {yi:.15e} {ei:.15e}\n")

    print(f"Saved file: {filename}")
    return filename, x,y,yerr_vals


def normalize_lightcurve(y, yerr_vals, err_correction=None, err_set=None):
    yerr_vals = calculate_standard_error_for_lightcurves(y, yerr_vals, err_correction=err_correction, err_set=err_set)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    yerr_norm = yerr_vals / y_std
    return y_norm, yerr_norm

# =======================
#   PLOT CREATION
# =======================

def plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict,
                                                  xlabel_ccfs,
                                                  ylabel_ccfs,
                                                  xlabel_lightcurves,
                                                  save_only,
                                                  output_dir,
                                                  shared_y,
                                                  file_name,
                                                  centroid_data=None,
                                                  only_one_label=False,
                                                  config: PlotConfig = None,
                                                  panel_height=1.2,          # inches per row
                                                  right_panel_width=1.2,     # inches for right column (CCF)
                                                  padding=(1.0, 1.6),
                                                  ):
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
    save_only : bool
        If True, plots are saved to disk only; if False, they are also displayed.
    output_dir : str or pathlib.Path
        Directory to save the output files.
    shared_y : bool
        Whether subplots should share the same y-axis limits.
    file_name : str
        Base name for the saved files (PDF and PNG).
    centroid_data : dict or None
        Centroid and MC correlation data for tau annotations on CCF panels.
    only_one_label : bool, optional
        If True, only one label per y-axis (left/right) is shown to avoid clutter.
    config : PlotConfig, optional
        Layout and appearance settings. Defaults to PlotConfig() if None.
    panel_height : float, optional
        Height per subplot row in inches. Default is 1.2.
    right_panel_width : float, optional
        Width of the right (CCF) column in inches. Default is 1.2.
    padding : tuple, optional
        Extra (width, height) padding in inches added to the figure size.

    Returns:
    -----------
    None
    """

    if config is None:
        config = PlotConfig()

    figsize = config.figsize
    if figsize is None:
        # width_ratios = [4,1] -> total width per row = (4+1) * right_panel_width
        pad_w, pad_h = padding
        fig_w = (4 + 1) * right_panel_width + pad_w
        fig_h = config.rows * panel_height + pad_h
        figsize = (fig_w, fig_h)

    x_values_ccfs = final_sorted_data_dict['time shift (tau)']
    final_sorted_data_dict.pop('time shift (tau)')

    for current_data, group_index in prepare_ccfs_references_data(final_sorted_data_dict, config.rows, config.cols):
        fig, axes = plt.subplots(config.rows, config.cols, figsize=figsize, sharex=False, sharey=shared_y,
                                 gridspec_kw={'width_ratios': [4, 1]})

        # Share x-axis within each column
        # Left column (column 0)
        for i in range(1, config.rows):
            axes[i, 0].sharex(axes[0, 0])

        # Right column (column 1)
        for i in range(1, config.rows):
            axes[i, 1].sharex(axes[0, 1])

            fig.subplots_adjust(hspace=(0 if config.row_spacing is None else float(config.row_spacing)), wspace=0)

        if only_one_label is True:
            # Left side (y-axis): shared "Normalized Flux" label
            fig.text(0.06, 0.5, "Normalized Flux", va='center', ha='left', rotation='vertical', fontsize=12)

            if config.layout_show_right_ccf_ylabel:
                fig.text(1.0, 0.5, ylabel_ccfs, va='center', ha='right', rotation='vertical', fontsize=12)

        for i, (line_name, line_data) in enumerate(current_data):

            row, col = divmod(i, config.cols)
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
                                              config.rows,
                                              col,
                                              ylabel_ccfs,
                                              color,
                                              x_values_ccfs,
                                              line_data,
                                              yerr,
                                              line_name_and_ref_name=line_name,
                                              centroid_data=centroid_data,
                                              only_one_label=only_one_label,
                                              config=config)

        check_for_empty_rows_ccfs_and_reference(axes, fig, x_label=(xlabel_lightcurves, xlabel_ccfs),
                                                adjust_last_row_gap_inch=config.adjust_last_row_gap_inch)

        finalize_figure_ccfs_and_reference(fig, file_name, save_only=save_only, output_dir=output_dir)



def configure_ccfs_and_reference_axis(ax,
                                      row,
                                      rows,
                                      col,
                                      ylabel_ccfs,
                                      color,
                                      x_values_ccfs,
                                      line_data,
                                      yerr,
                                      line_name_and_ref_name,
                                      centroid_data=None,
                                      only_one_label=False,
                                      config: PlotConfig = None):
    """
    Configures a single subplot axis to display either a normalized lightcurve pair
    or a CCF, depending on the data provided.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The subplot axis to configure.
    row : int
        Row index within the subplot grid.
    rows : int
        Total number of rows in the subplot grid.
    col : int
        Column index within the subplot grid (0 = lightcurve, 1 = CCF).
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
    centroid_data : dict or None
        Centroid lag data for tau annotations.
    only_one_label : bool, optional
        If True, y-axis labels are minimized to reduce clutter.
    config : PlotConfig, optional
        Layout and appearance settings. Defaults to PlotConfig() if None.

    Returns:
    -----------
    None
    """

    if config is None:
        config = PlotConfig()

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

        if config.show_subfigure_labels:
            ax.text(
                57582, 2.5,  # Position
                f"{NUMBER_MAPPING[row + 1]})",
                ha='right', va='top',
                fontsize=9,
                fontweight='bold'


            )

        if line_name in SYMBOLES_AND_COLORS_FOR_LIGHTCURVES.keys():
            line_color = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]["color"]
            fmt = f"{SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]['symbole']}{config.line_style}"
            markersize = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]["markersize"]
            alpha = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name].get("alpha", 1.0)
        else:
            line_color = color[0]
            fmt = f".{config.line_style}"
            markersize = 3
            alpha = 1.0

        if reference_name in SYMBOLES_AND_COLORS_FOR_LIGHTCURVES.keys():
            ref_line_color = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name]["color"]
            ref_fmt = f"{SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name]['symbole']}{config.line_style}"
            ref_markersize = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name]["markersize"]
            ref_alpha = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[reference_name].get("alpha", 1.0)
        else:
            ref_line_color = color[0]
            ref_fmt = f".{config.line_style}"
            ref_markersize = 3
            ref_alpha = 1.0

        if line_name != "UVW2":
            ax.errorbar(line_data["lightcurves"][x_key],
                        y_norm,
                        yerr=yerr_norm,
                        label = format_label(line_name,
                                             as_latex=False,
                                             for_paper=config.format_labels_as_paper),
                        color = line_color,
                        fmt = fmt,
                        capsize = 2,
                        markersize=markersize,
                        alpha=alpha,
                        linewidth=0.5,
                        elinewidth=0.5)
            if config.show_reference_label:
                ax.errorbar(line_data["lightcurves_ref"][x_key], y_ref_norm, yerr=yerr_ref_norm,
                            label=format_label(reference_name, as_latex=False, for_paper=config.format_labels_as_paper),
                            color=ref_line_color, fmt=ref_fmt, capsize=2,
                            markersize=ref_markersize, alpha=ref_alpha, linewidth=0.5, elinewidth=0.5)
            else:
               ax.errorbar(line_data["lightcurves_ref"][x_key], y_ref_norm, yerr=yerr_ref_norm,
                           color=ref_line_color, fmt=ref_fmt, capsize=2,
                           markersize=ref_markersize, alpha=ref_alpha, linewidth=0.5, elinewidth=0.5)

        else:
            ax.errorbar(line_data["lightcurves"][x_key],
                        y_norm,
                        yerr=yerr_norm,
                        label = format_label(line_name,
                                             as_latex=False,
                                             for_paper=config.format_labels_as_paper),
                        color = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]["color"],
                        fmt = f"{SYMBOLES_AND_COLORS_FOR_LIGHTCURVES[line_name]['symbole']}{config.line_style}",
                        alpha = 0.8,
                        capsize = 2,
                        markersize = 3,
                        linewidth = 0.5,
                        elinewidth = 0.5)
            if reference_name != "UVW2":
                if config.show_reference_label:
                    ax.errorbar(line_data["lightcurves_ref"][x_key], y_ref_norm, yerr=yerr_ref_norm,
                                label=format_label(reference_name, as_latex=False, for_paper=config.format_labels_as_paper),
                                color=ref_line_color, fmt=ref_fmt, capsize=2, markersize=ref_markersize,
                                alpha=ref_alpha, linewidth=0.5, elinewidth=0.5)
                else:
                    ax.errorbar(line_data["lightcurves_ref"][x_key], y_ref_norm, yerr=yerr_ref_norm,
                                color=ref_line_color, fmt=ref_fmt, capsize=2,
                                markersize=ref_markersize, alpha=ref_alpha, linewidth=0.5, elinewidth=0.5)

        configure_axes_for_lightcurves(ax,
                                       row,
                                       only_one_label,
                                       lightcurve_hide_yticklabels=config.lightcurve_hide_yticklabels,
                                       layout_show_top_secondary_labels=config.layout_show_top_secondary_labels)
        ax.legend(fontsize=7, loc="upper right", frameon=False, markerfirst=False)
        _apply_grid(ax, config.grid)
    else:
        ax.plot(x_values_ccfs, line_data["ccfs"], color=color)

        try:
            if "not_optical" not in line_name:
                _, mc_correlation_data = import_centroid_and_mc_data("NGC4593_optical_calibrated", reference_name, [line_name])
            else:
                _, mc_correlation_data = import_centroid_and_mc_data("NGC4593_not_optical_calibrated", reference_name, [line_name])
            merged_mc_correlation_data = mc_correlation_data

        except Exception as e:
            print(f"{e}")



        if centroid_data:

            try:
                tau = abs(centroid_data[reference_name][line_name]["tau_cent"])
                err_h = math.ceil(abs(centroid_data[reference_name][line_name]["tau_cent_err_high"])*10)/10
                err_l = math.ceil(abs(centroid_data[reference_name][line_name]["tau_cent_err_low"])*10)/10

                ax.text(
                    9, 0.95,
                    fr"${{\tau_\mathrm{{cent}} = {tau:.1f}^{{+{err_h:.1f}}}_{{-{err_l:.1f}}}}}$",
                    ha='right',
                    va='top',
                    fontsize=7.5
                )

                try:

                    ax.axvline(tau, color="grey", linestyle="--", linewidth=1)
                    if config.show_histogram:
                        ax.hist(merged_mc_correlation_data[line_name]["centroids"], bins=50, density=True, alpha=0.7,
                                color="grey")
                except KeyError:
                    print(f"No centroid data found for line {line_name}")
                except Exception as e:
                    print(f"{e}")
            except KeyError:
                print(f"No centroid data found for line {line_name}")



        ccfs_labels = format_label(line_name, as_latex=False).split(" ")[0]
        if "$" in ccfs_labels:
            ccfs_labels = ccfs_labels + "$"

        if config.ccf_show_inline_label_text:
            ax.text(9, 0.90, ccfs_labels, ha='right', va='top', fontsize=7)
        configure_axes_for_ccfs(ax, row, rows, ylabel_ccfs, only_one_label,
                                layout_show_top_secondary_labels=config.layout_show_top_secondary_labels,
                                row_spacing=config.row_spacing)
        _apply_grid(ax, config.grid)


def configure_axes_for_lightcurves(ax, row, only_one_label=False, lightcurve_hide_yticklabels=True, layout_show_top_secondary_labels=True):

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
    if lightcurve_hide_yticklabels:
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

    if row == 0 and layout_show_top_secondary_labels:
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        plt.setp(ax_top.get_xticklabels(), rotation=45, ha='right')



def configure_axes_for_ccfs(ax, row, nrows, ylabel_ccfs, only_one_label=False, layout_show_top_secondary_labels=True, row_spacing=None):
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

    # --- Tick label control: show "1" everywhere, show "0" only on bottom and second-to-last row ---
    is_bottom = (row == nrows - 1)
    if row_spacing == 0:
        is_bottom = (row == nrows)
    is_penultimate = (row == nrows - 2)
    if row_spacing == 0:
        is_penultimate = (row == nrows-1)

    def _yfmt(y, pos):
        if np.isclose(y, 0.0):
            return "0" if (is_bottom or is_penultimate) else ""
        if np.isclose(y, 1.0):
            return "1"                     # show "1" on every row
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

    if row == 0 and layout_show_top_secondary_labels:
        ax_top.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))



def check_for_empty_rows_ccfs_and_reference(
    axes, fig, x_label,
    adjust_last_row_gap_inch=0.0,   # e.g. -0.2 for paper layout
    xlabel_pad=2
):
    def _is_valid_axis(ax):
        return (ax in fig.axes)

    n_rows = axes.shape[0]

    # 1) Remove empty rows
    for r in range(n_rows):
        if all(not _is_valid_axis(axes[r, c]) for c in range(2)):
            for c in range(2):
                if axes[r, c] in fig.axes:
                    fig.delaxes(axes[r, c])

    # 2) Find remaining (non-empty) rows
    remaining = [r for r in range(n_rows) if any(axes[r, c] in fig.axes for c in range(2))]
    if not remaining:
        return
    lowest_row = max(remaining)
    penultimate_row = max([r for r in remaining if r < lowest_row], default=None)

    # 3) Set formatters/locators for all visible axes
    for r in remaining:
        for c in range(2):
            ax = axes[r, c]
            if ax not in fig.axes:
                continue
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    # 4) Tick label visibility + attach x-label to the bottom row
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

    # 5) Optionally shift the last row upward (reduces gap to second-to-last row)
    if (adjust_last_row_gap_inch != 0.0) and (penultimate_row is not None):
        fig_w, fig_h = fig.get_size_inches()
        dy_rel = adjust_last_row_gap_inch / fig_h  # inches -> figure fraction
        for c in range(2):
            ax_last = axes[lowest_row, c]
            if ax_last not in fig.axes:
                continue
            bbox = ax_last.get_position()
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            w, h = (x1 - x0), (y1 - y0)
            # shift slightly upward (reduces gap above)
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
#   HELPER FUNCTIONS
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

    mjd_start_date = datetime.datetime(1858, 11, 17)  # MJD epoch
    return mjd_start_date + datetime.timedelta(days=mjd)



def _apply_grid(ax, grid):
    """
    Apply a grid to the given axis.

    grid : None
        No grid is drawn.
    grid : tuple (show_minor, alpha, linewidth, linestyle)
        Draws a major grid with the given style; if show_minor is True,
        also draws a minor grid with reduced alpha and linewidth.
    """
    if grid is None:
        return
    grid_minor, grid_alpha, grid_linewidth, grid_linestyle = grid
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='both',
            alpha=grid_alpha, linewidth=grid_linewidth, linestyle=grid_linestyle)
    if grid_minor:
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='both',
                alpha=max(grid_alpha*0.7, 0.05),
                linewidth=max(grid_linewidth*0.8, 0.1),
                linestyle=grid_linestyle)

# =======================
#   FIGURE FUNCTIONS
# =======================
# Each figure is wrapped in its own function so it can be called
# individually from the command line (see main() below).
#
# Available base configurations:
#   EXPLORE_CONFIG  – all layout helpers enabled, no paper mode (exploration)
#   PAPER_CONFIG    – clean layout for publications / thesis
#
# Individual overrides via dataclasses.replace(), e.g.:
#   config = replace(PAPER_CONFIG, rows=6, figsize=(6, 8))
# =======================

def figure_1_uvw2_balmer_ly_o(save_only: bool = True):
    """
    FIGURE 1: UVW2 CCFs – Balmer lines, Lyα, OI
    Reference: UVW2 (optically calibrated) + Cont1150 as extra dataset.
    Output: UVW2_ccfs_Balmer_Ly_O
    """
    keyorders = {
        "UVW2": ["time shift (tau)", "UVW2", "HAlpha", "HBeta", "HGamma",
                 "HDelta", "LyAlpha_not_optical_calibrated", "OI8446"]
    }
    save_1d_corr_and_lightcurves_general(
        campaign_keys=[],
        keyorders_dict=keyorders,
        file_name="UVW2_ccfs_Balmer_Ly_O",
        config=replace(PAPER_CONFIG,
                       rows=8,
                       include_extra_data=True,
                       extra_data_name="UVW2_ref_Cont1150_not_optical_calibrated",
                       show_histogram=True),
        save_only=save_only,
    )


def figure_2_uvw2_helium_uv(save_only: bool = True):
    """
    FIGURE 2: UVW2 CCFs – helium and UV lines
    Reference: UVW2 (non-optically calibrated) + Cont1150 as extra dataset.
    Output: UVW2_ccfs_Helium_UV
    """
    keyorders = {
        "UVW2": ["time shift (tau)", "UVW2", "HeI5875", "HeII1640_not_optical_calibrated",
                 "HeII4685", "NV1238_not_optical_calibrated",
                 "SiIV1393_not_optical_calibrated", "CIV1548_not_optical_calibrated"]
    }
    save_1d_corr_and_lightcurves_general(
        campaign_keys=[],
        keyorders_dict=keyorders,
        file_name="UVW2_ccfs_Helium_UV",
        config=replace(PAPER_CONFIG,
                       rows=8,
                       include_extra_data=True,
                       extra_data_name="UVW2_ref_Cont1150_not_optical_calibrated",
                       show_histogram=True),
        save_only=save_only,
    )


def figure_3_oi_second_paper_halpha(save_only: bool = True):
    """
    FIGURE 3: OI8446 CCFs – second paper (Hα reference)
    Combined dataset; references: LyAlpha, HAlpha, UVW2.
    Output: OI_ccfs_and_reference_lightcurves_second_paper
    """
    keyorders = {
        "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446"],
        "HAlpha":                          ["time shift (tau)", "OI8446"],
        "UVW2":                            ["time shift (tau)", "OI8446"],
    }
    save_1d_corr_and_lightcurves_general(
        campaign_keys=[],
        keyorders_dict=keyorders,
        file_name="OI_ccfs_and_reference_lightcurves_second_paper",
        final_key_order=["time shift (tau)", "OI8446", "HAlpha"],
        config=replace(PAPER_CONFIG,
                       rows=3,
                       figsize=(6, 5),
                       adjust_last_row_gap_inch=0.0,
                       include_extra_data=True,
                       row_spacing=0.2,
                       line_style=":",
                       grid=(True, 0.5, 0.3, ':')),
        save_only=save_only,
    )


def figure_4_oi_paper_halpha(save_only: bool = True):
    """
    FIGURE 4: OI8446 CCFs – paper (Hα reference)
    Combined dataset; references: LyAlpha, UVW2.
    Output: OI_ccfs_and_reference_lightcurves_paper_HAlpha
    """
    keyorders = {
        "LyAlpha_not_optical_calibrated": ["time shift (tau)", "OI8446", "HAlpha"],
        "UVW2":                            ["time shift (tau)", "HAlpha", "OI8446",
                                            "LyAlpha_not_optical_calibrated"],
    }
    save_1d_corr_and_lightcurves_general(
        campaign_keys=[],
        keyorders_dict=keyorders,
        file_name="OI_ccfs_and_reference_lightcurves_paper_HAlpha",
        final_key_order=["time shift (tau)", "OI8446", "HAlpha"],
        config=replace(PAPER_CONFIG,
                       rows=6,
                       include_extra_data=True,
                       extra_data_name="OI8446_ref_HAlpha"),
        save_only=save_only,
    )


def figure_5_oi_hst_uv_halpha(save_only: bool = True):
    """
    FIGURE 5: OI8446 CCFs – HST/UV comparison (Hα reference)
    Combined dataset; references: LyAlpha, Cont1150.
    Output: OI_ccfs_and_reference_lightcurves_HST_UV_paper_HAlpha
    """
    keyorders = {
        "LyAlpha_not_optical_calibrated":    ["time shift (tau)", "OI8446", "HAlpha"],
        "Cont1150_not_optical_calibrated":   ["time shift (tau)", "HAlpha", "OI8446"],
    }
    save_1d_corr_and_lightcurves_general(
        campaign_keys=[],
        keyorders_dict=keyorders,
        file_name="OI_ccfs_and_reference_lightcurves_HST_UV_paper_HAlpha",
        final_key_order=["time shift (tau)", "OI8446", "HAlpha"],
        config=replace(PAPER_CONFIG,
                       rows=5,
                       figsize=(6, 8),
                       include_extra_data=True,
                       extra_data_name="OI8446_ref_HAlpha"),
        save_only=save_only,
    )


# =======================
#   MAIN
# =======================

# Registry: maps CLI name → (function, description)
FIGURES = {
    "fig1": (figure_1_uvw2_balmer_ly_o,      "UVW2 CCFs – Balmer lines, Lyα, OI"),
    "fig2": (figure_2_uvw2_helium_uv,         "UVW2 CCFs – helium and UV lines"),
    "fig3": (figure_4_oi_paper_halpha,        "OI8446 CCFs – paper (Hα ref)"),
    "fig4": (figure_3_oi_second_paper_halpha, "OI8446 CCFs – second paper (Hα ref)"),
    "fig5": (figure_5_oi_hst_uv_halpha,       "OI8446 CCFs – HST/UV comparison (Hα ref)"),
}


def main():
    """
    Command-line entry point for generating individual figures.

    Figures are saved to disk by default and not displayed.
    Pass --show to also open them in a window after saving.

    Usage
    -----
    # Generate a single figure (save only):
        python plot_1D_ccfs_and_reference_lightcurves.py fig1

    # Generate and display a figure:
        python plot_1D_ccfs_and_reference_lightcurves.py fig1 --show

    # Generate multiple figures:
        python plot_1D_ccfs_and_reference_lightcurves.py fig1 fig3 fig4

    # Generate all figures (fig1–fig5):
        python plot_1D_ccfs_and_reference_lightcurves.py all

    # List all available figure names:
        python plot_1D_ccfs_and_reference_lightcurves.py --list
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CCF and reference lightcurve figures for NGC4593.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:<6}  {desc}" for name, (_, desc) in FIGURES.items()
        ),
    )
    parser.add_argument(
        "figures",
        nargs="*",
        metavar="FIGURE",
        help=(
            "Figure name(s) to generate (e.g. fig1 fig3). "
            "Use 'all' to generate all figures, "
            "or '--list' to see available names."
        ),
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display figures in a window after saving (default: save only).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available figure names and exit.",
    )

    args = parser.parse_args()

    if args.list or not args.figures:
        print("\nAvailable figures:\n")
        for name, (_, desc) in FIGURES.items():
            print(f"  {name:<6}  {desc}")
        print("\nShortcut:  all → fig1 fig2 fig3 fig4 fig5\n")
        return

    # Resolve 'all' shortcut
    requested = []
    for token in args.figures:
        if token == "all":
            requested += list(FIGURES.keys())
        else:
            requested.append(token)

    # Deduplicate while preserving order
    seen = set()
    requested = [x for x in requested if not (x in seen or seen.add(x))]

    # Validate
    unknown = [r for r in requested if r not in FIGURES]
    if unknown:
        parser.error(
            f"Unknown figure(s): {', '.join(unknown)}. "
            f"Run with --list to see available names."
        )

    save_only = not args.show

    # Run
    for name in requested:
        fn, desc = FIGURES[name]
        print(f"\n{'='*60}")
        print(f"  Generating: [{name}] {desc}")
        print(f"{'='*60}")
        fn(save_only=save_only)

    print(f"\nDone. Generated {len(requested)} figure(s).")


if __name__ == "__main__":
    main()