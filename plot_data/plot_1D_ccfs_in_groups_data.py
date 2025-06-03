import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

from plot_utils import format_label
from plot_data.general_plot import prepare_data, finalize_figure, format_yaxis


def plot_all_1d_ccfs_in_groups_for_cont(data_dict, campaign, cont_name, output_dir, key_order=None,
                                        save_only=False, file_name=None, only_key_order=False):
    """
    Sorts and plots 1D CCFs for a specific continuum reference in grouped subplots.

    This function selects CCFs from `data_dict` based on a continuum name, sorts them according to a custom order,
    and delegates the grouped plotting to `plot_ccfs_in_groups()`.

    Parameters:
    -----------
    data_dict : dict
        Dictionary of CCF data structured as {continuum_name: {line_name: array-like, ...}, ...}.
    campaign : str
        Campaign identifier (used in output folder structure).
    cont_name : str
        Name of the continuum lightcurve serving as the reference.
    output_dir : str or pathlib.Path
        Base directory to store the output plots.
    key_order : list of str, optional
        Specific order of emission lines to plot.
    save_only : bool, optional
        If True, saves the plot(s) without displaying them. Default is False.
    file_name : str, optional
        Custom base name for saved files. Default is None.
    only_key_order : bool, optional
        If True, only includes emission lines listed in `key_order`. Default is False.

    Returns:
    -----------
    None
    """

    xlabel = "Time Lag $\\tau$ [d]"
    ylabel = "Correlation Coefficient"

    def sort_keys(key):
        for idx, prefix in enumerate(key_order):
            if key == prefix:
                return idx
        return len(key_order)


    save_folder = output_dir / campaign / "plot_1d_ccfs" / cont_name
    save_folder.mkdir(parents=True, exist_ok=True)

    sorted_data_dict = dict()

    try:
        sorted_data_dict = dict(sorted(data_dict[cont_name].items(), key=lambda item: sort_keys(item[0])))

        if only_key_order is True:
            keys_to_keep = key_order + ["time shift (tau)"]
            sorted_data_dict = {k: v for k, v in sorted_data_dict.items() if k in keys_to_keep}

    except KeyError:
        print(f"[Warning] Continuum name '{cont_name}' not found in campaign '{campaign}'. Skipping.")


    plot_ccfs_in_groups(sorted_data_dict, cont_name, xlabel, ylabel,
                        title=f"CCFs between Emission Lines and {format_label(cont_name)} for {campaign.split('_')[0]}",
                        save_only=save_only, output_dir=save_folder, shared_y=True, file_name=file_name)


def plot_ccfs_in_groups(data, compare_cont, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                        title=None, save_only=False, output_dir=None, color_dict=None, rows=4, cols=2, file_name=None):
    """
    Plots multiple 1D CCFs in subplot groups for a single reference continuum.

    Each emission line is plotted in a separate subplot, organized into grids of (rows x cols).
    Optionally uses shared y-axes and color coding.

    Parameters:
    -----------
    data : dict
        Dictionary with {line_name: y_values} pairs. Must include 'time shift (tau)' key for x-values.
    compare_cont : str
        Name of the reference continuum, used for labeling and file naming.
    xlabel : str, optional
        X-axis label for all plots. Default is 'X-axis'.
    ylabel : str, optional
        Y-axis label. Default is 'Y-axis'.
    shared_y : bool, optional
        If True, all subplots share the same y-axis limits.
    title : str, optional
        Title for the overall figure.
    save_only : bool, optional
        If True, saves the figure without displaying it.
    output_dir : str or Path, optional
        Output directory where plots will be saved.
    color_dict : dict, optional
        Optional dictionary that maps line names to specific colors.
    rows : int, optional
        Number of subplot rows. Default is 4.
    cols : int, optional
        Number of subplot columns. Default is 2.
    file_name : str, optional
        Optional custom filename (used as prefix).

    Returns:
    -----------
    None
    """

    x_values_ccfs = data['time shift (tau)']
    data.pop('time shift (tau)')

    for current_data, group_index in prepare_data(data, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=True, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            if line_data is not None:
                y_values = line_data
                color = color_dict.get(line_name, 'black') if color_dict else 'black'
            else:
                y_values = np.array([])
                color = "black"

            configure_ccfs_axis(ax, row, col, ylabel, color, x_values_ccfs, y_values, None, line_name)

        finalize_figure(fig, axes, x_label=xlabel, title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir, compare_cont=compare_cont, file_name=file_name)


def configure_ccfs_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """
    Configures a subplot axis for plotting a CCF.

    Includes line plotting, optional error bars, vertical zero-line, legends,
    and dual y-axes for the right column. Also sets tick formatting and limits.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis object to configure.
    row : int
        Row index in the subplot grid.
    col : int
        Column index in the subplot grid.
    ylabel : str
        Label for the y-axis.
    color : str
        Line color for the plot.
    x_values : np.ndarray
        X-axis values (e.g., time lags).
    y_values : np.ndarray
        Y-axis values (e.g., correlation coefficients).
    yerr_values : np.ndarray or None
        Optional error bars.
    line_name : str
        Label for the line (used in legend).

    Returns:
    -----------
    None
    """


    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr_values,
                fmt='.:', capsize=3, markersize=4, label=f'{format_label(line_name, as_latex=False)}', color=color
            )
        else:
            ax.plot(
                x_values, y_values, label=f'{format_label(line_name, as_latex=False)}', color=color
            )
        ax.axvline(x=0, color='black', linestyle=':', linewidth=0.5)
        ax.legend(fontsize=8, loc='upper right')

    if col == 0:
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax_right = ax.secondary_yaxis('right')
        ax_right.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_right.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    if row < 3:
        ax.set_xticklabels([])

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis))



    ax.set_xlim(-10, 14.999)
    ax.set_ylim(-0.1, 1)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.tick_params(axis='x')
