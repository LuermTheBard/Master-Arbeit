import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, AutoMinorLocator

from handle_data.handle_data_file import format_label
from plot_data.general_plot import prepare_data, finalize_figure, format_yaxis


def plot_all_1d_ccfs_in_groups_for_cont(galaxie_campaigns_correlation_data_dict, cont_name, output_dir, key_order=None,
                                        save_only=False):
    xlabel = "Time Lag $\\tau$ [d]"
    ylabel = "Correlation Coefficient"

    x_key = "time shift (tau)"
    y_key = cont_name

    def sort_keys(key):
        for idx, prefix in enumerate(key_order):
            if key.startswith(prefix):
                return idx
        return len(key_order)

    for campaign, data_dict in galaxie_campaigns_correlation_data_dict.items():

        save_folder = output_dir / campaign / "plot_1d_ccfs" / cont_name
        save_folder.mkdir(parents=True, exist_ok=True)

        try:
            sorted_data_dict = dict(sorted(data_dict[cont_name].items(), key=lambda item: sort_keys(item[0])))
        except KeyError:
            print(f"[Warning] Continuum name '{cont_name}' not found in campaign '{campaign}'. Skipping.")
            continue

        plot_ccfs_in_groups(sorted_data_dict, x_key, y_key, cont_name, xlabel, ylabel,
                            title=f"CCFs between Emission Lines and {format_label(cont_name)} for {campaign.split('_')[0]}",
                            save_only=save_only, output_dir=save_folder, shared_y=True)


def plot_ccfs_in_groups(data, x_key, y_key, compare_cont, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                        title=None, save_only=False, output_dir=None, color_dict=None, rows=4, cols=2):
    """
    Plots 1D-Daten in Gruppen für CCFs.

    Parameter:
    -----------
    data : dict
        Dictionary containing the data for the plots.
    x_key : str
        Key für die x-Daten in den Dictionaries der Daten.
    y_key : str
        Key für die y-Daten in den Dictionaries der Daten.
    xlabel : str, optional
        Label für die X-Achse (nicht immer verwendet).
    ylabel : str, optional
        Label für die Y-Achse.
    shared_y : bool, optional
        Wenn True, teilen sich alle Subplots die Y-Achse.
    title : str, optional
        Titel für die gesamte Figure.
    save_only : bool, optional
        Ob die Abbildungen gespeichert werden sollen, ohne sie anzuzeigen.
    output_dir : str or Path, optional
        Verzeichnis, in dem die Abbildungen gespeichert werden.
    color_dict : dict, optional
        Dictionary mit Farben für jede Datenserie.
    rows : int, optional
        Anzahl der Subplot-Reihen.
    cols : int, optional
        Anzahl der Subplot-Spalten.

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
                        save_only=save_only, output_dir=output_dir, compare_cont=compare_cont)


def configure_ccfs_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """
    Konfiguriert die Achse für CCFs.

    Parameter:
    -----------
    ax : matplotlib.axes.Axes
        Die jeweilige Achse, auf der geplottet wird.
    row : int
        Zeilenindex des Subplots.
    col : int
        Spaltenindex des Subplots.
    ylabel : str
        Beschriftung der Y-Achse.
    color : str
        Linienfarbe.
    x_values : np.ndarray
        X-Daten für den Plot.
    y_values : np.ndarray
        Y-Daten für den Plot.
    yerr_values : np.ndarray or None
        Fehlerbalkendaten, falls vorhanden.
    line_name : str
        Name / Label für die Datenlinie.

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
    ax.set_ylim(-0.1, 0.9)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.tick_params(axis='x')
