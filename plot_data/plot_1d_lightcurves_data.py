import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from plot_data.general_plot import format_yaxis, format_month_day, format_relative_days, \
    prepare_data, finalize_figure
from settings import COLORCODE_CONTINUA_NORMALIZED, BASE_MJD

matplotlib.use("Qt5Agg")


def plot_all_1d_lightcurves_in_groups(galaxie_campaigns_dict, output_dir, compare_cont, key_order=None, save_only=False):
    base_mjd = BASE_MJD

    xlabel = f"MJD - {base_mjd:.2f}"
    ylabel_cont = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
             r"\mathrm{\AA}^{-1}]$")
    ylabel_line = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1}]$")
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'



    x_key = 'timestamps [MJD]'
    y_key = 'fluxes [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, data_dict in galaxie_campaigns_dict.items():
        # Plot for lines
        super_title = f"{campaign} Lines"
        compare_cont_data = {compare_cont: data_dict["continua"][compare_cont]}
        compare_cont_data.update(data_dict["lines"])

        def sort_keys(key):
            for idx, prefix in enumerate(key_order):
                if key.startswith(prefix):
                    return idx
            return len(key_order)

        sorted_line_data_dict = dict(sorted(compare_cont_data.items(), key=lambda item: sort_keys(item[0])))

        plot_lightcurves_in_groups(sorted_line_data_dict, x_key, y_key, compare_cont, xlabel, ylabel_line, yerr_name=yerr_name, title=super_title,
                               save_only=save_only, output_dir=save_folder)

        # Plot for continua (with custom color dictionary if needed)
        super_title = f"{campaign} Continua"
        plot_lightcurves_in_groups(data_dict["continua"], x_key, y_key, compare_cont, xlabel, ylabel_cont, yerr_name=yerr_name,
                               title=super_title, save_only=save_only, output_dir=save_folder,
                               color_dict=COLORCODE_CONTINUA_NORMALIZED)


def plot_lightcurves_in_groups(data, x_key, y_key, compare_cont, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                               yerr_name=None, title=None, save_only=False,
                               output_dir=None, color_dict=None, rows=4, cols=2):
    """
    Plots 1D-Daten in Gruppen für Lightcurves.

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
    yerr_name : str, optional
        Key für die Fehlerbalken im Dictionary.
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
    for current_data, group_index in prepare_data(data, x_key, y_key, yerr_name, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=True, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            x_values = np.array(line_data.get(x_key, []))
            y_values = np.array(line_data.get(y_key, []))
            yerr_values = np.array(line_data.get(yerr_name, [])) if yerr_name else None
            color = color_dict.get(line_name, 'black') if color_dict else 'black'

            exponent_value = -15
            exponent = 10 ** exponent_value
            latex_exponent = f"10^{{{exponent_value}}}"

            ylabel_parts = ylabel.split("[")
            new_ylabel = ylabel_parts[0] + f"[{latex_exponent} " + ylabel_parts[1]

            y_values = y_values / exponent
            yerr_values = yerr_values / exponent

            configure_lightcurves_axis(ax, row, col, new_ylabel, color, x_values, y_values, yerr_values, line_name)

        finalize_figure(fig, axes, x_label=xlabel, title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir, compare_cont=compare_cont)


def configure_lightcurves_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """
    Konfiguriert die Achse für Lightcurves.

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
                format_relative_days(x_values),
                y_values,
                yerr=yerr_values,
                fmt='.:', capsize=3, markersize=4, label=f'{line_name}', color=color
            )
        else:
            ax.plot(
                format_relative_days(x_values),
                y_values, label=f'{line_name}', color=color
            )

        ax.legend(fontsize=8, loc='upper right')

    if col == 0:
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(-0.19, 0.5)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(1.19, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=10)


def plot_lightcurve(data, xlabel, ylabel, yerr_name, ax):
    """
    Plots a single lightcurve with normalized flux values.
    """
    for key, values in data.items():
        x = values[xlabel]
        y = values[ylabel]
        y_err = values[yerr_name]

        # Normierung der y-Werte
        max_y = max(y)
        y = y / max_y
        y_err = y_err / max_y  # Fehler ebenfalls normieren

        ax.errorbar(x, y, y_err, label=f"{key}", linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("fluxes (normalized)")
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8)


def plot_lightcurve_with_offset(data, xlabel, ylabel, yerr_name, ax, y_offset=0.2, y_multiplier=1.0):
    """
    Plots multiple lightcurves with normalized flux values and offsets in y-direction.

    Parameters:
    - data: dict
        A dictionary containing the lightcurve data. Each key corresponds to a dataset,
        and its value is another dictionary with keys for xlabel, ylabel, and yerr_name.
    - xlabel: str
        The key for x-axis data.
    - ylabel: str
        The key for y-axis data.
    - yerr_name: str
        The key for the y-error data.
    - ax: matplotlib.axes.Axes
        The matplotlib Axes object to plot on.
    - label_prefix: str
        Prefix for the legend labels.
    - y_offset: float
        The vertical offset to apply between datasets (default is 0.2).
    """
    for i, (key, values) in enumerate(data.items()):
        x = values[xlabel]
        y = values[ylabel]
        y_err = values[yerr_name]

        # Normierung der y-Werte
        max_y = max(y)
        y = y / max_y
        y_err = y_err / max_y  # Fehler ebenfalls normieren

        # Offset hinzufügen
        y_offset_applied = y * y_multiplier + i * y_offset

        ax.errorbar(x, y_offset_applied, y_err, label=f"{key}", linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fluxes (normalized with offset)")
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8)


def save_and_show_plot(fig, save_path, save_only):
    """
    Saves the figure and optionally shows it.
    """
    fig.savefig(save_path)
    if not save_only:
        plt.show()
    plt.close(fig)


def plot_1d_lightcurves(galaxie_campaigns_dict, output_dir, save_only=False):
    xlabel = 'timestamps [MJD]'
    ylabel = 'fluxes [ergs/s/cm2/A]'
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, light_curve_data in galaxie_campaigns_dict.items():
        super_title = campaign

        # Einzelne Plots: Linien
        fig_lines, ax_lines = plt.subplots(figsize=(8, 6))
        plot_lightcurve(light_curve_data["lines"], xlabel, ylabel, yerr_name, ax_lines)
        ax_lines.set_title("1D Lightcurves of lines")
        save_and_show_plot(fig_lines, save_folder / f"{campaign}_lines.pdf", save_only)

        # Einzelne Plots: Kontinua
        fig_continua, ax_continua = plt.subplots(figsize=(8, 6))
        plot_lightcurve(light_curve_data["continua"], xlabel, ylabel, yerr_name, ax_continua)
        ax_continua.set_title("1D Lightcurves of continua")
        save_and_show_plot(fig_continua, save_folder / f"{campaign}_continua.pdf", save_only)

        # Kombinierter Plot: Linien und Kontinua
        fig_combined, axs_combined = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        plot_lightcurve(light_curve_data["lines"], xlabel, ylabel, yerr_name, axs_combined[0])
        axs_combined[0].set_title("1D Lightcurves of lines")

        plot_lightcurve(light_curve_data["continua"], xlabel, ylabel, yerr_name, axs_combined[1])
        axs_combined[1].set_title("1D Lightcurves of continua")

        fig_combined.suptitle(f"Lightcurves of lines and continua of {super_title}")
        fig_combined.tight_layout(rect=(0, 0, 1, 0.95))
        save_and_show_plot(fig_combined, save_folder / f"{campaign}_combined.pdf", save_only)


def plot_1d_lightcurves_with_offset(galaxie_campaigns_dict, output_dir, save_only=False, y_offset=0.2):
    """
    Plots 1D lightcurves with normalized flux values and y-offsets for clarity.

    Parameters:
    - galaxie_campaigns_dict: dict
        Dictionary containing campaigns with lightcurve data for lines and continua.
    - output_dir: Path
        Path to the output directory where plots will be saved.
    - save_only: bool
        If True, only saves the plots without displaying them.
    - y_offset: float
        Vertical offset applied between multiple datasets (default is 0.2).
    """
    xlabel = 'timestamps [MJD]'
    ylabel = 'fluxes [ergs/s/cm2/A]'
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves_with_offset"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, light_curve_data in galaxie_campaigns_dict.items():
        super_title = campaign

        # Einzelne Plots: Linien
        fig_lines, ax_lines = plt.subplots(figsize=(8, 6))
        plot_lightcurve_with_offset(
            light_curve_data["lines"], xlabel, ylabel, yerr_name, ax_lines, y_offset
        )
        ax_lines.set_title("1D Lightcurves of lines (with offset)")
        save_and_show_plot(fig_lines, save_folder / f"{campaign}_lines_offset.pdf", save_only)

        # Einzelne Plots: Kontinua
        fig_continua, ax_continua = plt.subplots(figsize=(8, 6))
        plot_lightcurve_with_offset(
            light_curve_data["continua"], xlabel, ylabel, yerr_name, ax_continua, y_offset
        )
        ax_continua.set_title("1D Lightcurves of continua (with offset)")
        save_and_show_plot(fig_continua, save_folder / f"{campaign}_continua_offset.pdf", save_only)

        # Kombinierter Plot: Linien und Kontinua
        fig_combined, axs_combined = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        plot_lightcurve_with_offset(
            light_curve_data["lines"], xlabel, ylabel, yerr_name, axs_combined[0], 0.25, y_multiplier=1.0
        )
        axs_combined[0].set_title("1D Lightcurves of lines (with offset)")

        plot_lightcurve_with_offset(
            light_curve_data["continua"], xlabel, ylabel, yerr_name, axs_combined[1], y_offset
        )
        axs_combined[1].set_title("1D Lightcurves of continua (with offset)")

        fig_combined.suptitle(f"Lightcurves of lines and continua of {super_title} (with offset)")
        fig_combined.tight_layout(rect=(0, 0, 1, 0.95))
        save_and_show_plot(fig_combined, save_folder / f"{campaign}_combined_offset.pdf", save_only)
