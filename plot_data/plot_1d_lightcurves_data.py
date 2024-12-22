import datetime

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator

matplotlib.use("Qt5Agg")

COLORCODE_CONTINUA = {
    'Cont4010': (100, 0, 128, 255),
    'Cont4200': (80, 0, 255, 255),
    'Cont4440': (0, 34, 255, 255),
    'Cont4765': 'dodgerblue',
    'Cont5100': 'tab:green',
    'Cont5600': 'gold',
    'Cont6045': (255, 180, 0, 255),
    'Cont6110': (255, 99, 0, 255),
    'Cont6880': (222, 0, 0, 255),
    'Cont7390': (190, 0, 0, 255),
    'Cont8015': (165, 0, 0, 255),
    'Cont8900': (139, 0, 0, 255)
}


def normalize_color_values(colorcode_dict):
    """
    Normalisiert RGBA-Tupel im Bereich 0-255 zu RGBA-Tupeln im Bereich 0-1,
    falls erforderlich.
    """
    normalized_dict = {}
    for key, color in colorcode_dict.items():
        if isinstance(color, tuple) and len(color) == 4:
            # Normiere die Farbwerte in den Bereich 0-1
            normalized_dict[key] = tuple(c / 255.0 for c in color)
        else:
            # Wenn es sich um einen gültigen Farbstring handelt, behalte ihn bei
            normalized_dict[key] = color
    return normalized_dict


COLORCODE_CONTINUA_NORMALIZED = normalize_color_values(COLORCODE_CONTINUA)


def mjd_to_date(mjd):
    """Konvertiere MJD in ein Kalenderdatum."""
    mjd_start_date = datetime.datetime(1858, 11, 17)  # MJD Startdatum
    return mjd_start_date + datetime.timedelta(days=mjd)


def format_month_day(mjd, pos):
    """Formatter für die obere Achse, der MJD in Monats- und Tagesformat umwandelt."""
    date = mjd_to_date(mjd)
    return date.strftime('%b %d')  # Format: 'Aug 01', 'Sep 15', ...


def format_relative_days(mjd, pos):
    """Formatter für relative Tage (mit erstem Wert als Referenz)."""
    base_mjd = 57581.66  # Startwert (erster MJD)
    relative_day = mjd - base_mjd
    return f"{int(relative_day)}"  # Zeige nur relative Tage


def get_type_methodes(data_type):
    if data_type == 'lightcurves':
        return configure_axis_lightcurves, finalize_figure_lightcurves
    elif data_type == 'ccfs':
        return configure_axis_ccfs, finalize_figure_ccfs
    else:
        raise ValueError(f"Wrong datatype. Expected 'lightcurves' or 'ccfs', got '{data_type}'.")


def plot_1d_data_in_groups(data, x_key, y_key, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                           yerr_name=None, title=None, save_only=False,
                           output_dir=None, color_dict=None, rows=4, cols=2, data_type='lightcurves'):
    """
    Plot multiple 1D data sets in groups.

    Parameters:
        data_type:
        shared_y:
        y_key:
        x_key:
        data (dict): Dictionary containing the data for the plots.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        yerr_name (str): Optional name for the error bars.
        title (str): Title of the plot.
        save_only (bool): Whether to save the plots without displaying them.
        output_dir (str or Path): Directory to save the plots.
        color_dict (dict): Optional dictionary with colors for each data series.
        rows (int): Number of rows in the subplot grid.
        cols (int): Number of columns in the subplot grid.
    """

    # Get the appropriate methods based on the data type
    configure_axis, finalize_figure = get_type_methodes(data_type)

    # Main plotting loop
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

            # Call the correct configure_axis based on data_type
            configure_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name)

        # Call the correct finalize_figure based on data_type
        finalize_figure(fig, axes, title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir)


def prepare_data(data, xlabel, ylabel, yerr_name, rows, cols):
    """Prepare data and ensure all groups are filled."""
    total_plots = len(data)
    num_groups = (total_plots + (rows * cols) - 1) // (rows * cols)  # Calculate groups based on grid size
    data_items = list(data.items())

    for group_index in range(num_groups):
        start_index = group_index * (rows * cols)
        end_index = min(start_index + (rows * cols), total_plots)
        current_data = data_items[start_index:end_index]

        # Fill missing plots with empty placeholders
        while len(current_data) < (rows * cols):
            current_data.append((f'Empty {len(current_data) + 1}',
                                 {xlabel: np.array([]), ylabel: np.array([]),
                                  yerr_name: np.array([]) if yerr_name else np.array([])}))
        yield current_data, group_index


def configure_axis_lightcurves(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """Configure individual subplot axes."""
    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(x_values, y_values, yerr=yerr_values,
                        fmt='.:', capsize=3, markersize=4, label=f'{line_name}', color=color)
        else:
            ax.plot(x_values, y_values, label=f'{line_name}', color=color)

        ax.legend(fontsize=8, loc='upper right')

    if col == 0:
        ax.set_ylabel(ylabel)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(format_relative_days))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, linestyle='--', linewidth=0.5)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(4))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=8)


def finalize_figure_lightcurves(fig, axes, title, group_index, save_only, output_dir):
    """Finalize figure layout and save or show."""

    check_for_empty_rows(axes, fig, x_label='Relative Days')

    if title:
        fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def check_for_empty_rows(axes, fig, x_label, formating=True):
    base_mjd = 57581.66
    # Löschen leerer Reihen
    for row in range(4):
        if all(not axes[row, col].has_data() for col in range(2)):
            for col in range(2):
                fig.delaxes(axes[row, col])
    # Ermittlung der untersten verbleibenden Reihe
    remaining_rows = [row for row in range(4) if any(axes[row, col].has_data() for col in range(2))]
    if remaining_rows:  # Überprüfen, ob noch Reihen existieren
        lowest_row = max(remaining_rows)

        # X-Achsenbeschriftungen und Ticks setzen
        for row in remaining_rows:
            for col in range(2):
                if axes[row, col].has_data():  # Stelle sicher, dass die Achse existiert und Daten hat
                    axes[row, col].xaxis.set_major_locator(MultipleLocator(2))  # Ticks festlegen

                    if formating:
                        axes[row, col].xaxis.set_major_formatter(FuncFormatter(format_relative_days))  # Formatierung
                    else:
                        axes[row, col].xaxis.set_major_formatter(
                            plt.FuncFormatter(lambda x, pos: f"{x}"))  # Standard-Werte anzeigen

                    axes[row, col].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

                    if row == lowest_row:  # Nur in der untersten Reihe Beschriftungen und Ticks anzeigen
                        axes[row, col].tick_params(axis='x', which='both', direction='inout', labelbottom=True)
    len_remaining_rows = len(remaining_rows)
    text_heigth = 0.04 + (4 - len_remaining_rows) * 0.20
    fig.text(0.95, text_heigth, f"Base: {base_mjd:.2f} MJD", ha='right', fontsize=10)
    fig.text(0.5, text_heigth, x_label, ha='center', fontsize=12)


def configure_axis_ccfs(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """Configure individual subplot axes."""
    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(x_values, y_values, yerr=yerr_values,
                        fmt='.:', capsize=3, markersize=4, label=f'{line_name}', color=color)
        else:
            ax.plot(x_values, y_values, label=f'{line_name}', color=color)

        ax.legend(fontsize=8, loc='upper right')

    if col == 0:
        ax.set_ylabel(ylabel)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel)

    if row < 3:
        ax.set_xticklabels([])

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, linestyle='--', linewidth=0.5)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(2))
        ax_top.tick_params(axis='x')


def finalize_figure_ccfs(fig, axes, title, group_index, save_only, output_dir):
    """Finalize figure layout and save or show."""

    check_for_empty_rows(axes, fig, x_label='Time Lag', formating=False)

    if title:
        fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_all_1d_lightcurves_in_groups(galaxie_campaigns_dict, output_dir, save_only=False):
    xlabel = 'timestamps [MJD]'
    ylabel = 'fluxes [ergs/s/cm2/A]'
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    x_key = xlabel
    y_key = ylabel

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, data_dict in galaxie_campaigns_dict.items():
        # Plot for lines
        super_title = f"{campaign} Lines"
        plot_1d_data_in_groups(data_dict["lines"], x_key, y_key, xlabel, ylabel, yerr_name=yerr_name, title=super_title,
                               save_only=save_only, output_dir=save_folder)

        # Plot for continua (with custom color dictionary if needed)
        super_title = f"{campaign} Continua"
        plot_1d_data_in_groups(data_dict["continua"], x_key, y_key, xlabel, ylabel, yerr_name=yerr_name,
                               title=super_title, save_only=save_only, output_dir=save_folder,
                               color_dict=COLORCODE_CONTINUA_NORMALIZED)


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
