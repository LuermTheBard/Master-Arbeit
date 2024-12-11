import datetime

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
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

def plot_1d_lightcurves_in_groups(data, xlabel='timestamps [MJD]', ylabel='fluxes [ergs/s/cm2/A]',
                                  yerr_name='fluxerrs [ergs/s/cm2/A]', title=None, save_only=False, output_dir=None,
                                  color_dict=None):
    """
    Plot multiple 1D light curves in groups.

    Parameters:
        data (dict): Dictionary containing the data for the light curves.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        yerr_name (str): Label for the error bars.
        title (str): Title of the plot.
        save_only (bool): Whether to save the plots without displaying them.
        output_dir (str or Path): Directory to save the plots.
        color_dict (dict): Optional dictionary with colors for each light curve.
    """
    # Berechne die Anzahl der benötigten Gruppen von 8 Plots
    total_plots = len(data)
    num_groups = (total_plots + 7) // 8  # Aufrunden, um sicherzustellen, dass alle Plots abgedeckt sind

    data_items = list(data.items())

    for group_index in range(num_groups):
        # Daten für die aktuelle Gruppe extrahieren
        start_index = group_index * 8
        end_index = min(start_index + 8, total_plots)
        current_data = data_items[start_index:end_index]

        # Falls weniger als 8 Datenpunkte vorhanden sind, fülle die Liste auf
        while len(current_data) < 8:
            current_data.append((f'Empty {len(current_data) + 1}',
                                 {xlabel: np.array([]), ylabel: np.array([]), yerr_name: np.array([])}))

        # Erstelle die Figur mit Subplots
        fig, axes = plt.subplots(4, 2, figsize=(8, 12), sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0, wspace=0)  # Setze Abstände zwischen Subplots auf 0

        base_mjd = 57581.66  # Setze die Basis-MJD für die relative Darstellung

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            # Extrahiere die Werte aus der Datenstruktur
            timestamps = np.array(line_data.get(xlabel, []))
            fluxes = np.array(line_data.get(ylabel, []))
            fluxerrs = np.array(line_data.get(yerr_name, []))

            # Wähle die Farbe aus dem color_dict, falls verfügbar, sonst Standardfarbe
            color = color_dict.get(line_name, 'black') if color_dict else 'black'

            # Plotte die Daten mit Fehlerbalken, falls Daten vorhanden sind
            if timestamps.size > 0 and fluxes.size > 0:
                ax.errorbar(
                    timestamps, fluxes, yerr=fluxerrs,
                    fmt='.:', capsize=3, markersize=4, label=f'{line_name}', color=color
                )
                ax.legend(fontsize=8, loc='upper right')

            # Achsenbeschriftungen anpassen
            if col == 0:  # Linke Spalte
                ax.set_ylabel(r'$F_\lambda \, [10^{-15} \, \mathrm{erg \, cm^{-2} \, s^{-1} \, \AA^{-1}}]$')
            else:  # Rechte Spalte
                ax.yaxis.tick_right()  # Verschiebe Ticks nach rechts
                ax.yaxis.set_label_position("right")  # Verschiebe Label nach rechts
                ax.set_ylabel(r'$F_\lambda \, [10^{-15} \, \mathrm{erg \, cm^{-2} \, s^{-1} \, \AA^{-1}}]$')

            # Entferne x-Achsenbeschriftungen für nicht-untere Reihen
            if row < 3:
                ax.set_xticklabels([])

            # Gitterlinien und Ticks optimieren
            ax.xaxis.set_major_locator(MultipleLocator(4))  # Schritte von 4 Tagen auf der x-Achse
            ax.xaxis.set_major_formatter(FuncFormatter(format_relative_days))  # Nutze Formatter für relative Tage
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Automatisch passende Schritte auf der y-Achse
            ax.grid(True, linestyle='--', linewidth=0.5)

            # Füge die zweite Achse oben nur für die obersten Subplots hinzu
            if row == 0:
                ax_top = ax.secondary_xaxis('top')
                ax_top.xaxis.set_major_locator(MultipleLocator(4))  # Schritte von 4 Tagen
                ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))  # Monats- und Tagesformat
                ax_top.tick_params(axis='x', rotation=45, labelsize=8)  # Leicht gedrehte Labels

        # Entferne leere Zeilen, falls keine Daten vorhanden sind
        for row in range(4):
            if all(not axes[row, col].has_data() for col in range(2)):
                for col in range(2):
                    fig.delaxes(axes[row, col])

        # Füge die Basis-MJD an der rechten Seite der x-Achse hinzu
        fig.text(0.95, 0.04, f"Base: {base_mjd:.2f} MJD", ha='right', fontsize=10)

        # Gemeinsame Beschriftungen für alle Subplots
        fig.text(0.5, 0.04, 'Relative Days', ha='center', fontsize=12)

        # Gruppentitel hinzufügen, wenn benötigt
        if title:
            fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

        # Speichern oder Anzeigen der Grafik
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

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, light_curve_data in galaxie_campaigns_dict.items():
        super_title = f"{campaign} Lines"
        plot_1d_lightcurves_in_groups(light_curve_data["lines"], xlabel, ylabel, yerr_name, super_title, save_only, save_folder)

        super_title = f"{campaign} Continua"
        plot_1d_lightcurves_in_groups(light_curve_data["continua"], xlabel, ylabel, yerr_name, super_title, save_only, save_folder, color_dict=COLORCODE_CONTINUA_NORMALIZED)



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
