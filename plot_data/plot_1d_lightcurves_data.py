import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator

matplotlib.use("Qt5Agg")


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


def format_relative_days(mjd, pos):
    """Formatter für relative Tage (mit erstem Wert als Referenz)."""
    base_mjd = 57581.66  # Startwert (erster MJD)
    relative_day = mjd - base_mjd
    return f"{int(relative_day)}"  # Zeige nur relative Tage

def plot_1d_lightcurves_in_groups(data, xlabel='timestamps [MJD]', ylabel='fluxes [ergs/s/cm2/A]',
                                  yerr_name='fluxerrs [ergs/s/cm2/A]', title=None, save_only=False, output_dir=None):
    # Berechne die Anzahl der benötigten Gruppen von 6 Plots
    total_plots = len(data)
    num_groups = (total_plots + 5) // 6  # Aufrunden, um sicherzustellen, dass alle Plots abgedeckt sind

    data_items = list(data.items())

    for group_index in range(num_groups):
        # Daten für die aktuelle Gruppe extrahieren
        start_index = group_index * 6
        end_index = min(start_index + 6, total_plots)
        current_data = data_items[start_index:end_index]

        # Falls weniger als 6 Datenpunkte vorhanden sind, fülle die Liste auf
        while len(current_data) < 6:
            current_data.append((f'Empty {len(current_data) + 1}',
                                 {xlabel: np.array([]), ylabel: np.array([]), yerr_name: np.array([])}))

        # Erstelle die Figur mit Subplots
        fig, axes = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0.2, wspace=0.1)

        base_mjd = 57581.66  # Setze die Basis-MJD für die relative Darstellung

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            # Extrahiere die Werte aus der Datenstruktur
            timestamps = np.array(line_data.get(xlabel, []))
            fluxes = np.array(line_data.get(ylabel, []))
            fluxerrs = np.array(line_data.get(yerr_name, []))

            # Plotte die Daten mit Fehlerbalken, falls Daten vorhanden sind
            if timestamps.size > 0 and fluxes.size > 0:
                ax.errorbar(
                    timestamps, fluxes, yerr=fluxerrs,
                    fmt='.:', capsize=3, markersize=4, label=f'{line_name}'
                )
                ax.legend(fontsize=8, loc='upper right')

            # Achsenbeschriftungen nur an den äußeren Subplots
            if row < 2:  # Entferne x-Achsenbeschriftung für die obere Reihe
                ax.set_xticklabels([])
            if col > 0:  # Entferne y-Achsenbeschriftung für die rechte Spalte
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r'$F_\lambda \, [10^{-15} \, \mathrm{erg \, cm^{-2} \, s^{-1} \, \AA^{-1}}]$')

            # Gitterlinien und Ticks optimieren
            ax.xaxis.set_major_locator(MultipleLocator(4))  # Schritte von 2 Tagen auf der x-Achse
            ax.xaxis.set_major_formatter(FuncFormatter(format_relative_days))  # Nutze Formatter für relative Tage
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Automatisch passende Schritte auf der y-Achse
            ax.grid(True, linestyle='--', linewidth=0.5)

        for row in range(3):
            if all(not ax.has_data() for ax in axes[row, :]):
                for col in range(2):
                    fig.delaxes(axes[row, col])

        # Füge die Basis-MJD an der rechten Seite der x-Achse hinzu
        fig.text(0.95, 0.04, f"Base: {base_mjd:.2f} MJD", ha='right', fontsize=10)

        # Gemeinsame Beschriftungen für alle Subplots
        fig.text(0.5, 0.04, xlabel, ha='center', fontsize=12)

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
        plot_1d_lightcurves_in_groups(light_curve_data["continua"], xlabel, ylabel, yerr_name, super_title, save_only, save_folder)



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
