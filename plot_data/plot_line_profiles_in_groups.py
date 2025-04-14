from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter

from plot_data.general_plot import finalize_figure, prepare_data, format_month_day
from settings import DEFAULT_OUTPUT_DIR


def plot_normalized_line_profiles_in_groups(data, save_only=False, output_dir=None,
                                            rows=4, cols=2, key_order=None, title="Normalized Line Profiles", shared_y=False):
    """
    Plottet normalisierte Linienprofile (AVG und RMS) gruppiert in Subplots (z. B. 4x2).

    Parameter:
    -----------
    data : dict
        Dictionary mit 'avg' und 'rms'-Daten.
    compare_cont : str
        Bezeichner für Dateinamen-Suffix.
    save_only : bool
        Nur speichern oder auch anzeigen.
    output_dir : Path oder None
        Verzeichnis zum Speichern der Plots.
    rows : int
        Anzahl der Subplot-Zeilen.
    cols : int
        Anzahl der Subplot-Spalten.
    key_order : list oder None
        Falls angegeben, wird diese Reihenfolge für die Linien verwendet.
    title : str
        Titel der Abbildung.

    Returns:
    -----------
    None
    """

    x_key = 'velocity space (km/s)'
    y_key = 'normalized flux'
    ylabel = 'normalized flux (km/s)'

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / "plot_line_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    if key_order is None:
        key_order = list(data["avg"].keys())

    # Vorbereitung der Datenstruktur für prepare_data
    plot_data = {}
    for line in key_order:
        plot_data[line] = {
            "avg": {
                "x": data["avg"][line]["data_dict"][x_key],
                "y": data["avg"][line]["data_dict"][y_key]
            },
            "rms": {
                "x": data["rms"][line]["data_dict"][x_key],
                "y": data["rms"][line]["data_dict"][y_key]
            }
        }

    for current_data, group_index in prepare_data(plot_data, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=True, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            if line_data is not None:
                avg_x = line_data["avg"]["x"]
                avg_y = line_data["avg"]["y"]
                rms_x = line_data["rms"]["x"]
                rms_y = line_data["rms"]["y"]

            else:
                avg_x = np.array([])
                avg_y = np.array([])
                rms_x = np.array([])
                rms_y = np.array([])

            configure_line_profile_axis(ax, row=row, col=col, ylabel=ylabel,avg_x=avg_x, avg_y=avg_y,
                                        rms_x=rms_x, rms_y=rms_y, line_name=line_name)


        # Figure finalisieren und speichern/anzeigen
        finalize_figure(
            fig=fig,
            axes=axes,
            title=title,
            group_index=group_index,
            save_only=save_only,
            output_dir=output_dir,
            x_label="Velocity Space (km/s)",
            line_profile=True
        )


def configure_line_profile_axis(ax, row, col, ylabel, avg_x, avg_y, rms_x, rms_y, line_name,
                                 line_lightcurves=False):
    """
    Konfiguriert eine Achse für das Linienprofil (AVG und RMS).

    Parameter:
    -----------
    ax : matplotlib.axes.Axes
        Achse für den aktuellen Subplot.
    row : int
        Zeilenindex im Grid.
    col : int
        Spaltenindex im Grid.
    avg_x, avg_y : array-like
        Daten für den AVG-Plot.
    rms_x, rms_y : array-like
        Daten für den RMS-Plot.
    line_name : str
        Name der Linie (für Titel).
    rows : int
        Anzahl der Zeilen (für Achsenlogik).
    cols : int
        Anzahl der Spalten (für Achsenlogik).
    line_lightcurves : bool
        (Optional) für alternative Achsenbeschriftung.

    Returns:
    -----------
    None
    """
    ax.vlines(0, -0.1, 1.5, linestyles='dashed', color='black', label='0 km/s')
    ax.plot(avg_x, avg_y, label='AVG', color='blue')
    ax.plot(rms_x, rms_y, label='RMS', color='red')

    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-0.1, 1.5)
    ax.set_title(line_name, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    if col == 0:
        if row == 0 and line_lightcurves:
            ax.set_ylabel(
                (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{\AA}^{-1}]$"),
                fontsize=12)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)


    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(1.15, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    #ax.xaxis.set_major_locator(MultipleLocator(5))
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        #ax_top.xaxis.set_major_locator(MultipleLocator(5))
        #ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=10)