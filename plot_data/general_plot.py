import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator


# -----------------------------------------------------------------------------
# UTILITY-FUNKTIONEN
# -----------------------------------------------------------------------------

def mjd_to_date(mjd):
    """
    Konvertiere MJD (Modified Julian Date) in ein Kalenderdatum.

    Parameter:
    -----------
    mjd : float
        Wert des Modified Julian Date.

    Returns:
    -----------
    datetime.datetime
        Entsprechendes Kalenderdatum basierend auf dem MJD-Startdatum 1858-11-17.
    """
    mjd_start_date = datetime.datetime(1858, 11, 17)  # MJD Startdatum
    return mjd_start_date + datetime.timedelta(days=mjd)


def format_month_day(mjd, pos):
    """
    Formatter für die obere Achse, der MJD in Monats- und Tagesformat (z.B. 'Aug 01') umwandelt.

    Parameter:
    -----------
    mjd : float
        Modified Julian Date.
    pos : int
        Position (für Matplotlib-Formatter, hier nicht weiter verwendet).

    Returns:
    -----------
    str
        Datums-String im Format 'Monat Tag' (z.B. 'Aug 01').
    """
    date = mjd_to_date(mjd)
    return date.strftime('%b %d')


def format_relative_days(mjd, pos):
    """
    Formatter für die X-Achse, der die relativen Tage (gegenüber einem Basismjd) anzeigt.

    Parameter:
    -----------
    mjd : float
        Modified Julian Date.
    pos : int
        Position (für Matplotlib-Formatter, hier nicht weiter verwendet).

    Returns:
    -----------
    str
        String der Form '0', '1', '2', ... basierend auf dem Abstand zum base_mjd.
    """
    base_mjd = 57581.66  # Startwert (erster MJD)
    relative_day = mjd - base_mjd
    return f"{int(relative_day)}"


# -----------------------------------------------------------------------------
# FORMAT- & LAYOUT-HELFERFUNKTIONEN
# -----------------------------------------------------------------------------

def check_for_empty_rows(axes, fig, x_label, formating=True):
    """
    Prüft, ob in der Figure leere Subplot-Zeilen existieren, und entfernt diese gegebenenfalls.
    Außerdem wird die X-Achsenbeschriftung und -Formatierung für die verbleibenden Reihen gesetzt.

    Parameter:
    -----------
    axes : numpy.ndarray
        Array von Matplotlib-Achsenobjekten, typischerweise erzeugt durch plt.subplots().
    fig : matplotlib.figure.Figure
        Matplotlib-Figure, die die Subplots enthält.
    x_label : str
        Beschriftung für die X-Achse.
    formating : bool, optional
        Wenn True, wird ein spezieller Formatter für die Achsenlabels verwendet.
        Andernfalls wird der Standardformatter genutzt.
    """
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
                        axes[row, col].xaxis.set_major_formatter(FuncFormatter(format_relative_days))
                    else:
                        axes[row, col].xaxis.set_major_formatter(
                            plt.FuncFormatter(lambda x, pos: f"{x}")
                        )

                    axes[row, col].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

                    # Beschriftungen nur in der untersten vorhandenen Reihe anzeigen
                    if row == lowest_row:
                        axes[row, col].tick_params(axis='x', which='both', direction='inout', labelbottom=True)

    len_remaining_rows = len(remaining_rows)
    text_heigth = 0.04 + (4 - len_remaining_rows) * 0.20
    fig.text(0.95, text_heigth, f"Base: {base_mjd:.2f} MJD", ha='right', fontsize=10)
    fig.text(0.5, text_heigth, x_label, ha='center', fontsize=12)


def prepare_data(data, xlabel, ylabel, yerr_name, rows, cols):
    """
    Bereitet die Daten vor, indem sie gruppenweise anhand der verfügbaren Subplot-Größe (rows x cols) aufgeteilt werden.
    Eventuell unvollständige Gruppen werden mit leeren Platzhaltern aufgefüllt.

    Parameter:
    -----------
    data : dict
        Dictionary mit den zu plottenden Daten. Keys sind beispielsweise Kurvennamen,
        Werte sind selbst wiederum Dictionaries mit x- und y-Daten.
    xlabel : str
        Key im Dictionary, unter dem sich die x-Daten befinden.
    ylabel : str
        Key im Dictionary, unter dem sich die y-Daten befinden.
    yerr_name : str or None
        Key für die Fehlerbalken (optional). Wenn None, werden keine Fehlerbalken gezeichnet.
    rows : int
        Anzahl der Subplot-Zeilen.
    cols : int
        Anzahl der Subplot-Spalten.

    Yields:
    -----------
    current_data : list of tuples
        Teilmenge der Daten, die in einem Subplot-Grid (rows x cols) dargestellt werden sollen.
    group_index : int
        Index der aktuellen Gruppe (0-basiert).
    """
    total_plots = len(data)
    num_groups = (total_plots + (rows * cols) - 1) // (rows * cols)  # Anzahl der benötigten Gruppen
    data_items = list(data.items())

    for group_index in range(num_groups):
        start_index = group_index * (rows * cols)
        end_index = min(start_index + (rows * cols), total_plots)
        current_data = data_items[start_index:end_index]

        # Mit leeren Platzhaltern auffüllen, falls die letzte Gruppe nicht vollständig ist
        while len(current_data) < (rows * cols):
            current_data.append((
                f'Empty {len(current_data) + 1}',
                {
                    xlabel: np.array([]),
                    ylabel: np.array([]),
                    yerr_name: np.array([]) if yerr_name else np.array([])
                }
            ))
        yield current_data, group_index


# -----------------------------------------------------------------------------
# DATENTYP-SPEZIFISCHE FUNKTIONSAUSWAHL
# -----------------------------------------------------------------------------

def get_type_methodes(data_type):
    """
    Gibt passende Funktionen zur Achsenkonfiguration und zur Finalisierung der Figure
    abhängig vom übergebenen Datentyp (z.B. 'lightcurves' oder 'ccfs') zurück.

    Parameter:
    -----------
    data_type : str
        Erwartet entweder 'lightcurves' oder 'ccfs'.

    Returns:
    -----------
    tuple
        Ein 2-Tupel mit (configure_axis_function, finalize_figure_function).

    Raises:
    -----------
    ValueError
        Falls ein unbekannter Datentyp übergeben wird.
    """
    if data_type == 'lightcurves':
        return configure_axis_lightcurves, finalize_figure_lightcurves
    elif data_type == 'ccfs':
        return configure_axis_ccfs, finalize_figure_ccfs
    else:
        raise ValueError(f"Wrong datatype. Expected 'lightcurves' or 'ccfs', got '{data_type}'.")


# -----------------------------------------------------------------------------
# HAUPT-PLOT-FUNKTION
# -----------------------------------------------------------------------------

def plot_1d_data_in_groups(data, x_key, y_key, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                           yerr_name=None, title=None, save_only=False,
                           output_dir=None, color_dict=None, rows=4, cols=2, data_type='lightcurves'):
    """
    Plot multiple 1D data sets in Gruppen, abhängig von einem angegebenen Datentyp.

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
    data_type : str, optional
        Entweder 'lightcurves' oder 'ccfs', bestimmt die Konfiguration der Achsen.

    Returns:
    -----------
    None
    """
    # Rufe die passenden Methoden für den gewählten Datentyp ab
    configure_axis, finalize_figure = get_type_methodes(data_type)

    # Haupt-Schleife zur Erstellung der Subplots in Gruppen
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

            # Aufruf der Konfigurationsmethode (abhängig vom Datentyp)
            configure_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name)

        # Aufruf der Abschlussmethode (abhängig vom Datentyp)
        finalize_figure(fig, axes, title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir)


# -----------------------------------------------------------------------------
# LIGHTCURVE-SPEZIFISCHE FUNKTIONEN
# -----------------------------------------------------------------------------

def configure_axis_lightcurves(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """
    Konfiguriert die Achse für 'lightcurves'-Daten.

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
    """
    Finalisiert das Layout der Figure für 'lightcurves'-Plots und speichert bzw. zeigt sie an.

    Parameter:
    -----------
    fig : matplotlib.figure.Figure
        Die Figure, die finalisiert werden soll.
    axes : numpy.ndarray
        Array von Matplotlib-Achsenobjekten.
    title : str
        Titel der Abbildung.
    group_index : int
        Index der aktuellen Gruppe (0-basiert).
    save_only : bool
        Ob die Abbildung nur gespeichert werden soll (ohne plt.show()).
    output_dir : str or Path
        Pfad zum Speicherort.

    Returns:
    -----------
    None
    """
    check_for_empty_rows(axes, fig, x_label='Relative Days')

    if title:
        fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------------------------------
# CCF-SPEZIFISCHE FUNKTIONEN
# -----------------------------------------------------------------------------

def configure_axis_ccfs(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name):
    """
    Konfiguriert die Achse für 'ccfs'-Daten.

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
    """
    Finalisiert das Layout der Figure für 'ccfs'-Plots und speichert bzw. zeigt sie an.

    Parameter:
    -----------
    fig : matplotlib.figure.Figure
        Die Figure, die finalisiert werden soll.
    axes : numpy.ndarray
        Array von Matplotlib-Achsenobjekten.
    title : str
        Titel der Abbildung.
    group_index : int
        Index der aktuellen Gruppe (0-basiert).
    save_only : bool
        Ob die Abbildung nur gespeichert werden soll (ohne plt.show()).
    output_dir : str or Path
        Pfad zum Speicherort.

    Returns:
    -----------
    None
    """
    check_for_empty_rows(axes, fig, x_label='Time Lag', formating=False)

    if title:
        fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
