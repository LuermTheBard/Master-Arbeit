import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from settings import BASE_MJD


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


def format_relative_days(mjd):
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
    base_mjd = BASE_MJD # Startwert (erster MJD)
    relative_day = mjd - base_mjd
    return relative_day


def format_yaxis(value, _):
    return f"{value:.1f}"

# -----------------------------------------------------------------------------
# FORMAT- & LAYOUT-HELFERFUNKTIONEN
# -----------------------------------------------------------------------------

def check_for_empty_rows(axes, fig, x_label):
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
    """

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
                    axes[row, col].xaxis.set_major_locator(MultipleLocator(5))  # Ticks festlegen

                    axes[row, col].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

                    # Beschriftungen nur in der untersten vorhandenen Reihe anzeigen
                    if row == lowest_row:
                        axes[row, col].set_xlabel(x_label, fontsize=12)
                        axes[row, col].tick_params(axis='x', which='both', direction='inout', labelbottom=True)


def prepare_data(data, rows, cols):
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
                None
            ))
        yield current_data, group_index


def finalize_figure(fig, axes, title, group_index, save_only, output_dir, x_label, compare_cont):
    """
    Finalisiert das Layout der Figure und speichert bzw. zeigt sie an.

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
    x_label : str
        Beschriftung für die X-Achse.
    formating : bool, optional
        Ob spezielles Formatieren aktiviert werden soll (Standard: True).

    Returns:
    -----------
    None
    """
    check_for_empty_rows(axes, fig, x_label=x_label)

    if title:
        if group_index > 0:
            fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)
        else:
            fig.suptitle(f'{title}', fontsize=14)

    if save_only and output_dir:
        save_path = output_dir / f"{title.replace(' ', '_')}_compare_cont_{compare_cont}_group_{group_index + 1}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        save_path = output_dir / f"{title.replace(' ', '_')}_compare_cont_{compare_cont}_group_{group_index + 1}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
