import datetime

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

from import_data import import_1d_lightcurve_data
from plot_utils import ensure_output_dir, calculate_standard_error_for_lightcurves
from settings import BASE_MJD, DEFAULT_OUTPUT_DIR, ERR_CORRECTION, ERR_SET

matplotlib.use('Qt5Agg')


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

    mjd_start_date = datetime.datetime(1858, 11, 17)  # MJD Startdatum
    return mjd_start_date + datetime.timedelta(days=mjd)


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

    date = mjd_to_date(mjd+ 57582.66)
    return date.strftime('%b %d')


def format_relative_days(mjd):
    """
    Computes the number of days relative to a global BASE_MJD value.

    Parameters:
    -----------
    mjd : float
        Modified Julian Date.

    Returns:
    -----------
    float
        Number of days relative to BASE_MJD.
    """

    base_mjd = BASE_MJD # Startwert (erster MJD)
    relative_day = mjd - base_mjd
    return relative_day


def format_yaxis(value, _):
    return f"{value:.1f}"

# -----------------------------------------------------------------------------
# FORMAT- & LAYOUT-HELFERFUNKTIONEN
# -----------------------------------------------------------------------------


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


def check_for_empty_rows(axes, fig, x_label, line_profile=False):
    n_rows, n_cols = axes.shape

    # x_label: str -> für alle Spalten gleich
    per_col_labels = None
    if isinstance(x_label, (tuple, list)):
        per_col_labels = x_label

    # 1) Einzelne leere Achsen entfernen
    for row in range(n_rows):
        for col in range(n_cols):
            if not is_valid_axis(axes[row, col], fig):
                # falls die Achse noch in der Figure hängt, entfernen
                if axes[row, col] in fig.axes:
                    fig.delaxes(axes[row, col])

    # 2) Unterste sichtbare Zeile PRO SPALTE bestimmen
    lowest_row_per_col = {}
    for col in range(n_cols):
        valid_rows = [
            row for row in range(n_rows)
            if (axes[row, col] in fig.axes) and is_valid_axis(axes[row, col], fig)
        ]
        if valid_rows:
            lowest_row_per_col[col] = max(valid_rows)

    # 3) Ticks/Formatter setzen + X-Label auf die unterste sichtbare Achse jeder Spalte
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            if ax not in fig.axes:
                continue
            if not is_valid_axis(ax, fig):
                continue

            # Formatter wie bei dir
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

            # X-Labels nur unten pro Spalte
            is_bottom_in_col = (col in lowest_row_per_col) and (row == lowest_row_per_col[col])
            if is_bottom_in_col:
                if per_col_labels is not None:
                    label = per_col_labels[col] if col < len(per_col_labels) else per_col_labels[-1]
                else:
                    label = x_label
                ax.set_xlabel(label, fontsize=9)
                ax.tick_params(axis='x', which='both', direction='out', labelbottom=True)
            else:
                ax.set_xticklabels([])

            # Locator-Defaults (dein line_profile nutzt sowieso 2500 in configure_line_profile_axis)
            if not line_profile and per_col_labels is None:
                ax.xaxis.set_major_locator(MultipleLocator(5))

def prepare_data(data, rows, cols):
    """
    Splits input data into groups that fit within a subplot grid of size (rows x cols).
    Incomplete groups are filled with empty placeholders.

    Parameters:
    -----------
    data : dict
        Dictionary of items to plot. Keys are labels, values are data objects.
    rows : int
        Number of subplot rows per group.
    cols : int
        Number of subplot columns per group.

    Yields:
    -----------
    current_data : list of tuples
        List of (key, value) pairs for the current group.
    group_index : int
        Index of the current group (0-based).
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


def finalize_figure(fig, axes, title, group_index, save_only, output_dir, x_label, compare_cont=None, line_profile=False, file_name=None, supertitle=None):
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
    compare_cont : str, optional
        Optional identifier for use in filenames (e.g., when comparing continua).
    line_profile : bool, optional
        Whether the plots represent line profiles (affects formatting).
    file_name : str, optional
        Custom filename prefix. If None, the title is used instead.

    Returns:
    -----------
    None
    """

    check_for_empty_rows(axes, fig, x_label=x_label, line_profile=line_profile)

    if supertitle:
        if group_index > 0:
            fig.suptitle(f'{supertitle} - Group {group_index + 1}', fontsize=14, y=0.95)
        else:
            fig.suptitle(f'{supertitle}', fontsize=14)

    if title is None:
        title = f'No Title'

    if output_dir:
        if file_name:
            save_path = output_dir / f"{file_name}_group_{group_index + 1}.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            save_path = output_dir / f"{file_name}_group_{group_index + 1}.png"
            plt.savefig(save_path, bbox_inches='tight')
        else:
            save_path = output_dir / f"{title.replace(' ', '_')}_compare_cont_{compare_cont}_group_{group_index + 1}.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            save_path = output_dir / f"{title.replace(' ', '_')}_compare_cont_{compare_cont}_group_{group_index + 1}.png"
            plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if not save_only:
        plt.show()
    plt.close(fig)



def get_f_var(
    line_names=None,
    cont_names=None,
    campaign="NGC4593_optical_calibrated",
    output_dir=DEFAULT_OUTPUT_DIR
):
    # Sicherheitscheck: mindestens eine Liste muss angegeben sein
    if not line_names and not cont_names:
        print("Please specify at least one line_name or cont_name")
        return

    # Falls ein einzelner Name übergeben wird, in Liste umwandeln
    if isinstance(line_names, str):
        line_names = [line_names]
    if isinstance(cont_names, str):
        cont_names = [cont_names]

    # Importiere Daten einmal
    lightcurver_data = import_1d_lightcurve_data()

    # Ergebnisse speichern
    results = []

    # --- Hilfsfunktion für Berechnung ---
    def compute_f_var(lightcurve_type: str, lightcurve_name: str):
        data = lightcurver_data[campaign][lightcurve_type][lightcurve_name]
        flux = np.array(data['fluxes [ergs/s/cm2/A]'], dtype=float)

        err_corr=ERR_CORRECTION.get(lightcurve_name, None)
        err_set=ERR_SET.get(lightcurve_name, None)

        # Standardfehler je Punkt (so wie du es bisher machst)
        flux_err = calculate_standard_error_for_lightcurves(
            flux,
            np.array(data['fluxerrs [ergs/s/cm2/A]'], dtype=float),
            err_correction=err_corr,
            err_set=err_set
        )

        N = len(flux)
        if N < 2:
            print(f"{lightcurve_type[:-1]} '{lightcurve_name}': not enough points (N < 2).")
            results.append({
                "type": lightcurve_type,
                "name": lightcurve_name,
                "F_var": 0.0,
                "flux": flux.tolist(),
                "flux_err": flux_err.tolist(),
                "percent_errors": (np.full_like(flux, np.nan)).tolist()
            })
            return

        N = len(flux)
        unweighted_mean_flux = (1/N) * np.sum(flux)
        sigma2 = (1/(N-1)) * np.sum((flux-unweighted_mean_flux)**2)
        delta2 = (1/N) * np.sum(flux_err**2)

        F_var = (np.sqrt(sigma2 - delta2))/unweighted_mean_flux

        percent_errors = np.divide(
            flux_err, flux, out=np.full_like(flux_err, np.nan), where=flux != 0
        )

        results.append({
            "type": lightcurve_type,
            "name": lightcurve_name,
            "F_var": float(F_var),
            "flux": flux.tolist(),
            "flux_err": flux_err.tolist(),
            "percent_errors": percent_errors.tolist()
        })

        print(f"{lightcurve_type[:-1]} '{lightcurve_name}': F_var = {F_var:.4f}")

    # --- Berechnung für Linien ---
    if line_names:
        for name in line_names:
            compute_f_var("lines", name)

    # --- Berechnung für Kontinua ---
    if cont_names:
        for name in cont_names:
            compute_f_var("continua", name)

    # Ergebnisse speichern
    output_file_dir = output_dir / "f_var_data"
    ensure_output_dir(output_file_dir)
    output_file = output_file_dir / f"{campaign}_f_var.txt"

    with open(output_file, "w") as f:
        for r in results:
            header = f"{r['type'][:-1]} {r['name']}"
            f.write(f"{header}\n")
            f.write(f"F_var = {r['F_var']:.6f}\n")
            f.write("i\tf_i [ergs/s/cm2/A]\tf_i_err [ergs/s/cm2/A]\tf_i_err/f_i\n")

            flux_list = r["flux"]
            flux_err_list = r["flux_err"]
            perc_list = r["percent_errors"]

            for i, (fi, ei, pe) in enumerate(zip(flux_list, flux_err_list, perc_list)):
                # Wissenschaftliche Notation für Flüsse; Verhältnis als Dezimalzahl
                fi_str = f"{fi:.6e}" if fi is not None else "nan"
                ei_str = f"{ei:.6e}" if ei is not None else "nan"
                # pe kann NaN sein (bei fi==0)
                pe_str = "nan" if (pe is None or (isinstance(pe, float) and np.isnan(pe))) else f"{pe:.6f}"
                f.write(f"{i}\t{fi_str}\t{ei_str}\t{pe_str}\n")

            f.write("\n")  # Leerzeile zwischen Blöcken

    print(f"Saved F_var results with per-point table to: {output_file}")

    return results






# get_f_var(line_names=["HAlpha", "HBeta", "HGamma", "HDelta", "OI8446"])

# plot_ion_pot_FWHM_against_lag(lines=['HAlpha', 'HBeta', 'HGamma', "HDelta",'HeI5875', 'HeII4685', 'LyAlpha_not_optical_calibrated', "LyAlpha_not_optical_calibrated", "SiIV1393_not_optical_calibrated","CIV1548_not_optical_calibrated", "HeII1640_not_optical_calibrated"])
# plot_ccfs_lags_against_angstron()