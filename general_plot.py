import datetime
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from import_data import load_centroid_data_by_reference, find_prime_data_folder, import_1d_lightcurve_data
from plot_utils import format_label, ensure_output_dir
from settings import BASE_MJD, DEFAULT_OUTPUT_DIR, IONIZATION_POTENTIAL, FWHM_RMS

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

    date = mjd_to_date(mjd)
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
    """
    Removes completely empty subplot rows from a figure and sets x-axis labels
    and tick formatting for the lowest visible row.

    Parameters:
    -----------
    axes : numpy.ndarray
        2D array of Matplotlib Axes objects.
    fig : matplotlib.figure.Figure
        The figure containing the subplots.
    x_label : str or tuple of str
        X-axis label(s). If a tuple is provided, each column gets a different label.
    line_profile : bool, optional
        Whether the plots are line profiles. Affects tick formatting. Default is False.
    """

    n_rows = axes.shape[0]
    for row in range(n_rows):
        if all(not is_valid_axis(axes[row, col], fig) for col in range(2)):
            for col in range(2):
                fig.delaxes(axes[row, col])

    # Ermittlung der untersten verbleibenden Reihe
    remaining_rows = [row for row in range(n_rows) if any(is_valid_axis(axes[row, col], fig) for col in range(2))]
    if remaining_rows:
        lowest_row = max(remaining_rows)

        for row in remaining_rows:
            for col in range(2):
                if is_valid_axis(axes[row, col], fig):
                    if not line_profile and not isinstance(x_label, tuple):
                        axes[row, col].xaxis.set_major_locator(MultipleLocator(5))

                    axes[row, col].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

                    if row == lowest_row:
                        if isinstance(x_label, tuple):
                            label = x_label[col]
                            axes[row, col].set_xlabel(label, fontsize=12)
                            if col == 0:
                                axes[row, col].xaxis.set_major_locator(MultipleLocator(5))
                            elif col == 1:
                                axes[row, col].xaxis.set_major_locator(MultipleLocator(2))
                        else:
                            axes[row, col].set_xlabel(x_label, fontsize=12)

                        axes[row, col].tick_params(axis='x', which='both', direction='out', labelbottom=True)


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


def finalize_figure(fig, axes, title, group_index, save_only, output_dir, x_label, compare_cont=None, line_profile=False, file_name=None):
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

    if title:
        if group_index > 0:
            fig.suptitle(f'{title} - Group {group_index + 1}', fontsize=14, y=0.95)
        else:
            fig.suptitle(f'{title}', fontsize=14)

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



def plot_ion_pot_FWHM_against_lag(output_dir=DEFAULT_OUTPUT_DIR):
    output_file_dir = output_dir / "plot_against_lag"
    ensure_output_dir(output_file_dir)

    time_lag_data = load_centroid_data_by_reference()
    ionization = IONIZATION_POTENTIAL
    fwhm_rms = FWHM_RMS

    # Sammle Daten für Fit
    x_ip_list, y_ip_list = [], []
    x_fwhm_list, y_fwhm_list = [], []

    # ---------- 1. Plot: Ionisationspotenzial gegen Zeitverzögerung ----------
    for line, ionisation_potential in ionization.items():
        try:

            labels = format_label(line, as_latex=False).split(" ")[0]
            if "$" in labels:
                labels = labels + "$"
            y = time_lag_data["UVW2"][line]["tau_cent"]
            yerr_low = time_lag_data["UVW2"][line]["tau_cent_err_low"]
            yerr_high = time_lag_data["UVW2"][line]["tau_cent_err_high"]
            yerr = np.vstack((yerr_low, yerr_high))

            x = ionisation_potential + np.random.uniform(-0.2, 0.2)  # Jitter
            x_ip_list.append(ionisation_potential)
            y_ip_list.append(y)

            plt.errorbar(x, y, yerr=yerr,
                         label=labels,
                         fmt='.:', capsize=3, markersize=4)
        except KeyError:
            print(f"No values found for {line}")

    # Fit
    x_ip_array = np.array(x_ip_list)
    y_ip_array = np.array(y_ip_list)
    coeffs_ip = np.polyfit(x_ip_array, y_ip_array, 1)
    x_fit = np.linspace(min(x_ip_array), max(x_ip_array), 100)
    y_fit = np.polyval(coeffs_ip, x_fit)
    plt.plot(x_fit, y_fit, 'k--')

    plt.xlabel("Ionisation Potential [eV]")
    plt.ylabel("Time Lag τ [days]")
    plt.legend(loc='upper right')
    plt.savefig(output_file_dir / "Ionisation_potential_with_fit.pdf")
    plt.show()
    plt.close()

    # ---------- 2. Plot: FWHM gegen Zeitverzögerung ----------

    for line in time_lag_data:
        try:

            labels = format_label(line, as_latex=False).split(" ")[0]
            if "$" in labels:
                labels = labels + "$"
            y = time_lag_data[line]["tau_cent"]
            yerr_low = time_lag_data[line]["tau_cent_err_low"]
            yerr_high = time_lag_data[line]["tau_cent_err_high"]
            yerr = np.vstack((yerr_low, yerr_high))
            x = fwhm_rms[line]

            x_fwhm_list.append(x)
            y_fwhm_list.append(y)

            plt.errorbar(x, y, yerr=yerr,
                         fmt='o', label=labels,
                         capsize=3, markersize=5)
        except KeyError:
            print(f"Missing data for line: {line}")

    # Fit
    x_fwhm_array = np.array(x_fwhm_list)
    y_fwhm_array = np.array(y_fwhm_list)
    coeffs_fwhm = np.polyfit(x_fwhm_array, y_fwhm_array, 1)
    x_fit = np.linspace(min(x_fwhm_array), max(x_fwhm_array), 100)
    y_fit = np.polyval(coeffs_fwhm, x_fit)
    plt.plot(x_fit, y_fit, 'k--')

    plt.xlabel("FWHM RMS [km/s]")
    plt.ylabel("Time Lag τ [days]")

    plt.legend(loc='upper right')
    plt.savefig(output_file_dir / "Time_lag_vs_FWHM_with_fit.pdf")
    plt.show()
    plt.close()
    # ---------- 3. Vergleichsplot: normierte Fits ----------

    # Normiere IP-Fit
    x_ip_norm = (x_ip_array - x_ip_array.min()) / (x_ip_array.max() - x_ip_array.min())
    coeffs_ip_norm = np.polyfit(x_ip_norm, y_ip_array, 1)
    x_fit_ip = np.linspace(0, 1, 100)
    y_fit_ip = np.polyval(coeffs_ip_norm, x_fit_ip)
    plt.plot(x_fit_ip, y_fit_ip, label=f"Ionisation Fit (m={coeffs_ip_norm[0]:.2f})", linestyle='--')

    # Normiere FWHM-Fit
    x_fwhm_norm = (x_fwhm_array - x_fwhm_array.min()) / (x_fwhm_array.max() - x_fwhm_array.min())
    coeffs_fwhm_norm = np.polyfit(x_fwhm_norm, y_fwhm_array, 1)
    x_fit_fwhm = np.linspace(0, 1, 100)
    y_fit_fwhm = np.polyval(coeffs_fwhm_norm, x_fit_fwhm)
    plt.plot(x_fit_fwhm, y_fit_fwhm, label=f"FWHM Fit (m={coeffs_fwhm_norm[0]:.2f})", linestyle='-.')

    plt.xlabel("Normierter x-Wert")
    plt.ylabel("Time Lag τ")
    plt.title("Vergleich der normierten Steigungen")

    plt.legend(loc='upper right')
    plt.savefig(output_file_dir / "Fit_comparison_normalized.pdf")
    plt.show()


def plot_ccfs_lags_against_angstron(output_dir=DEFAULT_OUTPUT_DIR):
    output_file_dir = output_dir / "plot_against_angstron"
    ensure_output_dir(output_file_dir)
    data_path = Path(find_prime_data_folder()) / "divers"

    data_file = str(data_path / "ccf_and_lag_to_angstrom.txt")
    with open(data_file, "r") as f:
        header = f.readline().strip().split()
        data = np.loadtxt(data_file, skiprows=1, dtype=str).T
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

    # Konvertiere direkt zu float
    data_dict = {key: np.array(values, dtype=float) for key, values in zip(header, data)}


    plt.errorbar(data_dict["Angstron"], data_dict["ccf"], yerr=0.01, fmt='.:', label="CCF")
    plt.vlines(8460, 0,1.5, linestyle="-.", color="black", label="Center OI")
    plt.xlabel(r"Wavelength [$\AA$]")
    plt.ylabel("CCF")
    plt.ylim(0.5, 0.9)
    plt.legend()
    plt.savefig(output_file_dir / "CCF_against_Anstron.pdf")

    plt.show()

    plt.errorbar(data_dict["Angstron"], data_dict["tau"], yerr=0.1, fmt='.:', label=r"$\tau$")
    plt.vlines(8460, -1, 1, linestyle="-.", color="black", label="Center OI")
    plt.xlabel(r"Wavelength [$\AA$]")
    plt.ylabel(r"Time Lag [$\tau$]")
    plt.ylim(-0.6, 0.6)
    plt.legend()
    plt.savefig(output_file_dir / "lags_against_Anstron.pdf")
    plt.show()


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
    def compute_f_var(lightcurve_type, lightcurve_name):
        data = lightcurver_data[campaign][lightcurve_type][lightcurve_name]
        flux = np.array(data['fluxes [ergs/s/cm2/A]'])
        flux_err = np.array(data['fluxerrs [ergs/s/cm2/A]'])

        N = len(flux)
        mean_flux = np.mean(flux)
        sigma2 = np.var(flux, ddof=1)
        delta2 = np.mean(flux_err**2)
        excess_var = sigma2 - delta2
        if excess_var < 0:
            F_var = 0
        else:
            F_var = np.sqrt(excess_var) / mean_flux

        results.append({
            "type": lightcurve_type,
            "name": lightcurve_name,
            "F_var": F_var
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

    # Optional: Ergebnisse speichern
    output_file_dir = output_dir / "f_var_data"
    ensure_output_dir(output_file_dir)

    # Ergebnisse als einfache Textdatei oder CSV speichern
    output_file = output_file_dir / f"{campaign}_f_var.txt"
    with open(output_file, "w") as f:
        for r in results:
            f.write(f"{r['type'][:-1]} {r['name']}: F_var = {r['F_var']:.6f}\n")

    print(f"Saved F_var results to: {output_file}")

    return results





get_f_var(line_names=["HAlpha", "HBeta", "HGamma", "HDelta", "OI8446"])

# plot_ion_pot_FWHM_against_lag()
# plot_ccfs_lags_against_angstron()