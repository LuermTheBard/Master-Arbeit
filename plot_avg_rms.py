from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from import_data import import_fits_data, find_prime_data_folder
from settings import All_LINES, All_LINE_GROUPS, DEFAULT_OUTPUT_DIR


def validate_fits_data(fits_data):
    """
    Validates the structure of the FITS-like data dictionary.

    Parameters:
    -----------
    fits_data : dict
        Dictionary where each value must be another dict containing 'x_axis' and 'data' keys.

    Raises:
    -------
    ValueError
        If the structure does not match the expected format.
    """

    if not isinstance(fits_data, dict):
        raise ValueError("The input fits_data must be a dictionary.")

    for key, value in fits_data.items():
        if not isinstance(value, dict):
            raise ValueError(f"The value for key '{key}' must be a dictionary.")
        if "x_axis" not in value or "data" not in value:
            raise ValueError(f"Key '{key}' must contain 'x_axis' and 'data'.")
        if not isinstance(value["x_axis"], (list, np.ndarray)):
            raise ValueError(f"'x_axis' in key '{key}' must be a list or numpy array.")
        if not isinstance(value["data"], (list, np.ndarray)):
            raise ValueError(f"'data' in key '{key}' must be a list or numpy array.")


def plot_avg_rms(fits_data, save_path=None, file_name=None, log_scale=False, xlim=None, ylim=None, no_description=False, ax=None, show_ylabel=True, scale_factor=None, shift_factor=None, line_length=0.75):
    """
    Extracts and plots AVG and RMS spectra from FITS-like data.
    Also saves the flux data and plot if a save path is provided.

    Parameters:
    -----------
    fits_data : dict
        Dictionary containing entries like '<name>_avg' and '<name>_rms',
        each with 'x_axis' and 'data' arrays.
    save_path : Path or None
        Directory path to save the plot and output .txt files. If None, only plots.
    log_scale : bool
        If True, use a logarithmic y-axis for the plot.

    Raises:
    -------
    ValueError
        If AVG or RMS data is missing or improperly formatted.
    """

    validate_fits_data(fits_data)

    avg_dict = None
    rms_dict = None

    galaxy_name = list(fits_data.keys())[0].split("_")[0]

    for key, item in fits_data.items():
        if "avg".casefold() in key.casefold():
            avg_dict = item
        elif "rms".casefold() in key.casefold():
            rms_dict = item

    if avg_dict is None:
        raise ValueError("Missing AVG data")
    if rms_dict is None:
        raise ValueError("Missing RMS data")

    wavelengths = avg_dict["x_axis"][0]
    avg_flux = avg_dict["data"][0]
    rms_flux = rms_dict["data"][0]

    label = (r"$F_\lambda \, [\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
             r"\mathrm{\AA}^{-1}]$")
    wavelengths_label = r"Rest Wavelength $[\mathrm{\AA}]$"
    avg_title = "AVG"
    rms_title = "RMS"

    if no_description:
        lines = dict()
        groups = dict()
    else:
        lines = All_LINES
        groups = All_LINE_GROUPS



    fig, ax, ylable = plot_avg_rms_spectra(
        wavelengths,
        avg_flux,
        rms_flux,
        wavelengths_label,
        label,
        avg_title,
        rms_title,
        galaxy_name,
        lines=lines,
        groups=groups,
        save_path=save_path / file_name if save_path and file_name else None,
        log_scale=log_scale,
        xlim=(3800, 8900) if xlim is None else xlim,
        ylim=(0, 13.999) if ylim is None else ylim, #ylim=(0, 100)
        ax=ax,
        show_ylable=show_ylabel,
        scale_factor=scale_factor,
        shift_factor=shift_factor,
        line_length=line_length,
    )

    if save_path:
        avg_data = np.column_stack((wavelengths, avg_flux))
        rms_data = np.column_stack((wavelengths, rms_flux))

        header_line = 'wavelength flux [ergs/s/cm2/A]'

        np.savetxt(save_path / f"{galaxy_name}_avg_flux.txt", avg_data,
                   delimiter=" ", header=header_line, comments='')
        np.savetxt(save_path / f"{galaxy_name}_rms_flux.txt", rms_data,
                   delimiter=" ", header=header_line, comments='')

    return fig, ax, ylable

def plot_avg_rms_spectra(
    x, y1, y2, xlabel, ylabel, title1, title2, super_title,
    lines, groups, save_path=None, log_scale=False, xlim=None, ylim=None,
    ax=None, show_ylable=True, scale_factor=None, shift_factor=None, line_length = 0.75
):
    """
    Plots the AVG and RMS spectra in one figure with custom scaling and line annotations.

    Parameters:
    -----------
    x : array-like
        Wavelength values.
    y1 : array-like
        Flux values for the AVG spectrum.
    y2 : array-like
        Flux values for the RMS spectrum.
    xlabel : str
        X-axis label.
    ylabel : str
        Base Y-axis label (will be scaled with power of 10).
    title1 : str
        Label for the AVG spectrum.
    title2 : str
        Label for the RMS spectrum.
    super_title : str
        Main plot title.
    lines : dict
        Dictionary of emission lines with their properties (position, label, etc.).
    groups : dict
        Dictionary of grouped line annotations.
    save_path : Path or None
        If set, the figure is saved as both .pdf and .png.
    log_scale : bool
        Whether to use a log y-axis.
    xlim : tuple or None
        Tuple of (xmin, xmax) for limiting x-axis.
    ylim : tuple or None
        Tuple of (ymin, ymax) for limiting y-axis.
    """

    new_ylabel, x_filtered, y1_filtered, y2_filtered = prepare_fit_data(x, xlim, y1, y2, ylabel)

    if scale_factor is None:
        scale_factor = 8
    else:
        scale_factor = scale_factor

    if shift_factor is None:

        shift_factor_1 = 2
        shift_factor_2 = -0.8
    else:
        shift_factor_1, shift_factor_2 = shift_factor


    y2_scaled = y2_filtered * scale_factor + shift_factor_2
    y1_filtered = y1_filtered + shift_factor_1

    # Falls kein ax übergeben wurde → neue Figure erzeugen
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure  # Hole Figure aus dem vorhandenen ax

    ax.plot(x_filtered, y1_filtered, label=f"{title1} (shifted by {shift_factor_1:.1f})", linestyle="-", color="blue")
    ax.plot(x_filtered, y2_scaled, label=f"{title2} (scaled by {scale_factor:.1f}, shifted by {shift_factor_2:.1f})", linestyle="-", color="orange")

    ax.set_xlabel(xlabel, fontsize=12)
    if show_ylable:
        ax.set_ylabel(f"{new_ylabel} + const.", fontsize=12)

    if log_scale:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=13, loc="upper right", frameon=False, markerfirst=False)

    ax.tick_params(axis='both', labelsize=16)



    # Add lines with labels
    for label, props in lines.items():
        pos = props["position"]
        text_vertical_shift = props.get("text_vertical_shift", 0.1)
        text_shift = props.get("text_shift", 0)
        show_no_tick_avg = props.get("show_no_tick_avg", False)
        show_no_tick_rms = props.get("show_no_tick_rms", False)
        tick_shift_avg = props.get("tick_shift_avg", 0.2)
        tick_shift_rms = props.get("tick_shift_rms", 0.2)

        slanted = props["slanted"]
        rotation_angle = 45 if slanted else 90

        # Bestimme die y-Position für die Linie
        idx = np.abs(x_filtered - pos).argmin()
        y_pos1 = y1_filtered[idx] + tick_shift_avg
        y_pos2 = y2_scaled[idx] + tick_shift_rms

        # Zeichne die Linie und füge das Label hinzu
        if not show_no_tick_avg:
            ax.plot([pos, pos], [y_pos1, y_pos1 + line_length], color="black", linewidth=0.5)
        if not show_no_tick_rms:
            ax.plot([pos, pos], [y_pos2, y_pos2 + line_length], color="black", linewidth=0.5)
        ax.text(pos + text_shift, y_pos1 + line_length + text_vertical_shift, label, fontsize=8, color="black", rotation=rotation_angle,
                ha="left" if slanted else "center",
                va="bottom")

    for group, data in groups.items():
        plot_line_group(ax, data["position"], x_filtered, y1_filtered, y2_scaled, group, line_length=line_length,
                        tick_vertical_shift_avg=data.get("tick_vertical_shift_avg",0.2), tick_vertical_shift_rms=data.get("tick_vertical_shift_rms", 0.2), show_in_avg=data.get("show_in_avg", True), show_in_rms=data.get("show_in_rms", False), all_lines=data.get("all_lines", False), text_vertical_shift=data.get("text_vertical_shift", 0.2), text_horizontal_shift=data.get("text_horizontal_shift", 0))

    # plt.title(super_title, fontsize=20)

    if save_path:
        fig.savefig(save_path, dpi=300)
        fig.savefig(f"{save_path}.png", dpi=300)
        print(f"Plot saved to {save_path}")

    return fig, ax, new_ylabel


def prepare_fit_data(x, xlim, y1, y2, ylabel, exponent_value=-15):
    y2_filtered = None
    if xlim:
        x = np.array(x) if isinstance(x, list) else x
        y1 = np.array(y1) if isinstance(y1, list) else y1
        if y2 is not None:
            y2 = np.array(y2) if isinstance(y2, list) else y2

        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_filtered = x[mask]
        y1_filtered = y1[mask]
        if y2 is not None:
            y2_filtered = y2[mask]
    else:
        x = np.array(x) if isinstance(x, list) else x
        y1 = np.array(y1) if isinstance(y1, list) else y1
        if y2 is not None:
            y2 = np.array(y2) if isinstance(y2, list) else y2
        x_filtered = x
        y1_filtered = y1
        if y2 is not None:
            y2_filtered = y2
    # Berechnung des Exponenten
    exponent = 10 ** exponent_value
    latex_exponent = f"10^{{{exponent_value}}}"
    # Aktualisierung des y-Labels
    ylabel_parts = ylabel.split("[")
    new_ylabel = ylabel_parts[0] + f"[{latex_exponent} " + ylabel_parts[1]
    y1_filtered = y1_filtered / exponent
    if y2 is not None:
        y2_filtered = y2_filtered / exponent
    return new_ylabel, x_filtered, y1_filtered, y2_filtered


def plot_line_group(
    ax_obj, positions, x, y1_filtered, y2_scaled, group,
    line_length=0.5, tick_vertical_shift_avg=0.5, tick_vertical_shift_rms=0.5,
    all_lines=False, show_in_avg=True, show_in_rms=False, text_vertical_shift=0.1, text_horizontal_shift=0):
    """
    Draws vertical line annotations for a group of emission lines on the plot.

    Parameters:
    -----------
    ax_obj : matplotlib.axes.Axes
        The plot axis to draw on.
    positions : list of float
        Wavelength positions of the group lines.
    x : np.ndarray
        Wavelength array used to align annotations with spectrum data.
    y1_filtered : np.ndarray
        Shifted AVG spectrum for vertical positioning.
    y2_scaled : np.ndarray
        Scaled RMS spectrum for optional RMS annotations.
    group : str
        Name of the line group (e.g., "Balmer", "Helium").
    line_length : float, optional
        Height of the vertical tick marks.
    tick_vertical_shift_avg : float, optional
        Vertical offset for AVG ticks.
    tick_vertical_shift_rms : float, optional
        Vertical offset for RMS ticks.
    all_lines : bool, optional
        If True, draw a tick for every position in `positions`.
    show_in_rms : bool, optional
        If True, also draw ticks for RMS spectrum.

    Returns:
    --------
    None
    """

    min_pos = min(positions)
    max_pos = max(positions)

    # Finde den nächstgelegenen Y-Wert zu pos
    idx = np.abs(x - min_pos).argmin()
    spectrum_avg = y1_filtered[idx]
    spectrum_rms = y2_scaled[idx]

    line_length = line_length + 0.015

    # Linienstart und -ende berechnen
    line_ymin_avg = spectrum_avg + tick_vertical_shift_avg
    line_ymax_avg = line_ymin_avg + line_length

    line_ymin_rms = spectrum_rms + tick_vertical_shift_rms
    line_ymax_rms = line_ymin_rms + line_length
    if show_in_avg is True:
        if all_lines:

            for pos in positions:
                # Vertikale Linien zeichnen
                ax_obj.vlines(x=pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=0.5)

        else:
            ax_obj.vlines(x=min_pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=0.5)
            ax_obj.vlines(x=max_pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=0.5)

    if show_in_rms:
        for pos in positions:
            ax_obj.vlines(x=pos, ymin=line_ymin_rms, ymax=line_ymax_rms, color='black', linewidth=0.5)

    ax_obj.text(x=((max_pos - min_pos) / 2) + min_pos + text_horizontal_shift,
                y=line_ymax_avg + text_vertical_shift,
                s=group,
                ha='center',
                fontsize=8,
                rotation=90,
                va="bottom")
    # Horizontale Verbindungslinie
    if show_in_avg is True:
        ax_obj.hlines(y=line_ymax_avg, xmin=min_pos, xmax=max_pos, color='black', linewidth=0.5)

# methods to run

def plot_avg_rms_spec(input_dir=None, file_name=None,
                      output_dir=DEFAULT_OUTPUT_DIR,
                      no_description=False, xlim=None, ylim=None, ax=None, show_ylabel=True, scale_factor=None, shift_factor=None, line_length=0.75):
    avg_rms_spec_dir = output_dir / "avg_rms_spec"
    avg_rms_spec_dir.mkdir(parents=True, exist_ok=True)

    fits_data = import_fits_data(input_dir)
    return plot_avg_rms(fits_data, save_path=avg_rms_spec_dir,
                        file_name=file_name,
                        no_description=no_description,
                        ax=ax,
                        xlim=xlim,
                        ylim=ylim,
                        show_ylabel=show_ylabel,
                        scale_factor=scale_factor,
                        shift_factor=shift_factor,
                        line_length=line_length

                        )  # NEU



def plot_spectra(fits_data, spec_file=None, xlim=None, ylim=None, ax=None, show_ylabel=True):
    created_fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created_fig = fig
    else:
        fig = ax.figure

    ylabel = (r"$F_\lambda \, [\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
              r"\mathrm{\AA}^{-1}]$")
    xlabel = r"Rest Wavelength $[\mathrm{\AA}]$"

    for key, item in fits_data.items():
        new_ylabel, x_filtered, y1_filtered, _ = prepare_fit_data(
            item["x_axis"], xlim, item["data"], None, ylabel
        )
        ax.plot(x_filtered, y1_filtered, label=f"{key}", linestyle="-", color="black", linewidth=0.3)

    ax.set_xlabel(xlabel, fontsize=14)
    if show_ylabel:
        ax.set_ylabel(f"{new_ylabel} + const.", fontsize=14)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    ax.tick_params(axis='both', labelsize=14)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if spec_file is not None:
        fig.savefig(spec_file, dpi=300)
        fig.savefig(f"{spec_file}.png", dpi=300)
        print(f"Plot saved to {spec_file}")

    if created_fig is not None:
        plt.show()

    return fig, ax, new_ylabel



def plot_calibrated_and_uncalibrated_spectra_together(output_dir=DEFAULT_OUTPUT_DIR):
    uncalibrated_fits_data = import_fits_data(Path("fits") / "uncalibrated_fits")
    calibrated_fits_data   = import_fits_data(Path("fits") / "intercalibrated_fits")

    validate_fits_data(uncalibrated_fits_data)
    validate_fits_data(calibrated_fits_data)

    xlim = (3800, 8900)
    ylim = (0, 13.999)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6),
        gridspec_kw={"hspace": 0}
    )

    # Oberer Plot: uncalibrated
    _, _, ylabel = plot_spectra(uncalibrated_fits_data, xlim=xlim, ylim=ylim,
                                ax=ax1, show_ylabel=False)
    ax1.text(0.95, 0.95, "Original", transform=ax1.transAxes,
             ha="right", va="top", fontsize=18)

    # Unterer Plot: calibrated
    plot_spectra(calibrated_fits_data, xlim=xlim, ylim=ylim,
                 ax=ax2, show_ylabel=False)
    ax2.text(0.95, 0.95, "Intercalibrated", transform=ax2.transAxes,
             ha="right", va="top", fontsize=18)

    # X-Achse nur unten beschriften
    ax2.set_xlabel(r"Rest Wavelength $[\mathrm{\AA}]$", fontsize=18)
    for ax in (ax1, ax2):
        ax.label_outer()

    # gemeinsames Y-Label
    fig.text(0.02, 0.5, f"{ylabel}",
             va="center", rotation="vertical", fontsize=18)

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig(output_dir / "avg_rms_spec" / "comparison_spectra.pdf", dpi=300)
    print(f"Plot saved to {output_dir / 'avg_rms_spec' / 'comparison_spectra.pdf'}")
    plt.show()
    return fig, (ax1, ax2)


def plot_avg_rms_together(output_dir=DEFAULT_OUTPUT_DIR):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6),
        gridspec_kw={"hspace": 0}  # Subplots dicht übereinander
    )

    xlim=(3800, 8900)
    ylim = (0, 7.999)

    data_path = find_prime_data_folder()

    uncalibrate_avg_rms_data_path = data_path / "fits"/ "uncalibrated_AVG_RMS"
    calibrate_avg_rms_data_path = data_path / "fits"

    # Oberer Plot
    _,_,ylabel = plot_avg_rms_spec(input_dir=uncalibrate_avg_rms_data_path, xlim=xlim, ylim=ylim, no_description=True, ax=ax1, show_ylabel=False)
    ax1.text(0.3, 0.95, "Original",
             transform=ax1.transAxes, ha="left", va="top", fontsize=18)

    # Unterer Plot
    plot_avg_rms_spec(input_dir=calibrate_avg_rms_data_path, xlim=xlim, ylim=ylim, no_description=True, ax=ax2, show_ylabel=False)
    ax2.text(0.3, 0.95, "Intercalibrated",
             transform=ax2.transAxes, ha="left", va="top", fontsize=18)

    # Gemeinsame X-Achse nur unten beschriften
    ax2.set_xlabel("Rest Wavelength [Å]", fontsize=18)
    for ax in (ax1, ax2):
        ax.label_outer()

    # Zentrale Y-Achsenbeschriftung (für beide Plots gemeinsam)
    fig.text(0.02, 0.5,
             f"{ylabel} + const.",
             va="center", rotation="vertical", fontsize=18)

    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Platz für Y-Label lassen
    fig.savefig(output_dir / "avg_rms_spec" / "comparison_avg_rms.pdf", dpi=300)
    print(f"Plot saved to {output_dir / 'avg_rms_spec' / 'comparison_avg_rms.pdf'}")
    plt.show()
    return fig, (ax1, ax2)


def measure_line_flux(
    fits_data: dict,
    line_window: tuple,
    cont_windows: tuple,
    degree: int = 1,
    flux_key: str = "data",
    lam_key: str = "x_axis",
    sigma_key: str = "sigma",
):
    """
    Integrierter Linienfluss F_line aus einem AVG-Spektrum.

    Parameters
    ----------
    fits_data : dict
        Muss Arrays unter lam_key (Å), flux_key (z.B. erg s^-1 cm^-2 Å^-1) enthalten.
        Optional: sigma_key (gleiche Einheiten wie flux, pro Pixel).
    line_window : (lam_min, lam_max)
        Integrationsfenster der Linie in Angström (observed oder rest, aber konsistent).
    cont_windows : ((c1_min, c1_max), (c2_min, c2_max))
        Zwei linienfreie Fenster links/rechts für den Kontinuumfit.
    degree : int, default 1
        Grad des Polynoms fürs Kontinuum (1 = linear; 0 = konstant).
    flux_key, lam_key, sigma_key : str
        Dict-Schlüssel.

    Returns
    -------
    F_line : float
        Integrierter Linienfluss (Einheiten: flux * Å).
    F_err : float or np.nan
        1σ-Unsicherheit (inkl. grober Kontinuumskomponente), np.nan wenn keine Fehler.
    EW : float or np.nan
        Äquivalentbreite in Å (Emission > 0), np.nan wenn Kontinuum ≈ 0.
    meta : dict
        Zusatzinfos: Kontinuumkoeffs, Kontinuumfunktion (callable), Masken, usw.
    """
    lam = np.asarray(fits_data[lam_key], dtype=float)
    flux = np.asarray(fits_data[flux_key], dtype=float)
    sigma = None
    if sigma_key in fits_data and fits_data[sigma_key] is not None:
        sigma = np.asarray(fits_data[sigma_key], dtype=float)

    # sortieren & gültige Werte
    m_good = np.isfinite(lam) & np.isfinite(flux)
    if sigma is not None:
        m_good &= np.isfinite(sigma)
    lam, flux = lam[m_good], flux[m_good]
    sigma = (sigma[m_good] if sigma is not None else None)

    idx = np.argsort(lam)
    lam, flux = lam[idx], flux[idx]
    if sigma is not None:
        sigma = sigma[idx]

    # Fenster-Masken
    (lmin, lmax) = line_window
    (c1min, c1max), (c2min, c2max) = cont_windows

    m_cont = ((lam >= c1min) & (lam <= c1max)) | ((lam >= c2min) & (lam <= c2max))
    m_line = (lam >= lmin) & (lam <= lmax)

    lam_cont, flux_cont = lam[m_cont], flux[m_cont]
    lam_line, flux_line = lam[m_line], flux[m_line]
    if lam_line.size < 3 or lam_cont.size < (degree + 2):
        raise ValueError("Zu wenige Punkte im Linien- oder Kontinuumsfenster.")

    # Gewichte für Polyfit (np.polyfit erwartet 'w' ~ 1/σ)
    w = None
    if sigma is not None:
        w = 1.0 / np.clip(sigma[m_cont], 1e-300, np.inf)

    # Kontinuumfit (Polynom in λ)
    coeffs = np.polyfit(lam_cont, flux_cont, deg=degree, w=w)
    cont_poly = np.poly1d(coeffs)

    # Residuum im Linienfenster
    cont_line = cont_poly(lam_line)
    resid = flux_line - cont_line

    # Trapezintegration mit expliziten Gewichten (für Fehlerfortpflanzung)
    dlam = np.diff(lam_line)
    w_trap = np.empty_like(lam_line)
    w_trap[0] = dlam[0] / 2.0
    w_trap[1:-1] = (dlam[:-1] + dlam[1:]) / 2.0
    w_trap[-1] = dlam[-1] / 2.0

    F_line = float(np.sum(resid * w_trap))

    # Fehler der Integration
    if sigma is not None:
        sigma_line = sigma[m_line]

        # Varianz aus Pixelrauschen (ungekoppelt angenommen)
        var_pix = np.sum((sigma_line * w_trap) ** 2)

        # Grobe Kontinuum-Komponente: Streuung der Kontinuumresiduen × Linienbreite
        cont_resid = flux_cont - cont_poly(lam_cont)
        # robust: IQR/1.349 ~ std
        q25, q75 = np.percentile(cont_resid, [25, 75])
        cont_std = (q75 - q25) / 1.349 if np.isfinite(q25 + q75) else np.std(cont_resid, ddof=1)
        delta_lambda = lam_line[-1] - lam_line[0]
        var_cont = (cont_std * delta_lambda) ** 2

        F_err = float(np.sqrt(var_pix + var_cont))
    else:
        F_err = np.nan

    # Äquivalentbreite (Kontinuum am Linienzentrum)
    lam0 = 0.5 * (lmin + lmax)
    fcont0 = float(cont_poly(lam0))
    EW = (F_line / fcont0) if np.isfinite(fcont0) and fcont0 != 0 else np.nan

    meta = dict(
        coeffs=coeffs,
        continuum_fn=cont_poly,
        masks=dict(line=m_line, cont=m_cont),
        line_window=line_window,
        cont_windows=cont_windows,
        degree=degree,
        lam0=lam0,
        fcont0=fcont0,
    )
    return F_line, F_err, EW, meta


# -----------------------------
# Minimales Anwendungsbeispiel:
# -----------------------------
# fits_data = {
#     "lambda": lam_Angstrom,                 # 1D ndarray
#     "flux": flux_flam,                      # erg s^-1 cm^-2 Å^-1
#     "sigma": flux_err_flam (optional),      # gleiche Einheiten wie flux
# }
#
# F, Ferr, EW, meta = measure_line_flux(
#     fits_data,
#     line_window=(4845.0, 5010.0),                   # z.B. Hβ (inkl. Flügel, ohne [O III])
#     cont_windows=((4815.0, 4840.0), (5100.0, 5130.0)),
#     degree=1,
# )
# print(f"F_line = {F:.3e} ± {Ferr:.3e}  (EW = {EW:.2f} Å)")

def get_line_flux(line_window: tuple,cont_windows: tuple):
    data_path = find_prime_data_folder()
    avg_fits_data = import_fits_data(data_path / "fits")['NGC4593_avg.fits']

    line_window = (4995.66, 5021.75)
    cont_windows = ((4762, 4774),(5085, 5112))

    F_line, F_err, EW, meta= measure_line_flux(avg_fits_data, line_window, cont_windows)

    print(f"F_line = {F_line:.3e} ± {F_err:.3e}  (EW = {EW:.2f} Å)")



# get_line_flux(None, None)

#plot_avg_rms_spec(file_name='avg_rms_spec.pdf')
#plot_avg_rms_spec(input_dir=Path("fits") / "uncalibrated_AVG_RMS", file_name='UV_uncalibrated_AVG_RMS.pdf',xlim=(1130, 1800), ylim=(3, 70), scale_factor=5, shift_factor=(10, 0), line_length=3)
plot_avg_rms_together()
plot_calibrated_and_uncalibrated_spectra_together()