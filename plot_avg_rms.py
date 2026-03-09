from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from import_data import import_fits_data, find_prime_data_folder
from settings import All_LINES, All_LINE_GROUPS, DEFAULT_OUTPUT_DIR


matplotlib.use('Qt5Agg')

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


def plot_avg_rms(fits_data, save_path=None, file_name=None, log_scale=False, xlim=None, ylim=None, no_description=False, ax=None, show_ylabel=True, scale_factor=None, shift_factor=None, line_length=0.75, figsize=None, show_only_label=False, selected_broad_lines=None):
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
    avg_title = "mean"
    rms_title = "rms"

    if no_description:
        lines = dict()
        groups = dict()
    elif selected_broad_lines is not None:
        lines = selected_broad_lines[0]
        groups = selected_broad_lines[1]
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
        figsize=figsize,
        show_only_label=show_only_label
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
    ax=None, show_ylable=True, scale_factor=None, shift_factor=None, line_length = 0.75, figsize=None, show_only_label=False
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
        if figsize is None:
            figure_size = (10, 6)
        else:
            figure_size = figsize
        fig, ax = plt.subplots(figsize=figure_size)
        fig.tight_layout(rect=[0.04, 0.04, 1, 1])
    else:
        fig = ax.figure  # Hole Figure aus dem vorhandenen ax

    if show_only_label:
        label1 = f"{title1}"
        label2 = f"{title2} (scaled by {scale_factor:.1f}, shifted by {shift_factor_2:.1f})"
    else:
        label1 = f"{title1} (shifted by {shift_factor_1:.1f})"
        label2 = f"{title2} (scaled by {scale_factor:.1f}, shifted by {shift_factor_2:.1f})"

    ax.plot(x_filtered, y1_filtered, label=label1, linestyle="-", color="blue")
    ax.plot(x_filtered, y2_scaled, label=label2, linestyle="-", color="orange")

    ax.set_xlabel(xlabel, fontsize=12)
    if show_ylable:
        if show_only_label:
            ax.set_ylabel(new_ylabel, fontsize=12)
        else:
            ax.set_ylabel(f"{new_ylabel} + const.", fontsize=12)

    if log_scale:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=10, loc="upper right", frameon=False, markerfirst=False)

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
        ax.text(pos + text_shift, y_pos1 + line_length + text_vertical_shift, label, fontsize=10, color="black", rotation=rotation_angle,
                ha="left" if slanted else "center",
                va="bottom")

    for group, data in groups.items():
        plot_line_group(ax, data["position"], x_filtered, y1_filtered, y2_scaled, group, line_length=line_length,
                        tick_vertical_shift_avg=data.get("tick_vertical_shift_avg",0.2), tick_vertical_shift_rms=data.get("tick_vertical_shift_rms", 0.2), show_in_avg=data.get("show_in_avg", True), show_in_rms=data.get("show_in_rms", False), all_lines=data.get("all_lines", False), text_vertical_shift=data.get("text_vertical_shift", 0.2), text_horizontal_shift=data.get("text_horizontal_shift", 0))

    # plt.title(super_title, fontsize=20)

    if save_path:
        fig.savefig(save_path)
        fig.savefig(f"{save_path}.png")
        print(f"Plot saved to {save_path}")

    # plt.show()

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
                fontsize=10,
                rotation=90,
                va="bottom")
    # Horizontale Verbindungslinie
    if show_in_avg is True:
        ax_obj.hlines(y=line_ymax_avg, xmin=min_pos, xmax=max_pos, color='black', linewidth=0.5)

# methods to run

def plot_avg_rms_spec(input_dir=None, file_name=None,
                      output_dir=DEFAULT_OUTPUT_DIR,
                      no_description=False, xlim=None, ylim=None, ax=None, show_ylabel=True, scale_factor=None, shift_factor=None, line_length=0.75, figsize=None, show_only_label=False, selected_broad_lines=None):
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
                        line_length=line_length,
                        figsize=figsize,
                        show_only_label=show_only_label,
                        selected_broad_lines=selected_broad_lines

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



def plot_calibrated_and_uncalibrated_spectra_separately(xlim=None, ylim=None, file_name=None, output_dir=DEFAULT_OUTPUT_DIR):
    uncalibrated_fits_data = import_fits_data(Path("fits") / "uncalibrated_fits")
    calibrated_fits_data   = import_fits_data(Path("fits") / "intercalibrated_fits")

    validate_fits_data(uncalibrated_fits_data)
    validate_fits_data(calibrated_fits_data)

    if xlim is None:
        xlim = (3800, 8900)
    if ylim is None:
        ylim = (0, 13.999)

    if file_name is None:
        file_name = "optical_NIR"

    datasets = [
        (uncalibrated_fits_data, f"comparison_spectra_{file_name}_original.pdf"),
        (calibrated_fits_data, f"comparison_spectra_{file_name}_intercalibrated.pdf"),
    ]

    figs = []
    for fits_data, filename in datasets:
        fig, ax = plt.subplots(figsize=(10, 6))

        _, _, ylabel = plot_spectra(fits_data, xlim=xlim, ylim=ylim,
                                    ax=ax, show_ylabel=False)
        ax.set_xlabel(r"Rest Wavelength $[\mathrm{\AA}]$", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)

        plt.tight_layout()
        out_path = output_dir / "avg_rms_spec" / filename
        fig.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")
        plt.show()
        figs.append((fig, ax))

    return figs



def plot_avg_rms_together(output_dir=DEFAULT_OUTPUT_DIR):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6),
        gridspec_kw={"hspace": 0}  # Subplots dicht übereinander
    )

    xlim=(3800, 8900)
    ylim = (-2, 5.999)

    data_path = find_prime_data_folder()

    uncalibrate_avg_rms_data_path = data_path / "fits"/ "uncalibrated_AVG_RMS"
    calibrate_avg_rms_data_path = data_path / "fits"

    # Oberer Plot
    _,_,ylabel = plot_avg_rms_spec(input_dir=uncalibrate_avg_rms_data_path, xlim=xlim, ylim=ylim, no_description=True, ax=ax1, show_ylabel=False, scale_factor=8, shift_factor=(0, -3), show_only_label=True)
    ax1.text(0.3, 0.95, "Original",
             transform=ax1.transAxes, ha="left", va="top", fontsize=18)

    # Unterer Plot
    plot_avg_rms_spec(input_dir=calibrate_avg_rms_data_path, xlim=xlim, ylim=ylim, no_description=True, ax=ax2, show_ylabel=False, scale_factor=8, shift_factor=(0, -3), show_only_label=True)
    ax2.text(0.3, 0.95, "Intercalibrated",
             transform=ax2.transAxes, ha="left", va="top", fontsize=18)

    # Gemeinsame X-Achse nur unten beschriften
    ax2.set_xlabel("Rest Wavelength [Å]", fontsize=18)
    for ax in (ax1, ax2):
        ax.label_outer()

    # Zentrale Y-Achsenbeschriftung (für beide Plots gemeinsam)
    fig.text(0.02, 0.5,
             f"{ylabel}",
             va="center", rotation="vertical", fontsize=18)

    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Platz für Y-Label lassen
    fig.savefig(output_dir / "avg_rms_spec" / "comparison_avg_rms.pdf", dpi=300)
    print(f"Plot saved to {output_dir / 'avg_rms_spec' / 'comparison_avg_rms.pdf'}")
    plt.show()
    return fig, (ax1, ax2)



def _robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return 1.4826 * mad  # MAD -> std
    q25, q75 = np.percentile(x, [25, 75])
    if np.isfinite(q25 + q75) and (q75 - q25) > 0:
        return (q75 - q25) / 1.349  # IQR -> std
    return np.std(x, ddof=1)



def measure_line_flux(
    fits_data: dict,
    line_window: tuple,
    cont_windows: tuple,
    degree: int = 1,
    flux_key: str = "data",
    lam_key: str = "x_axis",
    sigma_key: str = "sigma",
):
    lam = np.asarray(fits_data[lam_key], dtype=float)
    flux = np.asarray(fits_data[flux_key], dtype=float)

    sigma = None
    if sigma_key in fits_data and fits_data[sigma_key] is not None:
        sigma = np.asarray(fits_data[sigma_key], dtype=float)

    # gültige Werte (sigma darf fehlen -> NICHT hier rausfiltern)
    m_good = np.isfinite(lam) & np.isfinite(flux)
    lam, flux = lam[m_good], flux[m_good]
    sigma = (sigma[m_good] if sigma is not None else None)

    idx = np.argsort(lam)
    lam, flux = lam[idx], flux[idx]
    if sigma is not None:
        sigma = sigma[idx]

    (lmin, lmax) = line_window
    (c1min, c1max), (c2min, c2max) = cont_windows

    m_cont = ((lam >= c1min) & (lam <= c1max)) | ((lam >= c2min) & (lam <= c2max))
    m_line = (lam >= lmin) & (lam <= lmax)

    lam_cont, flux_cont = lam[m_cont], flux[m_cont]
    lam_line, flux_line = lam[m_line], flux[m_line]
    if lam_line.size < 3 or lam_cont.size < (degree + 2):
        raise ValueError("Zu wenige Punkte im Linien- oder Kontinuumsfenster.")

    # Gewichte: nur wo sigma finite & >0
    w = None
    if sigma is not None:
        sig_cont = sigma[m_cont]
        m_sig_ok = np.isfinite(sig_cont) & (sig_cont > 0)
        if np.any(m_sig_ok):
            w = np.zeros_like(sig_cont, dtype=float)
            w[m_sig_ok] = 1.0 / np.clip(sig_cont[m_sig_ok], 1e-300, np.inf)
        else:
            w = None  # fallback unweighted

    # Kontinuumfit
    coeffs = np.polyfit(lam_cont, flux_cont, deg=degree, w=w)
    cont_poly = np.poly1d(coeffs)

    cont_line = cont_poly(lam_line)
    resid = flux_line - cont_line

    # Trapezgewichte
    dlam = np.diff(lam_line)
    w_trap = np.empty_like(lam_line)
    w_trap[0] = dlam[0] / 2.0
    w_trap[1:-1] = (dlam[:-1] + dlam[1:]) / 2.0
    w_trap[-1] = dlam[-1] / 2.0

    F_line = float(np.sum(resid * w_trap))

    # --- NEU: sigma aus Kontinuum schätzen & fehlende sigma auffüllen ---
    cont_resid = flux_cont - cont_poly(lam_cont)
    sigma0 = _robust_std(cont_resid)  # Pixelrauschen-Schätzer
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = np.nan

    if sigma is None:
        # keine sigma geliefert -> überall sigma0 nutzen (wenn möglich)
        sigma_use = None if not np.isfinite(sigma0) else np.full_like(flux, sigma0, dtype=float)
    else:
        sigma_use = sigma.copy()
        # fehlende/ungültige sigma auffüllen
        m_bad = (~np.isfinite(sigma_use)) | (sigma_use <= 0)
        if np.any(m_bad) and np.isfinite(sigma0):
            sigma_use[m_bad] = sigma0

    # Fehler der Integration
    if sigma_use is not None:
        sigma_line = sigma_use[m_line]

        # Varianz aus Pixelrauschen
        var_pix = np.sum((sigma_line * w_trap) ** 2)

        # Kontinuum-Komponente (wie bei dir)
        delta_lambda = lam_line[-1] - lam_line[0]
        cont_std = _robust_std(cont_resid)
        var_cont = (cont_std * delta_lambda) ** 2 if np.isfinite(cont_std) else 0.0

        F_err = float(np.sqrt(var_pix + var_cont))
    else:
        F_err = np.nan

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
        sigma0=sigma0,
        sigma_was_filled=(sigma is not None) and np.any((~np.isfinite(sigma)) | (sigma <= 0)),
        sigma_was_missing=(sigma is None),
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
    avg_fits_data = import_fits_data(data_path / "fits" / "intercalibrated_fits")

    F_line_list, F_err_list, EW_list, meta_list = [], [], [], []

    for data in avg_fits_data.values():

        F_line, F_err, EW, meta= measure_line_flux(data, line_window, cont_windows)
        F_line_list.append(F_line)
        F_err_list.append(F_err)
        EW_list.append(EW)
        meta_list.append(meta)


    F_line_mean = np.mean(F_line_list)
    F_sigma = np.std(F_line_list)

    mean_square_sigma = np.sum(np.array(F_err_list)**2)/ len(F_err_list)

    F_var = np.sqrt((F_sigma**2 - mean_square_sigma)) / F_line_mean


    print(f"F_var = {F_var:.3e}")



#get_line_flux((4995.66, 5021.75), ((4762, 4774),(5085, 5112)))

#plot_avg_rms_spec(file_name='avg_rms_spec.pdf',shift_factor=(0, -3), ylim=(-2.5, 13), show_only_label=True)

#plot_avg_rms_spec(file_name='avg_rms_spec_OI.pdf', scale_factor=20, shift_factor=(0, -2.5), ylim=(-0.5, 3), xlim=(8300,8700), figsize=(4,3.5), no_description=True, show_only_label=True)


#plot_avg_rms_spec(input_dir=Path("fits") / "uncalibrated_AVG_RMS", file_name='UV_uncalibrated_AVG_RMS.pdf',xlim=(1130, 1710), ylim=(-10, 70), scale_factor=5, shift_factor=(0, -10), line_length=3, show_only_label=True)


selected_lines = {
    # UV
    r"$\mathrm{Ly}\alpha$": {
        "position": 1215.67, "text_vertical_shift": -10, "slanted": False,
        "show_no_tick_avg": True, "text_shift": -15, "tick_shift_rms": 3
    },


    r"$\mathrm{He\,II}\,\lambda\,1640$": {
        "position": 1640.42, "text_vertical_shift": 3, "slanted": False,
        "tick_shift_avg": 2.5, "text_shift": 0, "tick_shift_rms": 3
    },

    # Balmer
    r"$\mathrm{H}\alpha$": {
        "position": 6562.82, "text_vertical_shift": -10, "slanted": False,
        "text_shift": 90, "show_no_tick_avg": True, "tick_shift_rms": 1
    },
    r"$\mathrm{H}\beta$": {
        "position": 4861.33, "text_vertical_shift": -0.7, "slanted": False,
        "show_no_tick_avg": True, "tick_shift_rms": 0.6
    },
    r"$\mathrm{H}\gamma$": {
        "position": 4340.47, "text_vertical_shift": 0.1, "tick_shift_avg": 0.31,
        "text_shift": -30, "slanted": False, "tick_shift_rms": 0.57
    },
    r"$\mathrm{H}\delta$": {"position": 4101.74, "text_vertical_shift": 0.1, "slanted": False},


    r"$\mathrm{He\,I}\,\lambda\,5876$": {"position": 5875.6, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.3},

    # He II optical
    r"$\mathrm{He\,II}\,\lambda\,4686$": {"position": 4685.7, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.4},



    r"$\mathrm{O\,I}\,\lambda\,8446$": {"position": 8446.35, "text_vertical_shift": 0.1, "slanted": False,  "show_no_tick_rms": True, "tick_shift_rms": 0.29},
}


selected_groups = {
    r"$\mathrm{N\,V}\,\lambda\lambda\,1238,\,1242$": {
        "position": [1238.82, 1242.8],
        "tick_vertical_shift_avg": 2,
        "tick_vertical_shift_rms": 7,
        "show_in_rms": False,
        "text_vertical_shift": 2
    },

    r"$\mathrm{Si\,IV}\,\lambda\lambda\,1393,\,1402,\,\mathrm{O\,IV]\,\lambda\lambda\,1397,\,1399}$": {
        "position": [1393.75, 1402.7],
        "tick_vertical_shift_avg": 2,
        "tick_vertical_shift_rms": 2.5,
        "show_in_rms": True,
        "text_vertical_shift": 2
    },

    r"$\mathrm{C\,IV}\,\lambda\lambda\,1548,\,1550$": {
        "position": [1548.19, 1550.77],
        "tick_vertical_shift_avg": 2,
        "tick_vertical_shift_rms": 10,
        "show_in_rms": True,
        "show_in_avg": False,
        "text_vertical_shift": -15,
        "text_horizontal_shift": -15
    },

}

#selected = (selected_lines, selected_groups)

#plot_avg_rms_spec(file_name='avg_rms_spec_selected_lines.pdf',shift_factor=(0, -3), ylim=(-2.5, 13), selected_broad_lines=selected, show_only_label=True)

#plot_avg_rms_spec(input_dir=Path("fits") / "uncalibrated_AVG_RMS", file_name='UV_uncalibrated_AVG_RMS_selected_lines.pdf',xlim=(1130, 1710), ylim=(-10, 70), scale_factor=5, shift_factor=(0, -10), line_length=3, selected_broad_lines=selected, show_only_label=True)

#plot_avg_rms_together()
#plot_calibrated_and_uncalibrated_spectra_together()
#plot_calibrated_and_uncalibrated_spectra_separately()
#plot_calibrated_and_uncalibrated_spectra_separately(xlim=(1130, 1710), ylim=(-2, 80), file_name= "UV", output_dir=DEFAULT_OUTPUT_DIR)