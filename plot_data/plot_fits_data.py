from turtle import pd

import matplotlib.pyplot as plt
import numpy as np

from settings import All_LINES, All_LINE_GROUPS


def validate_fits_data(fits_data):
    """
    Validates the structure of the input FITS-like data dictionary.

    Parameters:
    - fits_data: dict
        The input data dictionary to validate.

    Raises:
    - ValueError: If the data structure is invalid.
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


def plot_avg_rms(fits_data, save_path=None, log_scale=False):
    """
    Plots the average (AVG) and RMS flux for a given galaxy spectrum and saves the data to files.

    Parameters:
    - fits_data: dict
        A dictionary containing FITS-like data. Expected keys are:
        - "<galaxy_name>_avg": dict
        - "<galaxy_name>_rms": dict
    - save_path: Path object or None
        Path to save the plot and data files. If None, only the plot is displayed.
    - log_scale: bool
        If True, use a logarithmic y-scale for the plots.
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

    plot_avg_rms_spectra(
        wavelengths,
        avg_flux,
        rms_flux,
        wavelengths_label,
        label,
        avg_title,
        rms_title,
        galaxy_name,
        lines=All_LINES,
        groups=All_LINE_GROUPS,
        save_path=save_path / f"avg_rms_spec.pdf" if save_path else None,
        log_scale=log_scale,
        xlim=(3800, 8900), #xlim=(3800, 10100),
        ylim=(0, 14), #ylim=(0, 100)
    )

    if save_path:
        avg_data = np.column_stack((wavelengths, avg_flux))
        rms_data = np.column_stack((wavelengths, rms_flux))

        header_line = 'wavelength flux [ergs/s/cm2/A]'

        np.savetxt(save_path / f"{galaxy_name}_avg_flux.txt", avg_data,
                   delimiter=" ", header=header_line, comments='')
        np.savetxt(save_path / f"{galaxy_name}_rms_flux.txt", rms_data,
                   delimiter=" ", header=header_line, comments='')

def plot_avg_rms_spectra(
        x, y1, y2, xlabel, ylabel, title1, title2, super_title, lines, groups, save_path=None, log_scale=False,
        xlim=None,
        ylim=None):
    """
    Plots two spectra in the same figure, scaling the first spectrum to overlay it above the second spectrum.
    """

    if xlim:
        x = np.array(x) if isinstance(x, list) else x
        y1 = np.array(y1) if isinstance(y1, list) else y1
        y2 = np.array(y2) if isinstance(y2, list) else y2

        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_filtered = x[mask]
        y1_filtered = y1[mask]
        y2_filtered = y2[mask]
    else:
        x = np.array(x) if isinstance(x, list) else x
        y1 = np.array(y1) if isinstance(y1, list) else y1
        y2 = np.array(y2) if isinstance(y2, list) else y2
        x_filtered = x
        y1_filtered = y1
        y2_filtered = y2

    # Berechnung des Exponenten
    exponent_value = int(f"{min(y1_filtered):.1e}".split("e")[1])
    exponent = 10 ** exponent_value
    latex_exponent = f"10^{{{exponent_value}}}"

    line_length = 0.75

    # Aktualisierung des y-Labels
    ylabel_parts = ylabel.split("[")
    new_ylabel = ylabel_parts[0] + f"[{latex_exponent} " + ylabel_parts[1]

    y1_filtered = y1_filtered / exponent
    y2_filtered = y2_filtered / exponent

    shift_factor_1 = 2  # 50
    shift_factor_2 = -0.8 # 0

    # Scale y1 to overlay it above y2
    scale_factor = 8
    y2_scaled = y2_filtered * scale_factor + shift_factor_2

    y1_filtered = y1_filtered + shift_factor_1


    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 12))

    ax.plot(x_filtered, y1_filtered, label=f"{title1} (shifted by {shift_factor_1:.1f})", linestyle="-", color="blue")
    ax.plot(x_filtered, y2_scaled, label=f"{title2} (scaled by {scale_factor:.1f}, shifted by {shift_factor_2:.1f})", linestyle="-", color="orange")

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(f"{new_ylabel} + const.", fontsize=16)

    if log_scale:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=16)

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
            ax.plot([pos, pos], [y_pos1, y_pos1 + line_length], color="black", linewidth=1.5)
        if not show_no_tick_rms:
            ax.plot([pos, pos], [y_pos2, y_pos2 + line_length], color="black", linewidth=1.5)
        ax.text(pos + text_shift, y_pos1 + line_length + text_vertical_shift, label, fontsize=14, color="black", rotation=rotation_angle,
                ha="left" if slanted else "center",
                va="bottom")

    for group, data in groups.items():
        plot_line_group(ax, data["position"], x_filtered, y1_filtered, y2_scaled, group, line_length=line_length,
                        tick_vertical_shift_avg=data.get("tick_vertical_shift_avg",0.2), tick_vertical_shift_rms=data.get("tick_vertical_shift_rms", 0.2), show_in_rms=data.get("show_in_rms", False), all_lines=data.get("all_lines", False))

    plt.title(super_title, fontsize=20)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

        savepath_png = f"{save_path}.png"
        fig.savefig(savepath_png, dpi=300)

    plt.show()


def plot_line_group(ax_obj, positions, x, y1_filtered, y2_scaled, group, line_length=0.5, tick_vertical_shift_avg=0.5, tick_vertical_shift_rms=0.5, all_lines=False,
                    show_in_rms=False):
    """Funktion, um verbundene Linien zu zeichnen."""
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

    if all_lines:

        for pos in positions:
            # Vertikale Linien zeichnen
            ax_obj.vlines(x=pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=1.5)

    else:
        ax_obj.vlines(x=min_pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=1.5)
        ax_obj.vlines(x=max_pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=1.5)

    if show_in_rms:
        for pos in positions:
            ax_obj.vlines(x=pos, ymin=line_ymin_rms, ymax=line_ymax_rms, color='black', linewidth=1.5)

    ax_obj.text(x=((max_pos - min_pos) / 2) + min_pos,
                y=line_ymax_avg + 0.1,
                s=group,
                ha='center',
                fontsize=14,
                rotation=90,
                va="bottom")
    # Horizontale Verbindungslinie
    ax_obj.hlines(y=line_ymax_avg, xmin=min_pos, xmax=max_pos, color='black', linewidth=1.2)
