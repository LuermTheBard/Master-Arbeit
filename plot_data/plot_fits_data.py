import math

import matplotlib.pyplot as plt
import numpy as np

All_LINES = {
    "Hα": {"position": 6562.82, "offset_avg": 0.01, "slanted_avg": True},
    "Hβ": {"position": 4861.33, "offset_avg": 0.1, "slanted_avg": True},
    "Hγ": {"position": 4340.47, "offset_avg": 0.2, "slanted_avg": True},
    "Hδ": {"position": 4101.74, "offset_avg": 0.1, "slanted_avg": True},
    "Hε": {"position": 3970.08, "offset_avg": 0.1, "slanted_avg": True},

    # "[Ne V] 3345": {"position": 3345.82, "offset_avg": 0.01, "slanted_avg": False},
    # "[Ne V] 3425": {"position": 3425.88, "offset_avg": 0.01, "slanted_avg": False},
    "[Ne III] 3868": {"position": 3868.76, "offset_avg": 0.1, "slanted_avg": True},

    # "He I 3487": {"position": 3487.72, "offset_avg": 0.1, "slanted_avg": True},
    "He I 4471": {"position": 4471.48, "offset_avg": 0.1, "slanted_avg": False},
    "He I 5875": {"position": 5875.6, "offset_avg": 0.1, "slanted_avg": True},
    "He I 5016": {"position": 5015.67, "offset_avg": 0.3, "slanted_avg": True},
    "He I 7065": {"position": 7065.2, "offset_avg": 0.1, "slanted_avg": True},
    "He II 4685": {"position": 4685.7, "offset_avg": 0.1, "slanted_avg": True},
    # "He I 3187": {"position": 3187.74, "offset_avg": 0.2, "slanted_avg": True},
    # "He II 3203": {"position": 3203.1, "offset_avg": 0.1, "slanted_avg": True},

    "[O III] 4363": {"position": 4363.21, "offset_avg": 0.05, "slanted_avg": True},
    "[O III] 4958": {"position": 4958.91, "offset_avg": 0.1, "slanted_avg": False},
    "[O III] 5006": {"position": 5006.84, "offset_avg": 0.1, "slanted_avg": True},

    "[O I] 6364": {"position": 6363.77, "offset_avg": 0.4, "slanted_avg": True},
    "[Fe X] 6375": {"position": 6374.51, "offset_avg": 0.1, "slanted_avg": True},
    "Fe II 6516": {"position": 6516.08, "offset_avg": 0.3, "slanted_avg": False},
    "[Fe VII] 5721": {"position": 5721.7, "offset_avg": 0.1, "slanted_avg": True},
    "[Fe VII] 6087": {"position": 6087, "offset_avg": 0.1, "slanted_avg": True},
    # "[Fe VII] 3586": {"position": 3586.32, "offset_avg": 0.1, "slanted_avg": True},

    "[S II] 6716": {"position": 6716.44, "offset_avg": 0.35, "slanted_avg": True},
    "[S II] 6731": {"position": 6730.81, "offset_avg": 0.1, "slanted_avg": True},

    "[N II] 6548": {"position": 6548.05, "offset_avg": 0.1, "slanted_avg": False},
    "[N II] 6584": {"position": 6583.46, "offset_avg": 0.1, "slanted_avg": False},

    "[Ar III] 7135": {"position": 7135.79, "offset_avg": 0.1, "slanted_avg": True},

    # "O III 3132": {"position": 3132.79, "offset_avg": 0.1, "slanted_avg": True},
    "O I 8446": {"position": 8446.35, "offset_avg": 0.1, "slanted_avg": True},
    "[O I] 6300": {"position": 6300.30, "offset_avg": 0.1, "slanted_avg": False},

    "Ca II 8498": {"position": 8498.02, "offset_avg": 0.1, "slanted_avg": True},
    "Ca II 8542": {"position": 8542.09, "offset_avg": 0.1, "slanted_avg": True},
    "Ca II 8662": {"position": 8662.14, "offset_avg": 0.1, "slanted_avg": True},
}

All_LINE_GROUPS = {
    "Fe II": {"position": [4489,
                           4629.33],
              "offset": 0.2},

    "Fe II ": {"position": [5169.03,
                           5336.18],
              "offset": 0.2},

}


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
    Plots the average (AVG) and RMS flux for a given galaxy spectrum.

    Parameters:
    - fits_data: dict
        A dictionary containing FITS-like data. Expected keys are:
        - "<galaxy_name>_avg": dict
        - "<galaxy_name>_rms": dict
    - save_path: str or None
        Path to save the plot as a file. If None, the plot will only be displayed.
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

    avg_label = (r"$F_\lambda \, (\mathrm{AVG}) \, [\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                 r"\mathrm{\AA}^{-1}]$")
    rms_label = (r"$F_\lambda \, (\mathrm{RMS}) \, [\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                 r"\mathrm{\AA}^{-1}]$")
    wavelengths_label = r"Rest Wavelength $[\mathrm{\AA}]$"
    avg_title = "AVG"
    rms_title = "RMS"

    plot_two_spectra(
        wavelengths,
        avg_flux,
        rms_flux,
        wavelengths_label,
        avg_label,
        rms_label,
        avg_title,
        rms_title,
        galaxy_name,
        lines=All_LINES,
        save_path=save_path,
        log_scale=log_scale,
        xlim=(3800, 8900),
    )


def plot_two_spectra(
        x, y1, y2, xlabel, ylabel1, ylabel2, title1, title2, super_title, lines, save_path=None, log_scale=False,
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

    # Scale y1 to overlay it above y2
    scale_factor = np.max(y2_filtered) / np.max(y1_filtered) * 5
    y1_scaled = y1_filtered * scale_factor

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(x_filtered, y1_scaled, label=f"{title1} (scaled by {scale_factor:.2f})", linestyle="-", color="blue")
    ax.plot(x_filtered, y2_filtered, label=title2, linestyle="-", color="orange")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel1} (scaled) / {ylabel2}")

    if log_scale:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    # ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Add lines with labels
    for label, props in lines.items():
        pos = props["position"]
        offset = props.get("offset", 0.1)

        idx = np.abs(x_filtered - pos).argmin()
        y_pos = y2_filtered[idx] * (1 + offset)

       # ax.axvline(x=pos, color="gray", linestyle="--", linewidth=0.8)
       # ax.text(pos, y_pos, label, fontsize=10, color="black", rotation=90, ha="center", va="bottom")

    fig.suptitle(super_title, fontsize=14)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()



def plot_line_group(ax_obj, positions, x, y_values_avg, group, offset=0.1, all_lines=False):
    """Funktion, um verbundene Linien zu zeichnen."""
    min_pos = min(positions)
    max_pos = max(positions)

    # Bestimme die Größenordnung der Daten für die aktuelle Achse
    max_y_value_avg = np.max(y_values_avg)
    order_of_magnitude_avg = 10 ** math.floor(math.log10(max_y_value_avg))

    # Definiere die konstante Linienlänge (z.B. 10% der Größenordnung)
    line_length_avg = 0.1 * order_of_magnitude_avg

    # Finde den nächstgelegenen Y-Wert zu pos
    idx = np.abs(x - min_pos).argmin()
    spectrum_y = y_values_avg[idx]

    # Linienstart und -ende berechnen
    line_ymin_avg = spectrum_y * (1 + offset)
    line_ymax_avg = line_ymin_avg + line_length_avg

    if all_lines:

        for pos in positions:
            # Vertikale Linien zeichnen
            ax_obj.vlines(x=pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=1.2)
    else:
        ax_obj.vlines(x=min_pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=1.2)
        ax_obj.vlines(x=max_pos, ymin=line_ymin_avg, ymax=line_ymax_avg, color='black', linewidth=1.2)

    ax_obj.text(x=((max_pos - min_pos) / 2) + min_pos,
                y=line_ymax_avg + 0.01 * order_of_magnitude_avg,
                s=group,
                ha='center',
                fontsize=10,
                va="bottom")
    # Horizontale Verbindungslinie
    ax_obj.hlines(y=line_ymax_avg, xmin=min_pos, xmax=max_pos, color='black', linewidth=1.2)
