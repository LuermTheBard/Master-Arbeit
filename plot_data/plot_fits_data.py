import matplotlib.pyplot as plt
import numpy as np


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
        save_path=save_path,
        log_scale=log_scale,
        xlim=(3000, 9000),
    )


def plot_two_spectra(x, y1, y2, xlabel, ylabel1, ylabel2, title1, title2, super_title, save_path=None, log_scale=False, xlim=None):
    """
    Plots two spectra with the same x-axis in a single figure with two stacked subplots.

    Parameters:
    - x: array-like
        Shared x-axis values.
    - y1: array-like
        y-values for the first spectrum.
    - y2: array-like
        y-values for the second spectrum.
    - xlabel: str
        Label for the shared x-axis.
    - ylabel1: str
        Label for the y-axis of the first spectrum.
    - ylabel2: str
        Label for the y-axis of the second spectrum.
    - title1: str
        Title for the first subplot.
    - title2: str
        Title for the second subplot.
    - super_title: str
        Supertitle for the entire figure.
    - save_path: str or None
        Path to save the plot as a file. If None, the plot will only be displayed.
    - log_scale: bool
        If True, use a logarithmic y-scale for the plots.
    - xlim: list or None
        A list [min, max] specifying the limits of the x-axis. If None, the x-axis is not restricted.
    """
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Filter data based on xlim
    if xlim:
        # Konvertiere x (und ggf. andere Daten) in numpy-Arrays, falls sie Listen sind
        x = np.array(x) if isinstance(x, list) else x
        y1 = np.array(y1) if isinstance(y1, list) else y1
        y2 = np.array(y2) if isinstance(y2, list) else y2

        # Filter Daten basierend auf xlim
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_filtered = x[mask]
        y1_filtered = y1[mask]
        y2_filtered = y2[mask]
    else:
        x_filtered = x
        y1_filtered = y1
        y2_filtered = y2

    # Plot the first spectrum
    axs[0].plot(x_filtered, y1_filtered, label=title1)
    axs[0].set_ylabel(ylabel1)
    axs[0].set_title(title1)
    if xlim:
        axs[0].set_xlim(xlim)  # Apply x-axis limits
    if log_scale:
        axs[0].set_yscale("log")
    else:
        axs[0].set_ylim(min(y1_filtered) * 0.9, max(y1_filtered) * 1.1)  # Auto-adjust y-axis
    axs[0].grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()

    # Plot the second spectrum
    axs[1].plot(x_filtered, y2_filtered, label=title2, color='orange')
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel2)
    axs[1].set_title(title2)
    if xlim:
        axs[1].set_xlim(xlim)  # Apply x-axis limits
    if log_scale:
        axs[1].set_yscale("log")
    else:
        axs[1].set_ylim(min(y2_filtered) * 0.9, max(y2_filtered) * 1.1)  # Auto-adjust y-axis
    axs[1].grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    axs[1].legend()

    # Supertitle and layout
    fig.suptitle(super_title, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()