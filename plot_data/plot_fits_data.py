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


def plot_two_spectra(
    x, y1, y2, xlabel, ylabel1, ylabel2, title1, title2, super_title, save_path=None, log_scale=False, xlim=None
):
    """
    Plots two spectra with specific lines and labels placed vertically above the lines.
    Removes excessive space between the title and the figure.
    """
    # Linienpositionen und Namen
    lines = {
        "Hα": 6562.82,
        "Hβ": 4861.33,
        "Hγ": 4340.47,
        "Hδ": 4101.74,
        "He I 5875": 5875.6,
        "He I 5015": 5015.7,
        "He II 4685": 4685.7,
        "O I 8446": 8446.35,
    }

    if xlim:
        x = np.array(x) if isinstance(x, list) else x
        y1 = np.array(y1) if isinstance(y1, list) else y1
        y2 = np.array(y2) if isinstance(y2, list) else y2

        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_filtered = x[mask]
        y1_filtered = y1[mask]
        y2_filtered = y2[mask]
    else:
        x_filtered = x
        y1_filtered = y1
        y2_filtered = y2

    # Adjusted figure size
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot for y1
    axs[0].plot(x_filtered, y1_filtered, label=title1)
    axs[0].set_ylabel(ylabel1)
    if xlim:
        axs[0].set_xlim(xlim)
    if log_scale:
        axs[0].set_yscale("log")
    axs[0].grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    axs[0].legend()

    # Plot for y2
    axs[1].plot(x_filtered, y2_filtered, label=title2, color="orange")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel2)
    if xlim:
        axs[1].set_xlim(xlim)
    if log_scale:
        axs[1].set_yscale("log")
    axs[1].grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    axs[1].legend()

    # Add vertical lines through both plots
    for label, pos in lines.items():
        for ax in axs:
            ax.axvline(pos, color="gray", linestyle="--", linewidth=0.8)

    # Add vertical labels above the lines
    for label, pos in lines.items():
        axs[0].text(
            x=pos,
            y=1.05,  # Position slightly above the plot
            s=label,
            fontsize=10,
            color="black",
            rotation=90,  # Vertical text
            ha="center",
            va="bottom",
            transform=axs[0].get_xaxis_transform(),  # Align with x-axis
        )

    fig.suptitle(super_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Reduce space between title and figure

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()
