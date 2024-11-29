import matplotlib.pyplot as plt


def plot_two_spectra(x, y1, y2, xlabel, ylabel1, ylabel2, title1, title2, super_title):
    """
    Plots two spectra with the same x-axis in a single figure with two stacked subplots.

    Parameters:
    - x: array-like
        Shared x-axis values (e.g., wavelengths or frequencies).
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
    """
    # Gemeinsame Figur mit zwei übereinanderliegenden Subplots erstellen
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)  # Zwei Zeilen, gemeinsame x-Achse

    # Erster Subplot
    axs[0].plot(x, y1, label="Spectrum 1")
    axs[0].set_ylabel(ylabel1)
    axs[0].set_title(title1)
    axs[0].grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()

    # Zweiter Subplot
    axs[1].plot(x, y2, label="Spectrum 2", color='orange')
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel2)
    axs[1].set_title(title2)
    axs[1].grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    axs[1].legend()

    # Supertitle für die gesamte Figur
    fig.suptitle(super_title, fontsize=16)

    # Layout anpassen, um Überlappungen zu vermeiden
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Platz für den Supertitle lassen

    # Plot anzeigen
    plt.show()


def plot_avg_rms(fits_data):
    avg_dict = None
    rms_dict = None

    galaxy_name = list(fits_data.keys())[0].split("_")[0]

    for key, item in fits_data.items():

        if "avg".casefold() in key.casefold():
            avg_dict = item
        elif "rms".casefold() in key.casefold():
            rms_dict = item

    if avg_dict is None:
        raise Exception("Missing AVG data")

    if rms_dict is None:
        raise Exception("Missing RMS data")

    wavelengths = avg_dict["x_axis"][0]

    avg_flux = avg_dict["data"][0]
    rms_flux = rms_dict["data"][0]

    avg_label = (r"$F_\lambda \, (\mathrm{AVG}) \, [\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                 r"\mathrm{\AA}^{-1}]$")

    rms_label = (r"$F_\lambda \, (\mathrm{RMS}) \, [\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                 r"\mathrm{\AA}^{-1}]$")

    wavelengths_label = r"Rest Wavelength $[\mathrm{\AA}]$"

    avg_titel = "AVG"
    rms_title = "RMS"

    plot_two_spectra(wavelengths,
                     avg_flux,
                     rms_flux,
                     wavelengths_label,
                     avg_label,
                     rms_label,
                     avg_titel,
                     rms_title,
                     galaxy_name)



