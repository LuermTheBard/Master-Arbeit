import numpy as np
from astropy.constants.codata2018 import c
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

from plot_utils import format_label
from plot_data.general_plot import finalize_figure, prepare_data
from settings import DEFAULT_OUTPUT_DIR, CENTRAL_WAVELENGTH

line_mapping = {
    'HAlpha': 'Hα',
    'HBeta': 'Hβ',
    'HGamma': 'Hγ',
    'HDelta': 'Hδ',
    'HeI5875': 'He I 5875',
    'HeI7065': 'He I 7065',
    'HeI4471': 'He I 4471',
    'HeI5015': 'He I 5015',
    'HeII4685': 'He II 4685',
    'OI8446': 'O I 8446'
}


def plot_normalized_line_profiles_in_groups(data, save_only=False, output_dir=None,
                                            rows=4, cols=2, key_order=None, title="Normalized Line Profiles", shared_y=False):
    """
    Plottet normalisierte Linienprofile (AVG und RMS) gruppiert in Subplots (z. B. 4x2).

    Parameter:
    -----------
    data : dict
        Dictionary mit 'avg' und 'rms'-Daten.
    compare_cont : str
        Bezeichner für Dateinamen-Suffix.
    save_only : bool
        Nur speichern oder auch anzeigen.
    output_dir : Path oder None
        Verzeichnis zum Speichern der Plots.
    rows : int
        Anzahl der Subplot-Zeilen.
    cols : int
        Anzahl der Subplot-Spalten.
    key_order : list oder None
        Falls angegeben, wird diese Reihenfolge für die Linien verwendet.
    title : str
        Titel der Abbildung.

    Returns:
    -----------
    None
    """

    x_key = 'velocity space (km/s)'
    y_key = 'normalized flux'
    ylabel = 'normalized flux'

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / "plot_line_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    if key_order is None:
        key_order = list(data["avg"].keys())

    # Vorbereitung der Datenstruktur für prepare_data
    plot_data = {}
    for line in key_order:
        plot_data[line] = {
            "avg": {
                "x": data["avg"][line]["data_dict"][x_key],
                "y": data["avg"][line]["data_dict"][y_key]
            },
            "rms": {
                "x": data["rms"][line]["data_dict"][x_key],
                "y": data["rms"][line]["data_dict"][y_key]
            }
        }

    for current_data, group_index in prepare_data(plot_data, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(6, 12), sharex=True, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            if line_data is not None:
                avg_x = line_data["avg"]["x"]
                avg_y = line_data["avg"]["y"]
                rms_x = line_data["rms"]["x"]
                rms_y = line_data["rms"]["y"]

            else:
                avg_x = np.array([])
                avg_y = np.array([])
                rms_x = np.array([])
                rms_y = np.array([])

            configure_line_profile_axis(ax, row=row, col=col, ylabel=ylabel,avg_x=avg_x, avg_y=avg_y,
                                        rms_x=rms_x, rms_y=rms_y, line_name=line_name)


        # Figure finalisieren und speichern/anzeigen
        finalize_figure(
            fig=fig,
            axes=axes,
            title=title,
            group_index=group_index,
            save_only=save_only,
            output_dir=output_dir,
            x_label="Velocity (km/s)",
            line_profile=True
        )


def configure_line_profile_axis(ax, row, col, ylabel, avg_x, avg_y, rms_x, rms_y, line_name,
                                 line_lightcurves=False):
    """
    Konfiguriert eine Achse für das Linienprofil (AVG und RMS).

    Parameter:
    -----------
    ax : matplotlib.axes.Axes
        Achse für den aktuellen Subplot.
    row : int
        Zeilenindex im Grid.
    col : int
        Spaltenindex im Grid.
    avg_x, avg_y : array-like
        Daten für den AVG-Plot.
    rms_x, rms_y : array-like
        Daten für den RMS-Plot.
    line_name : str
        Name der Linie (für Titel).
    rows : int
        Anzahl der Zeilen (für Achsenlogik).
    cols : int
        Anzahl der Spalten (für Achsenlogik).
    line_lightcurves : bool
        (Optional) für alternative Achsenbeschriftung.

    Returns:
    -----------
    None
    """

    ax.plot(avg_x, avg_y, label=f'AVG', color='black')
    ax.plot(rms_x, rms_y, label=f'RMS', color='red')
    ax.vlines(0, -0.1, 1.5, linestyles='dashed', color='black')
    ax.text(0.95, 0.95, f'{format_label(line_name, as_latex=False)}', transform=ax.transAxes,
            ha='right', va='top', fontsize=11)

    ax.set_xlim(-9000, 8999)
    ax.set_ylim(-0.1, 1.2)
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(loc='upper left')

    if col == 0:
        if row == 0 and line_lightcurves:
            ax.set_ylabel(
                (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{\AA}^{-1}]$"),
                fontsize=12)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)


    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(1.15, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(2500))
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(2500))
        #ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=9)


pseudo_conts_for_line_avg = {

    'HAlpha': {'blue': (6107, 6129), 'red': (6861, 6900)},
    'HBeta': {'blue': (4762, 4774), 'red': (5085, 5112)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (4026, 4033), 'red': (4197, 4220)},
    'HeI5875': {'blue': (5645, 5653), 'red': (6044, 6057)},
    'HeI7065': {'blue': (6934, 6941), 'red': (7331, 7357)},
    'HeI4471': {'blue': (4210, 4225), 'red': (4762, 4774)},
    'HeI5015': {'blue': (4976, 4981), 'red': (5085, 5112)},
    'HeII4685': {'blue': (4198, 4225), 'red': (4762, 4774)},
    'OI8446': {'blue': (7999, 8025), 'red': (8775, 8798)},
    'OIII5007': {'blue': (4976, 4987), 'red': (5085, 5112)},
}
pseudo_conts_for_line_rms = {

    'HAlpha': {'blue': (6201, 6223), 'red': (6970, 6994)},
    'HBeta': {'blue': (4762, 4774), 'red': (4970, 4990)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (3939, 3950), 'red': (4197, 4220)},
    'HeI5875': {'blue': (5649, 5660), 'red': (6068, 6081)},
    'HeI7065': {'blue': (6934, 6941), 'red': (7335, 7349)},
    'HeI4471': {'blue': (4210, 4225), 'red': (4762, 4774)},
    'HeI5015': {'blue': (4976, 4981), 'red': (5119, 5133)},
    'HeII4685': {'blue': (4198, 4225), 'red': (4770, 4787)},
    'OI8446': {'blue': (8222, 8238), 'red': (8748, 8767)},
    'OIII5007': {'blue': (4976, 4987), 'red': (5085, 5112)},
}


def process_spectrum(wavelength, intensity, line_name, spec_type="rms", output_dir=DEFAULT_OUTPUT_DIR,  plot=False):
    """
    Berechnet das pseudo-continum-subtrahierte Spektrum und speichert die Daten in Dateien.
    """
    line_wavelength = CENTRAL_WAVELENGTH[line_name]

    if spec_type == "avg":
        pseudo_conts_for_line = pseudo_conts_for_line_avg
    else:
        pseudo_conts_for_line = pseudo_conts_for_line_rms


    blue_pseudo_cont = pseudo_conts_for_line[line_name]['blue']
    red_pseudo_cont = pseudo_conts_for_line[line_name]['red']

    wavelength_x_lim = (blue_pseudo_cont[0]-100, red_pseudo_cont[1]+100)

    mask_x_lim = (wavelength >= wavelength_x_lim[0]) & (wavelength <= wavelength_x_lim[1])
    max_intensity = np.max(intensity[mask_x_lim])
    min_intensity = np.min(intensity[mask_x_lim])

    y_lim_original = (min_intensity * 0.9, max_intensity * 1.1)  # Optional: 10% Puffer nach oben

    corrected_intensity, continuum = subtract_continuum(wavelength, intensity, line_wavelength, blue_pseudo_cont, red_pseudo_cont)

    velocity = convert_to_velocity(wavelength, line_wavelength)

    output_dir = output_dir / "substracted_pseudocont_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(str(output_dir / f"{line_name}_{spec_type}_corrected_spectrum.txt"), np.column_stack((wavelength, corrected_intensity, continuum)),
               header="Wavelength (Å)  Intensity  Continuum")
    np.savetxt(str(output_dir /f"{line_name}_normalized_{spec_type}_line_profile-{blue_pseudo_cont}-{red_pseudo_cont}.txt"), np.column_stack((velocity, corrected_intensity)),
               header="velocity space (km/s) \t normalized flux")

    if plot is True:
        plt.figure(figsize=(8, 5))
        plt.plot(wavelength, intensity, label="Originalspektrum")
        plt.plot(wavelength, continuum, label="Interpoliertes Kontinuum", linestyle="dashed")
        plt.axvline(line_wavelength, color="r", linestyle=":", label="Linienzentrum")
        plt.axvspan(blue_pseudo_cont[0], blue_pseudo_cont[1], alpha=0.2, color="r")
        plt.axvspan(red_pseudo_cont[0], red_pseudo_cont[1], alpha=0.2, color="r")
        plt.xlim(wavelength_x_lim)
        plt.ylim(y_lim_original)
        plt.legend()
        plt.xlabel("Wellenlänge (Å)")
        plt.ylabel("Intensität")
        plt.title(line_name + " AVG")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(velocity, corrected_intensity, label="Linienprofil im Geschwindigkeitsraum")
        plt.axvline(0, color="r", linestyle=":", label="v = 0 km/s (Zentrum)")
        plt.xlim(-10000,10000)
        plt.ylim(-0.1, 1.2)
        plt.axhline(0, color="black", linestyle=":")
        plt.legend()
        plt.xlabel("Geschwindigkeit (km/s)")
        plt.ylabel("Intensität")
        plt.title(line_name + " RMS")
        plt.show()


def plot_cut_out_line_profile(cut_out_range, intensity_avg, intensity_rms, line_name, output_path, plot, velocity_avg,
                              velocity_rms):
    range_str = f"{cut_out_range[0]}_{cut_out_range[1]}"
    save_velocity_data_to_txt(output_path / f"{line_name}_avg_profile_{range_str}.txt", velocity_avg, intensity_avg)
    save_velocity_data_to_txt(output_path / f"{line_name}_rms_profile_{range_str}.txt", velocity_rms, intensity_rms)
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        axs[0].plot(velocity_avg, intensity_avg, linestyle='-', marker='o', color='b')
        axs[0].set_ylabel("Avg Profile")
        axs[0].grid(True)

        axs[1].plot(velocity_rms, intensity_rms, linestyle='--', marker='s', color='r')
        axs[1].set_xlabel("Velocity (km/s)")
        axs[1].set_ylabel("RMS Profile")
        axs[1].grid(True)

        plt.suptitle(f"Line Profile: {line_name}")
        plt.show()


def subtract_continuum(wavelength, intensity, line_wavelength, left_range, right_range):
    """
    Subtrahiert das Pseudokontinuum von einer Emissionslinie in einem Spektrum und normalisiert auf das Maximum in einer gegebenen Umgebung.

    :param wavelength: Array der Wellenlängen
    :param intensity: Array der Intensitätswerte
    :param line_wavelength: Zentrale Wellenlänge der Linie
    :param left_range: Tupel (min, max) des linken Bereichs für die Kontinuumsschätzung
    :param right_range: Tupel (min, max) des rechten Bereichs für die Kontinuumsschätzung
    :return: (korrigierte Intensität, Pseudokontinuum)
    """
    left_mask = (wavelength > left_range[0]) & (wavelength < left_range[1])
    right_mask = (wavelength > right_range[0]) & (wavelength < right_range[1])

    left_mean = (np.mean(wavelength[left_mask]), np.mean(intensity[left_mask]))
    right_mean = (np.mean(wavelength[right_mask]), np.mean(intensity[right_mask]))

    continuum_fit = interp1d([left_mean[0], right_mean[0]], [left_mean[1], right_mean[1]], kind="linear",
                             fill_value="extrapolate")
    continuum = continuum_fit(wavelength)

    corrected_intensity = intensity - continuum

    # Bestimme das Maximum innerhalb des Bereichs ±50 um line_wavelength
    norm_range = (line_wavelength - 50, line_wavelength + 50)
    norm_mask = (wavelength > norm_range[0]) & (wavelength < norm_range[1])
    if np.any(norm_mask):
        max_value = np.max(corrected_intensity[norm_mask])
    else:
        max_value = np.max(corrected_intensity)  # Falls der Bereich leer ist, fallback auf das globale Maximum

    corrected_intensity /= max_value

    return corrected_intensity, continuum


def convert_to_velocity(wavelength, line_wavelength):
    """
    Wandelt Wellenlängenwerte in Geschwindigkeitswerte um.
    """
    c_km_s = c.to('km/s').value  # Lichtgeschwindigkeit in km/s
    return (wavelength - line_wavelength) / line_wavelength * c_km_s


def transform_wavelength_to_velocity_and_cut(wavelength, intensity, line_name, velocity_range=None, filename=None):
    """
    Transformiert die Wellenlängenachse in den Geschwindigkeitsraum, normalisiert die Intensität
    auf das Maximum und schneidet optional die Daten auf einen Bereich um 0.

    :param wavelength: Array der Wellenlängen
    :param intensity: Array der Intensitätswerte
    :param line_name: Zentrale Wellenlänge der Linie
    :param velocity_range: Tupel (min, max) zur Begrenzung des Geschwindigkeitsbereichs; min muss negativ, max positiv sein
    :param filename: Falls definiert, wird das Ergebnis in eine Datei gespeichert
    :return: (Geschwindigkeiten, normalisierte Intensität)
    """

    line_wavelength = CENTRAL_WAVELENGTH[line_name]

    # Transformation der Wellenlängen in den Geschwindigkeitsraum
    velocity = convert_to_velocity(wavelength, line_wavelength)

    intensity = normalize_to_maximum(intensity, line_wavelength, wavelength)

    intensity, velocity = cut_normalized_line_out(intensity, velocity, velocity_range)

    # Falls eine Datei angegeben ist, speichern
    if filename is not None:
        with open(str(filename), 'w') as file:
            file.write("# velocity space (km/s) \t normalized flux\n")
            for v, i in zip(velocity, intensity):
                file.write(f"{v}\t{i}\n")

    return velocity, intensity


def normalize_to_maximum(intensity, line_wavelength, wavelength):
    # Normalisierung der Intensität auf das Maximum
    if np.max(intensity) != 0:
        # Bestimme das Maximum innerhalb des Bereichs ±10 um line_wavelength
        norm_range = (line_wavelength - 10, line_wavelength + 10)
        norm_mask = (wavelength > norm_range[0]) & (wavelength < norm_range[1])
        if np.any(norm_mask):
            max_value = np.max(intensity[norm_mask])
        else:
            max_value = np.max(intensity)  # Falls der Bereich leer ist, fallback auf das globale Maximum

        intensity /= max_value
    return intensity


def cut_normalized_line_out(intensity, velocity, velocity_range):
    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if velocity_range is not None:
        min_v, max_v = velocity_range
        min_v = min(min_v, -abs(min_v))  # Sicherstellen, dass min negativ ist
        max_v = max(max_v, abs(max_v))  # Sicherstellen, dass max positiv ist
        mask = (velocity >= min_v) & (velocity <= max_v)
        velocity = velocity[mask]
        intensity = intensity[mask]
    return intensity, velocity


def cut_line_out(intensity, wavelength, wavelength_range):
    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if wavelength_range is not None:
        min_v, max_v = wavelength_range
        mask = (wavelength >= min_v) & (wavelength <= max_v)
        wavelength = wavelength[mask]
        intensity = intensity[mask]
    return intensity, wavelength


def save_velocity_data_to_txt(filename, velocity, intensity):
    """
    Speichert die Geschwindigkeits- und Intensitätswerte in eine TXT-Datei im gewünschten Format.

    :param filename: Name der Datei, in die gespeichert werden soll
    :param velocity: Array der Geschwindigkeitswerte
    :param intensity: Array der Intensitätswerte
    """
    with open(filename, 'w') as file:
        file.write("# velocity space (km/s) \t normalized flux\n")
        for v, i in zip(velocity, intensity):
            file.write(f"{v}\t{i}\n")
