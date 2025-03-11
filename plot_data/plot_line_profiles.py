import matplotlib.pyplot as plt
import numpy as np
from astropy.constants.codata2018 import c
from scipy.interpolate import interp1d

from settings import VALUES_CONTINUA, All_LINES, DEFAULT_OUTPUT_DIR

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

central_wave_length = {

    'HAlpha': 6562.82,
    'HBeta': 4861.33,
    'HGamma': 4340.47,
    'HDelta': 4101.74,
    'HeI5875': 5875.6,
    'HeI7065': 7065.2,
    'HeI4471': 4471.48,
    'HeI5015': 5015.67,
    'HeII4685': 4685.7,
    'OI8446': 8446.35
}


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
    'OI8446': {'blue': (7999, 8025), 'red': (8775, 8798)}
}

pseudo_conts_for_line_rms = {

    'HAlpha': {'blue': (6201, 6223), 'red': (6970, 6994)},
    'HBeta': {'blue': (4435, 4450), 'red': (4980, 4987)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (3939, 3950), 'red': (4197, 4220)},
    'HeI5875': {'blue': (5649, 5660), 'red': (6068, 6081)},
    'HeI7065': {'blue': (6934, 6941), 'red': (7335, 7349)},
    'HeI4471': {'blue': (4210, 4225), 'red': (4762, 4774)},
    'HeI5015': {'blue': (4976, 4981), 'red': (5119, 5133)},
    'HeII4685': {'blue': (4198, 4225), 'red': (4770, 4787)},
    'OI8446': {'blue': (8222, 8238), 'red': (8748, 8767)}
}


def plot_normalized_line_profiles_in_pairs(data, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    x_keys = 'velocity space (km/s)'
    y_keys = 'normalized flux'

    if key_order is None:
        key_order = list(data["avg"].keys())

    save_path_dir = output_dir / "plot_line_profiles"
    save_path_dir.mkdir(parents=True, exist_ok=True)

    for line in key_order:
        avg_data = data["avg"][line]
        rms_data = data["rms"][line]

        avg_x, avg_y = avg_data["data_dict"][x_keys], avg_data["data_dict"][y_keys]
        rms_x, rms_y = rms_data["data_dict"][x_keys], rms_data["data_dict"][y_keys]


        y_limit_avg = (-0.1, 1.5)
        y_limit_rms = (-0.1, 1.5)

        x_limit = (-10000, 10000)

        fig, axes = plt.subplots(2, 1, figsize=(6, 16))  # Höheres Format für gestreckte Y-Achse
        fig.suptitle(f'Line Profile: {line}', fontsize=16)


        axes[0].vlines(0, y_limit_avg[0], y_limit_avg[1], linestyles='dashed', color='black')

        # **Plot AVG**
        axes[0].plot(avg_x, avg_y, label='AVG', color='blue')
        axes[0].set_xlim(x_limit)
        axes[0].set_ylim(y_limit_avg)  # Unverändert lassen
        axes[0].set_xlabel("Velocity Space (km/s)", fontsize=14)
        axes[0].set_ylabel("Normalized Flux (AVG)", fontsize=14)
        axes[0].legend(fontsize=10, loc='upper right')

        # **Plot RMS**

        axes[1].vlines(0, y_limit_avg[0], y_limit_avg[1], linestyles='dashed', color='black')

        axes[1].plot(rms_x, rms_y, label='RMS', color='red')
        axes[1].set_xlim(x_limit)
        axes[1].set_ylim(y_limit_rms)  # Unverändert lassen
        axes[1].set_xlabel("Velocity Space (km/s)", fontsize=14)
        axes[1].set_ylabel("Normalized Flux (RMS)", fontsize=14)
        axes[1].legend(fontsize=10, loc='upper right')

        plt.subplots_adjust(hspace=0.5)  # Mehr Abstand zwischen den Subplots

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # **Speichern oder Anzeigen**
        save_path_png = save_path_dir / f"{line}.png"
        save_path_pdf = save_path_dir / f"{line}.pdf"

        plt.savefig(save_path_png, dpi=500)
        plt.savefig(save_path_pdf, dpi=500)
        if not save_only:
            plt.show()
        plt.close(fig)


def plot_normalized_line_profiles_type_together(data, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    x_keys = 'velocity space (km/s)'
    y_keys = 'normalized flux'

    if key_order is None:
        key_order = list(data["avg"].keys())

    save_path_dir = output_dir / "plot_line_profiles"
    save_path_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6, 16))  # Höheres Format für gestreckte Y-Achse
    fig.suptitle(f'Line Profile: {"/".join(key_order)}', fontsize=16)

    for line in key_order:
        avg_data = data["avg"][line]
        rms_data = data["rms"][line]

        avg_x, avg_y = avg_data["data_dict"][x_keys], avg_data["data_dict"][y_keys]
        rms_x, rms_y = rms_data["data_dict"][x_keys], rms_data["data_dict"][y_keys]


        y_limit_avg = (-0.1, 1.5)
        y_limit_rms = (-0.1, 1.5)

        x_limit = (-15000, 15000)




        axes[0].vlines(0, y_limit_avg[0], y_limit_avg[1], linestyles='dashed', color='black')

        # **Plot AVG**
        axes[0].plot(avg_x, avg_y, label=line)
        axes[0].set_xlim(x_limit)
        axes[0].set_ylim(y_limit_avg)  # Unverändert lassen
        axes[0].set_xlabel("Velocity Space (km/s)", fontsize=14)
        axes[0].set_ylabel("Normalized Flux (AVG)", fontsize=14)
        axes[0].legend(fontsize=10, loc='upper right')

        # **Plot RMS**

        axes[1].vlines(0, y_limit_avg[0], y_limit_avg[1], linestyles='dashed', color='black')

        axes[1].plot(rms_x, rms_y, label=line)
        axes[1].set_xlim(x_limit)
        axes[1].set_ylim(y_limit_rms)  # Unverändert lassen
        axes[1].set_xlabel("Velocity Space (km/s)", fontsize=14)
        axes[1].set_ylabel("Normalized Flux (RMS)", fontsize=14)
        axes[1].legend(fontsize=10, loc='upper right')

    plt.subplots_adjust(hspace=0.5)  # Mehr Abstand zwischen den Subplots

    plt.tight_layout(rect=[0, 0, 1, 0.96])

        # **Speichern oder Anzeigen**
    save_path_png = save_path_dir / f"{'-'.join(key_order)}_type_together.png"
    save_path_pdf = save_path_dir / f"{'-'.join(key_order)}_type_together.pdf"

    plt.savefig(save_path_png, dpi=500)
    plt.savefig(save_path_pdf, dpi=500)
    if not save_only:
        plt.show()
    plt.close(fig)


def plot_normalized_line_profiles_together(data, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    x_keys = 'velocity space (km/s)'
    y_keys = 'normalized flux'

    if key_order is None:
        key_order = list(data["avg"].keys())

    save_path_dir = output_dir / "plot_line_profiles"
    save_path_dir.mkdir(parents=True, exist_ok=True)

    for line in key_order:
        avg_data = data["avg"][line]
        rms_data = data["rms"][line]

        avg_x, avg_y = avg_data["data_dict"][x_keys], avg_data["data_dict"][y_keys]
        rms_x, rms_y = rms_data["data_dict"][x_keys], rms_data["data_dict"][y_keys]

        y_limit = (-0.1, 1.5)
        x_limit = (-10000, 10000)

        fig, ax = plt.subplots(figsize=(8, 6))  # Gemeinsamer Plot
        fig.suptitle(f'Line Profile: {line}', fontsize=16)

        # Vertikale Linie bei 0
        ax.vlines(0, y_limit[0], y_limit[1], linestyles='dashed', color='black', label="0 km/s")

        # Plot AVG und RMS
        ax.plot(avg_x, avg_y, label='AVG', color='blue')
        ax.plot(rms_x, rms_y, label='RMS', color='red')

        ax.set_xlim(x_limit)
        ax.set_ylim(y_limit)
        ax.set_xlabel("Velocity Space (km/s)", fontsize=14)
        ax.set_ylabel("Normalized Flux", fontsize=14)
        ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout()

        # **Speichern oder Anzeigen**
        save_path_png = save_path_dir / f"{line}_compared.png"
        save_path_pdf = save_path_dir / f"{line}_compared.pdf"

        plt.savefig(save_path_png, dpi=500)
        plt.savefig(save_path_pdf, dpi=500)
        if not save_only:
            plt.show()
        plt.close(fig)


def plot_line_profiles_in_pairs(data, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    x_keys = 'velocity space (km/s)'
    y_keys = 'flux ergs/s/cm2/A'

    if key_order is None:
        key_order = list(data["avg"].keys())

    save_path_dir = output_dir / "plot_line_profiles"
    save_path_dir.mkdir(parents=True, exist_ok=True)

    for line in key_order:
        avg_data = data["avg"][line]
        rms_data = data["rms"][line]

        avg_x, avg_y = avg_data["data_dict"][x_keys], avg_data["data_dict"][y_keys]
        rms_x, rms_y = rms_data["data_dict"][x_keys], rms_data["data_dict"][y_keys]

        real_line_name = line_mapping[line]
        line_position = All_LINES[real_line_name]["position"]

        # Original x_limit Werte
        x_min_avg, x_max_avg = VALUES_CONTINUA[avg_data["pseudo_conts"][0]][0], VALUES_CONTINUA[avg_data["pseudo_conts"][1]][1]
        x_min_rms, x_max_rms = VALUES_CONTINUA[rms_data["pseudo_conts"][0]][0], VALUES_CONTINUA[rms_data["pseudo_conts"][1]][1]

        # Begrenzung der x-Werte auf max. 300 Å
        avg_half_width = min(max(line_position - x_min_avg, x_max_avg - line_position), 150)
        rms_half_width = min(max(line_position - x_min_rms, x_max_rms - line_position), 150)

        x_limit_avg = (line_position - avg_half_width, line_position + avg_half_width)
        x_limit_rms = (line_position - rms_half_width, line_position + rms_half_width)

        ### 🔹 **Nur den Peak direkt an der Linienposition verwenden – separat für AVG und RMS**
        peak_window = 10  # Enge Auswahl um die Linie

        peak_avg_mask = (avg_x >= line_position - peak_window) & (avg_x <= line_position + peak_window)
        peak_rms_mask = (rms_x >= line_position - peak_window) & (rms_x <= line_position + peak_window)

        # Falls keine Werte im Peak-Bereich gefunden werden, benutze die x-Limits als Fallback
        if not np.any(peak_avg_mask):
            peak_avg_mask = (avg_x >= x_limit_avg[0]) & (avg_x <= x_limit_avg[1])
        if not np.any(peak_rms_mask):
            peak_rms_mask = (rms_x >= x_limit_rms[0]) & (rms_x <= x_limit_rms[1])

        # **Normierung: AVG und RMS getrennt**
        peak_avg_max = avg_y[peak_avg_mask].max()
        peak_rms_max = rms_y[peak_rms_mask].max()

        avg_y_norm = avg_y / peak_avg_max  # **Normierung auf den Peak von AVG**
        rms_y_norm = rms_y / peak_rms_max  # **Normierung auf den Peak von RMS**

        # Fixierte y-Limits 10 % Puffer) für normierte Werte
        y_limit_avg = (-0.1, 1.1)
        y_limit_rms = (-0.1, 1.1)

        fig, axes = plt.subplots(2, 1, figsize=(6, 16))  # Höheres Format für gestreckte Y-Achse
        fig.suptitle(f'Line Profile: {line}', fontsize=16)

        # **Plot AVG**
        axes[0].plot(avg_x, avg_y_norm, label='AVG', color='blue')
        axes[0].set_xlim(x_limit_avg)
        axes[0].set_ylim(y_limit_avg)  # Unverändert lassen
        axes[0].set_xlabel("Velocity Space (km/s)", fontsize=14)
        axes[0].set_ylabel("Normalized Flux (AVG)", fontsize=14)
        axes[0].legend(fontsize=10, loc='upper right')

        # **Plot RMS**
        axes[1].plot(rms_x, rms_y_norm, label='RMS', color='red')
        axes[1].set_xlim(x_limit_rms)
        axes[1].set_ylim(y_limit_rms)  # Unverändert lassen
        axes[1].set_xlabel("Velocity Space (km/s)", fontsize=14)
        axes[1].set_ylabel("Normalized Flux (RMS)", fontsize=14)
        axes[1].legend(fontsize=10, loc='upper right')

        plt.subplots_adjust(hspace=0.5)  # Mehr Abstand zwischen den Subplots

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # **Speichern oder Anzeigen**
        save_path_png = save_path_dir / f"{line}.png"
        save_path_pdf = save_path_dir / f"{line}.pdf"

        plt.savefig(save_path_png, dpi=500)
        plt.savefig(save_path_pdf, dpi=500)
        if not save_only:
            plt.show()
        plt.close(fig)


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

    # Bestimme das Maximum innerhalb des Bereichs ±10 um line_wavelength
    norm_range = (line_wavelength - 10, line_wavelength + 10)
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


def transform_wavelength_to_velocity(wavelength, intensity, line_wavelength, velocity_range=None, filename=None):
    """
    Transformiert die Wellenlängenachse in den Geschwindigkeitsraum und schneidet optional die Daten auf einen Bereich um 0.

    :param wavelength: Array der Wellenlängen
    :param intensity: Array der Intensitätswerte
    :param line_wavelength: Zentrale Wellenlänge der Linie
    :param velocity_range: Tupel (min, max) zur Begrenzung des Geschwindigkeitsbereichs; min muss negativ, max positiv sein
    :param filename: Falls definiert, wird das Ergebnis in eine Datei gespeichert
    :return: (Geschwindigkeiten, Intensität)
    """
    # Transformation der Wellenlängen in den Geschwindigkeitsraum
    velocity = convert_to_velocity(wavelength, line_wavelength)

    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if velocity_range is not None:
        min_v, max_v = velocity_range
        min_v = min(min_v, -abs(max_v))  # Sicherstellen, dass min negativ ist
        max_v = max(max_v, abs(min_v))  # Sicherstellen, dass max positiv ist
        mask = (velocity >= min_v) & (velocity <= max_v)
        velocity = velocity[mask]
        intensity = intensity[mask]

    # Falls eine Datei angegeben ist, speichern
    if filename is not None:
        with open(filename, 'w') as file:
            file.write("# velocity space (km/s) \t normalized flux\n")
            for v, i in zip(velocity, intensity):
                file.write(f"{v}\t{i}\n")

    return velocity, intensity


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


def process_spectrum(wavelength, intensity, line_name, spec_type="rms", output_dir=DEFAULT_OUTPUT_DIR,  plot=False):
    """
    Berechnet das pseudo-continum-subtrahierte Spektrum und speichert die Daten in Dateien.
    """
    line_wavelength = central_wave_length[line_name]

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

    mask_x_lim = (velocity >= -20000) & (velocity <= 20000)
    max_intensity = np.max(corrected_intensity[mask_x_lim])
    min_intensity = np.min(corrected_intensity[mask_x_lim])

    y_lim_velocity = (min_intensity * 0.9, max_intensity * 1.1)

    output_dir = output_dir / "substracted_pseudocont_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(str(output_dir / f"{line_name}_{spec_type}_corrected_spectrum.txt"), np.column_stack((wavelength, corrected_intensity, continuum)),
               header="Wavelength (Å)  Intensity  Continuum")
    np.savetxt(str(output_dir /f"{line_name}_normalized_{spec_type}_line_profile-{blue_pseudo_cont}-{red_pseudo_cont}.txt"), np.column_stack((velocity, corrected_intensity)),
               header="# velocity space (km/s) 	 normalized flux")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(wavelength, intensity, label="Originalspektrum")
        plt.plot(wavelength, continuum, label="Interpoliertes Kontinuum", linestyle="dashed")
        plt.axvline(line_wavelength, color="r", linestyle=":", label="Linienzentrum")
        plt.xlim(wavelength_x_lim)
        plt.ylim(y_lim_original)
        plt.legend()
        plt.xlabel("Wellenlänge (Å)")
        plt.ylabel("Intensität")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(velocity, corrected_intensity, label="Linienprofil im Geschwindigkeitsraum")
        plt.axvline(0, color="r", linestyle=":", label="v = 0 km/s (Zentrum)")
        plt.xlim(-20000,20000)
        plt.ylim(y_lim_velocity)
        plt.legend()
        plt.xlabel("Geschwindigkeit (km/s)")
        plt.ylabel("Intensität")
        plt.show()

