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

    'HAlpha': 6564.27,
    'HBeta': 4862.26,
    'HGamma': 4362.21,
    'HDelta': 4101.32,
    #'HeI5875': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeI7065': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeI4471': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeI5015': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeII4685': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'OI8446': {'blue': (6107, 6129), 'red': (6861, 6900)}
}


pseudo_conts_for_line_avg = {

    'HAlpha': {'blue': (6107, 6129), 'red': (6861, 6900)},
    'HBeta': {'blue': (4762, 4774), 'red': (5085, 5112)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (4026, 6033), 'red': (4197, 4220)},
    #'HeI5875': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeI7065': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeI4471': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeI5015': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'HeII4685': {'blue': (6107, 6129), 'red': (6861, 6900)},
    #'OI8446': {'blue': (6107, 6129), 'red': (6861, 6900)}
}

pseudo_conts_for_line_rms = {

    'HAlpha': {'blue': (6107, 6129), 'red': (6861, 6900)},
    'HBeta': {'blue': (4435, 4450), 'red': (4980, 4987)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (4026, 6033), 'red': (4197, 4220)},
    #'HeI5875': 'He I 5875',
    #'HeI7065': 'He I 7065',
    #'HeI4471': 'He I 4471',
    #'HeI5015': 'He I 5015',
    #'HeII4685': 'He II 4685',
    #'OI8446': 'O I 8446'
}


def plot_normalized_line_profiles_in_pairs(data, campaign, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
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
        fig.suptitle(f'{campaign}\n\nLine Profile: {line}', fontsize=16)


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
        save_path_png = save_path_dir / f"{campaign}_{line}.png"
        save_path_pdf = save_path_dir / f"{campaign}_{line}.pdf"

        plt.savefig(save_path_png, dpi=500)
        plt.savefig(save_path_pdf, dpi=500)
        if not save_only:
            plt.show()
        plt.close(fig)


def plot_normalized_line_profiles_together(data, campaign, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
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
        fig.suptitle(f'{campaign}\n\nLine Profile: {line}', fontsize=16)

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
        save_path_png = save_path_dir / f"{campaign}_{line}_compared.png"
        save_path_pdf = save_path_dir / f"{campaign}_{line}_compared.pdf"

        plt.savefig(save_path_png, dpi=500)
        plt.savefig(save_path_pdf, dpi=500)
        if not save_only:
            plt.show()
        plt.close(fig)


def plot_line_profiles_in_pairs(data, campaign, key_order=None, output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
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
        fig.suptitle(f'{campaign}\n\nLine Profile: {line}', fontsize=16)

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
        save_path_png = save_path_dir / f"{campaign}_{line}.png"
        save_path_pdf = save_path_dir / f"{campaign}_{line}.pdf"

        plt.savefig(save_path_png, dpi=500)
        plt.savefig(save_path_pdf, dpi=500)
        if not save_only:
            plt.show()
        plt.close(fig)


def subtract_continuum(wavelength, intensity, left_range, right_range):
    """
    Subtrahiert das Pseudokontinuum von einer Emissionslinie in einem Spektrum.
    """
    left_mask = (wavelength > left_range[0]) & (wavelength < left_range[1])
    right_mask = (wavelength > right_range[0]) & (wavelength < right_range[1])

    left_mean = (np.mean(wavelength[left_mask]), np.mean(intensity[left_mask]))
    right_mean = (np.mean(wavelength[right_mask]), np.mean(intensity[right_mask]))

    continuum_fit = interp1d([left_mean[0], right_mean[0]], [left_mean[1], right_mean[1]], kind="linear",
                             fill_value="extrapolate")
    continuum = continuum_fit(wavelength)

    corrected_intensity = intensity - continuum

    # Normalisierung auf das Maximum der Linie
    corrected_intensity /= np.max(corrected_intensity)

    return corrected_intensity, continuum


def convert_to_velocity(wavelength, line_wavelength):
    """
    Wandelt Wellenlängenwerte in Geschwindigkeitswerte um.
    """
    c_km_s = c.to('km/s').value  # Lichtgeschwindigkeit in km/s
    return (wavelength - line_wavelength) / line_wavelength * c_km_s


def process_spectrum(wavelength, intensity, line_name, spec_type="rms", output_dir=DEFAULT_OUTPUT_DIR,  plot=False):
    """
    Berechnet das kontaminumsubtrahierte Spektrum und speichert die Daten in Dateien.
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

    y_lim_original = (0, max_intensity * 1.1)  # Optional: 10% Puffer nach oben

    corrected_intensity, continuum = subtract_continuum(wavelength, intensity, blue_pseudo_cont, red_pseudo_cont)

    velocity = convert_to_velocity(wavelength, line_wavelength)

    mask_x_lim = (velocity >= -20000) & (velocity <= 20000)
    max_intensity = np.max(corrected_intensity[mask_x_lim])

    y_lim_velocity = (0, max_intensity * 1.1)

    output_dir = output_dir / "substracted_pseudocont_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(str(output_dir / f"{line_name}_{spec_type}_corrected_spectrum.txt"), np.column_stack((wavelength, corrected_intensity, continuum)),
               header="Wavelength (Å)  Intensity  Continuum")
    np.savetxt(str(output_dir /f"{line_name}_{spec_type}_velocity_spectrum.txt"), np.column_stack((velocity, corrected_intensity)),
               header="Velocity (km/s)  Intensity")

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

