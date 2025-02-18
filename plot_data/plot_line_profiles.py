import matplotlib.pyplot as plt
import numpy as np

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
