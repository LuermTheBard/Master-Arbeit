import matplotlib.pyplot as plt

from settings import VALUES_CONTINUA


def plot_line_profiles_in_pairs(data, campaign, key_order=None):
    x_keys = 'velocity space (km/s)'
    y_keys = 'flux ergs/s/cm2/A'

    x_label = x_keys
    y_label = y_keys

    if key_order is None:
        key_order = list(data["avg"].keys())

    for line in key_order:
        avg_line_data = data["avg"][line]
        pseudo_conts_avg = avg_line_data["pseudo_conts"]
        rms_line_data = data["rms"][line]
        pseudo_conts_rms = rms_line_data["pseudo_conts"]

        x_limit_avg = (VALUES_CONTINUA[pseudo_conts_avg[0]][0], VALUES_CONTINUA[pseudo_conts_avg[1]][1])
        x_limit_rms = (VALUES_CONTINUA[pseudo_conts_rms[0]][0], VALUES_CONTINUA[pseudo_conts_rms[1]][1])

        avg_x = avg_line_data["data_dict"][x_keys]
        avg_y = avg_line_data["data_dict"][y_keys]

        rms_x = rms_line_data["data_dict"][x_keys]
        rms_y = rms_line_data["data_dict"][y_keys]

        fig, axes = plt.subplots(2, 1, figsize=(8, 12))

        axes[0].plot(avg_x, avg_y, label=f'AVG')

        axes[0].legend(fontsize=8, loc='upper right')

        axes[0].set_ylabel(y_label, fontsize=12)
        axes[0].set_xlabel(x_label, fontsize=12)
        # axes[0]yaxis.set_label_coords(-0.19, 0.5)

        # axes[0].set_xticklabels([])

        mask = (avg_x >= x_limit_avg[0]) & (avg_x <= x_limit_avg[1])
        max_y_in_range = max(avg_y[mask])

        axes[0].set_xlim(x_limit_avg)
        axes[0].set_ylim(0, max_y_in_range)

        axes[1].plot(rms_x, rms_y, label=f'RMS')

        axes[1].legend(fontsize=8, loc='upper right')

        axes[1].set_ylabel(y_label, fontsize=12)
        axes[1].set_xlabel(x_label, fontsize=12)
        # axes[1]yaxis.set_label_coords(-0.19, 0.5)

        # axes[1].set_xticklabels([])

        mask = (rms_x >= x_limit_rms[0]) & (rms_x <= x_limit_rms[1])
        max_y_in_range = max(rms_y[mask])

        axes[1].set_xlim(x_limit_rms)
        axes[1].set_ylim(0, max_y_in_range)

        plt.show()

    print()
