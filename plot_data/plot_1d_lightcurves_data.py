import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")


def plot_1d_lightcurves(galaxie_campaigns_dict, output_dir, save_only=False):
    xlabel = 'timestamps [MJD]'
    ylabel = 'fluxes [ergs/s/cm2/A]'
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, light_curve_data in galaxie_campaigns_dict.items():

        super_title = campaign

        fig_line_continua, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        plt.figure()

        ax1 = axs[0]
        for line, data in light_curve_data["lines"].items():
            x = data[xlabel]
            y = data[ylabel]
            y_err = data[yerr_name]

            # Normierung der y-Werte
            max_y = max(y)
            y = y / max_y
            y_err = y_err / max_y  # Fehler ebenfalls normieren

            ax1.errorbar(x, y, y_err, label=line)
            plt.errorbar(x, y, y_err, label=line)

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("fluxes (normalized)")
        ax1.set_title(f"1D Lightcurves of lines")
        ax1.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend(fontsize=8)

        plt.xlabel(xlabel)
        plt.ylabel("fluxes (normalized)")
        plt.title(f"1D Lightcurves of lines")
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=8)

        save_path = save_folder / f"{campaign}_lines.pdf"

        plt.savefig(save_path)

        plt.figure()
        ax2 = axs[1]

        for continuum, data in light_curve_data["continua"].items():
            x = data[xlabel]
            y = data[ylabel]
            y_err = data[yerr_name]

            # Normierung der y-Werte
            max_y = max(y)
            y = y / max_y
            y_err = y_err / max_y  # Fehler ebenfalls normieren

            ax2.errorbar(x, y, y_err, label=continuum)
            plt.errorbar(x, y, y_err, label=continuum)

        ax2.set_xlabel(xlabel)
        # ax2.set_ylabel(ylabel)
        ax2.set_title(f"1D Lightcurves of continua")
        ax2.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(fontsize=8)

        plt.xlabel(xlabel)
        plt.ylabel("fluxes (normalized)")
        plt.title(f"1D Lightcurves of lines")
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        plt.legend( fontsize=8)

        save_path = save_folder / f"{campaign}_continua.pdf"

        plt.savefig(save_path)

        fig_line_continua.suptitle(f"Lightcurves of lines and continua of the {super_title}")
        fig_line_continua.tight_layout(rect=(0, 0, 1, 0.95))

        save_path = save_folder / f"{campaign}.pdf"

        fig_line_continua.savefig(save_path)

        if not save_only:
            plt.show()

        plt.close(fig_line_continua)

