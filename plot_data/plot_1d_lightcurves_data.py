import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")


def plot_lightcurve(data, xlabel, ylabel, yerr_name, ax, label_prefix):
    """
    Plots a single lightcurve with normalized flux values.
    """
    for key, values in data.items():
        x = values[xlabel]
        y = values[ylabel]
        y_err = values[yerr_name]

        # Normierung der y-Werte
        max_y = max(y)
        y = y / max_y
        y_err = y_err / max_y  # Fehler ebenfalls normieren

        ax.errorbar(x, y, y_err, label=f"{label_prefix}: {key}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("fluxes (normalized)")
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8)


def save_and_show_plot(fig, save_path, save_only):
    """
    Saves the figure and optionally shows it.
    """
    fig.savefig(save_path)
    if not save_only:
        plt.show()
    plt.close(fig)


def plot_1d_lightcurves(galaxie_campaigns_dict, output_dir, save_only=False):
    xlabel = 'timestamps [MJD]'
    ylabel = 'fluxes [ergs/s/cm2/A]'
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    for campaign, light_curve_data in galaxie_campaigns_dict.items():
        super_title = campaign

        # Einzelne Plots: Linien
        fig_lines, ax_lines = plt.subplots(figsize=(8, 6))
        plot_lightcurve(light_curve_data["lines"], xlabel, ylabel, yerr_name, ax_lines, "Line")
        ax_lines.set_title("1D Lightcurves of lines")
        save_and_show_plot(fig_lines, save_folder / f"{campaign}_lines.pdf", save_only)

        # Einzelne Plots: Kontinua
        fig_continua, ax_continua = plt.subplots(figsize=(8, 6))
        plot_lightcurve(light_curve_data["continua"], xlabel, ylabel, yerr_name, ax_continua, "Continuum")
        ax_continua.set_title("1D Lightcurves of continua")
        save_and_show_plot(fig_continua, save_folder / f"{campaign}_continua.pdf", save_only)

        # Kombinierter Plot: Linien und Kontinua
        fig_combined, axs_combined = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        plot_lightcurve(light_curve_data["lines"], xlabel, ylabel, yerr_name, axs_combined[0], "Line")
        axs_combined[0].set_title("1D Lightcurves of lines")

        plot_lightcurve(light_curve_data["continua"], xlabel, ylabel, yerr_name, axs_combined[1], "Continuum")
        axs_combined[1].set_title("1D Lightcurves of continua")

        fig_combined.suptitle(f"Lightcurves of lines and continua of {super_title}")
        fig_combined.tight_layout(rect=(0, 0, 1, 0.95))
        save_and_show_plot(fig_combined, save_folder / f"{campaign}_combined.pdf", save_only)
