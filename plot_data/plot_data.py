import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")


def process_plot(params, save_path=None, save_only=False):
    """
    Erstellt eine Grafik basierend auf den übergebenen Parametern und speichert sie optional.

    Args:
        params (dict): Parameter für die Grafik.
        save_path (str, optional): Pfad, unter dem die Grafik gespeichert wird.
        save_only (bool, optional): Wenn True, wird die Grafik nur gespeichert und nicht angezeigt.
    """
    plt.figure()
    plt.plot(params["x"], params["y"], label=params["label"], color="b", linewidth=2)
    plt.xlabel(params["xlabel"], fontsize=params["label_fontsize"])
    plt.ylabel(params["ylabel"], fontsize=params["label_fontsize"])
    plt.suptitle(params["super_title"], fontsize=params["title_fontsize"] * 1.1)
    plt.title(params["title"], fontsize=params["title_fontsize"])
    plt.legend(loc=params["legend_loc"], fontsize=params["legend_fontsize"])
    plt.xlim(params["xlim"])
    plt.ylim(params["ylim"])
    plt.xticks(params["x_ticks"])
    plt.yticks(params["y_ticks"])
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    if not save_only:
        plt.show()

    plt.close()


def process_1d_correlation_data(data_list, line, super_title, output_dir=None, save_only=False):
    """
    Erstellt oder speichert 1D-Korrelationsdaten als Bilder.

    Args:
        data_list (list): Liste von Daten für die Korrelation.
        line (str): Linienbezeichnung.
        super_title (str): Übergeordneter Titel der Grafik.
        output_dir (str, optional): Verzeichnis, in dem die Grafiken gespeichert werden.
        save_only (bool, optional): Wenn True, werden die Plots nur gespeichert und nicht angezeigt.
    """
    xlabel = "time shift (tau)"
    ylabel = "Correlation Coefficient"
    label_fontsize = 12
    title_fontsize = 14
    legend_fontsize = 10
    legend_loc = "upper right"

    for data in data_list:
        title = f"{line} / {data[0]}"
        x = data[1]
        y = data[2]

        label = "1D Correlation"

        ylim = (min(y) - 0.1, max(y) + 0.1)
        xlim = None
        x_ticks = np.arange(min(x), max(x) + 1, 2)
        y_ticks = np.arange(round(min(y) - 0.1, 1), max(y) + 0.1, 0.1)

        params = {
            "x": x,
            "y": y,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "xlim": xlim,
            "ylim": ylim,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "label": label,
            "label_fontsize": label_fontsize,
            "legend_fontsize": legend_fontsize,
            "legend_loc": legend_loc,
            "super_title": super_title,
            "title": title,
            "title_fontsize": title_fontsize
        }

        save_path = None
        if output_dir:
            save_dir = output_dir / super_title
            if not save_dir.exists():
                save_dir.mkdir()
            save_path = f"{save_dir}/{title.replace(' ', '_').replace('/', '_')}.png"

        process_plot(params, save_path=save_path, save_only=save_only)


def process_1d_correlations(plot_data_1d_correlation, output_dir=None, save_only=False, line_name=None):
    """
    Erstellt oder speichert 1D-Korrelationen für alle Linien oder eine spezifische Linie als Bilder.

    Args:
        plot_data_1d_correlation (dict): Plot-Daten.
        output_dir (str, optional): Verzeichnis, in dem die Bilder gespeichert werden.
        save_only (bool, optional): Wenn True, werden die Plots nur gespeichert und nicht angezeigt.
        line_name (str, optional): Spezifische Linie. Wenn None, werden alle Linien verarbeitet.

    Raises:
        ValueError: Wenn line_name angegeben, aber nicht in den Daten vorhanden ist.
    """
    for campaign, plot_data in plot_data_1d_correlation.items():
        super_title = campaign

        all_lines = {key.casefold(): key for key in plot_data.keys()}
        if line_name:
            normalized_line_name = line_name.casefold()
            if normalized_line_name not in all_lines:
                available_lines = ", ".join(all_lines.values())
                raise ValueError(
                    f"The specified line_name '{line_name}' does not exist in the plot data for campaign '{campaign}'. "
                    f"Available lines are: {available_lines}."
                )
            line_name = all_lines[normalized_line_name]

        for line, data_list in plot_data.items():
            if not line_name or line.casefold() == line_name.casefold():
                process_1d_correlation_data(data_list, line, super_title, output_dir=output_dir, save_only=save_only)
