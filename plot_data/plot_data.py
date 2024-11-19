import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")


def simple_plot(params):
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
    plt.show()


def plot_1d_correlation_data(data_list, line, super_title):
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
        simple_plot(params)


def plot_1d_correlations(plot_data_1d_correlation, line_name=None):
    """
    Plots 1D correlations for all lines or a specific line.

    Args:
        plot_data_1d_correlation (dict): The plot data dictionary.
        line_name (str, optional): The specific line name to plot. If None, plots all lines.

    Raises:
        ValueError: If line_name is specified but does not exist in the plot data keys.
    """
    for campaign, plot_data in plot_data_1d_correlation.items():
        super_title = campaign

        if line_name and line_name not in plot_data.keys():
            available_lines = ", ".join(plot_data.keys())
            raise ValueError(
                f"The specified line_name '{line_name}' does not exist in the plot data for campaign '{campaign}'. "
                f"Available lines are: {available_lines}."
            )

        for line, data_list in plot_data.items():
            if not line_name or line == line_name:
                plot_1d_correlation_data(data_list, line, super_title)

