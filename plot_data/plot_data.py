from pathlib import Path

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


def get_all_line_names(plot_data_1d_correlation):
    """
    Gathers all unique line names from the provided plot data.

    Args:
        plot_data_1d_correlation (dict): Dictionary containing campaign names as keys
                                         and their respective plot data as values.

    Returns:
        set: A set of all unique line names across campaigns.
    """
    all_line_names = set()
    for campaign, plot_data in plot_data_1d_correlation.items():
        all_line_names.update(plot_data.keys())
    return all_line_names


def collect_comparison_data(plot_data_1d_correlation, current_line_name):
    """
    Collects plot data for the specified line across campaigns.

    Args:
        plot_data_1d_correlation (dict): Dictionary containing campaign names as keys
                                         and their respective plot data as values.
        current_line_name (str): The name of the line to collect data for.

    Returns:
        list: A list of tuples, each containing a campaign name and its associated plot data for the line.
    """
    comparison_data = []
    for campaign, plot_data in plot_data_1d_correlation.items():
        all_lines = {key.casefold(): key for key in plot_data.keys()}
        normalized_line_name = current_line_name.casefold()
        if normalized_line_name in all_lines:
            line_data = plot_data[all_lines[normalized_line_name]]
            comparison_data.append((campaign, line_data))
    return comparison_data


def plot_individual(data_list, current_line_name, campaign, output_dir, save_only):
    """
    Creates and optionally saves or displays individual plots for a specific line in a campaign.

    Args:
        data_list (list): List of data points for the line. Each entry contains a label, x-values, and y-values.
        current_line_name (str): Name of the line being plotted.
        campaign (str): The name of the campaign for which the plot is generated.
        output_dir (Path or None): Directory where the plot will be saved. If None, plots are not saved.
        save_only (bool): If True, the plot will only be saved and not displayed.
    """
    plt.figure(figsize=(10, 6))
    x_min, x_max, y_min, y_max = float("inf"), float("-inf"), float("inf"), float("-inf")

    for data in data_list:
        x, y = data[1], data[2]
        x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
        y_min, y_max = min(y_min, min(y)), max(y_max, max(y))
        plt.plot(x, y, label=f"{data[0]}")

    plt.xlabel("time shift (tau)")
    plt.ylabel("Correlation Coefficient")
    plt.title(f"Comparison for Line: {current_line_name} ({campaign})")
    plt.xticks(range(int(x_min), int(x_max) + 2, 1))
    plt.yticks([round(i / 10, 2) for i in range(int(y_min * 10 - 2), int(y_max * 10 + 2), 1)])
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    if output_dir:
        save_path = prepare_output_path(output_dir, campaign, current_line_name, is_combined=False)
        plt.savefig(save_path)

    if not save_only:
        plt.show()

    plt.close()


def plot_combined(comparison_data, current_line_name, output_dir, save_only):
    """
    Creates and optionally saves or displays combined plots for a specific line across campaigns.

    Args:
        comparison_data (list): List of tuples, each containing campaign names and their respective data points.
        current_line_name (str): Name of the line being plotted.
        output_dir (Path or None): Directory where the plot will be saved. If None, plots are not saved.
        save_only (bool): If True, the plot will only be saved and not displayed.
    """
    num_campaigns = len(comparison_data)
    fig_combined, axs_combined = plt.subplots(1, num_campaigns, figsize=(15, 6), sharey=True)
    axs_combined = axs_combined if num_campaigns > 1 else [axs_combined]

    for i, (campaign, data_list) in enumerate(comparison_data):
        ax = axs_combined[i]
        x_min, x_max, y_min, y_max = float("inf"), float("-inf"), float("inf"), float("-inf")

        for data in data_list:
            x, y = data[1], data[2]
            x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
            y_min, y_max = min(y_min, min(y)), max(y_max, max(y))
            ax.plot(x, y, label=f"{data[0]}")

        ax.set_xlabel("time shift (tau)")
        if i == 0:
            ax.set_ylabel("Correlation Coefficient")
        ax.set_title(f"Comparison: {campaign}")
        ax.set_xticks(range(int(x_min), int(x_max) + 2, 1))
        ax.set_yticks([round(i / 10, 2) for i in range(int(y_min * 10 - 2), int(y_max * 10 + 2), 1)])
        ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)

    fig_combined.suptitle(f"Comparison for Line: {current_line_name}")
    fig_combined.tight_layout(rect=(0, 0, 1, 0.95))

    if output_dir:
        save_path = prepare_output_path(output_dir, None, current_line_name, is_combined=True)
        fig_combined.savefig(save_path)

    if not save_only:
        plt.show()

    plt.close(fig_combined)


def prepare_output_path(output_dir, campaign, line_name, is_combined):
    """
    Prepares the output path for saving plots.

    Args:
        output_dir (Path): Base directory for saving plots.
        campaign (str or None): Campaign name for individual plots. None for combined plots.
        line_name (str): Name of the line being plotted.
        is_combined (bool): If True, the plot is for combined campaigns; otherwise, for individual campaigns.

    Returns:
        Path: The full path for the output file.
    """
    sub_dir = "combined" if is_combined else campaign
    comparison_dir = Path(output_dir) / "plot_data" / "compare_plots_across_continua" / sub_dir
    comparison_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{line_name.replace(' ', '_')}_{'combined' if is_combined else campaign}_comparison.png"
    return comparison_dir / file_name


def compare_plots_across_continua(
    plot_data_1d_correlation,
    line_name=None,
    output_dir=None,
    save_only=False,
):
    """
    Main function to compare plots for specific lines across multiple campaigns.

    Args:
        plot_data_1d_correlation (dict): Dictionary containing campaign names as keys
                                         and their respective plot data as values.
        line_name (str or None): Specific line to compare. If None, all lines are compared.
        output_dir (Path or None): Directory where plots will be saved. If None, plots are not saved.
        save_only (bool): If True, plots will only be saved and not displayed.
    """
    # Gather all lines if none is specified
    all_line_names = get_all_line_names(plot_data_1d_correlation)
    lines_to_compare = [line_name] if line_name else all_line_names

    for current_line_name in lines_to_compare:
        comparison_data = collect_comparison_data(plot_data_1d_correlation, current_line_name)
        if not comparison_data:
            print(f"No data found for line '{current_line_name}'.")
            continue

        for campaign, data_list in comparison_data:
            plot_individual(data_list, current_line_name, campaign, output_dir, save_only)

        plot_combined(comparison_data, current_line_name, output_dir, save_only)

