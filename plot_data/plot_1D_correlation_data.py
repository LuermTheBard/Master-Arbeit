from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from handle_data.handle_data import gaussian_with_baseline

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
        save_path = prepare_output_path(output_dir, campaign, current_line_name, "plot_individual", is_combined=False)
        plt.savefig(save_path)

    if not save_only:
        plt.show()

    plt.close()


def plot_combined(comparison_data, current_line_name, output_dir, save_only, show_average=False,
                  show_only_average=False):
    """
    Creates and optionally saves or displays combined plots for a specific line across campaigns,
    with an option to show only the average of the data points.

    Args:
        show_only_average:
        comparison_data (list): List of tuples, each containing campaign names and their respective data points.
        current_line_name (str): Name of the line being plotted.
        output_dir (Path or None): Directory where the plot will be saved. If None, plots are not saved.
        save_only (bool): If True, the plot will only be saved and not displayed.
        show_average (bool): If True, only the average of the data points will be displayed.
    """
    num_campaigns = len(comparison_data)
    fig_combined, axs_combined = plt.subplots(1, num_campaigns, figsize=(15, 6), sharey=True)
    axs_combined = axs_combined if num_campaigns > 1 else [axs_combined]

    for i, (campaign, data_list) in enumerate(comparison_data):
        ax = axs_combined[i]
        x_min, x_max, y_min, y_max = float("inf"), float("-inf"), float("inf"), float("-inf")

        all_x = []
        all_y = []

        for data in data_list:
            x, y = data[1], data[2]
            all_x.append(x)
            all_y.append(y)
            x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
            y_min, y_max = min(y_min, min(y)), max(y_max, max(y))

            if not show_only_average:
                if show_average:
                    linestyle = '--'
                else:
                    linestyle = "-"
                ax.plot(x, y, label=f"{data[0]}", linestyle=linestyle)

        # Compute and plot average if show_average is True
        if show_average or show_only_average:
            avg_x = np.linspace(x_min, x_max, 500)  # Interpolated x for consistent averaging
            avg_y = np.zeros_like(avg_x)
            count = np.zeros_like(avg_x)

            for x, y in zip(all_x, all_y):
                y_interp = np.interp(avg_x, x, y)  # Interpolate y values to avg_x
                avg_y += y_interp
                count += 1

            avg_y /= count
            ax.plot(avg_x, avg_y, label="Average over continua", color="blue", linewidth=2)

        ax.set_xlabel("time shift (tau)")
        if i == 0:
            ax.set_ylabel("Correlation Coefficient")
        ax.set_title(f"Comparison: {campaign}")
        ax.set_xticks(range(int(x_min), int(x_max) + 2, 1))
        ax.set_yticks([round(i / 10, 2) for i in range(int(y_min * 10 - 2), int(y_max * 10 + 2), 1)])
        ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)

    fig_combined.suptitle(f"{'Average ' if show_average else ''}Comparison for Line: {current_line_name}")
    fig_combined.tight_layout(rect=(0, 0, 1, 0.95))

    if output_dir:
        if show_average:
            methode_name = "plot_combined_with_average"
        elif show_only_average:
            methode_name = "plot_average"
        else:
            methode_name = "plot_combined"

        save_path = prepare_output_path(output_dir, None,
                                        current_line_name,
                                        methode_name,
                                        is_combined=True)
        fig_combined.savefig(save_path)

    if not save_only:
        plt.show()

    plt.close(fig_combined)


def prepare_output_path(output_dir, campaign, line_name, methode_name, is_combined):
    """
    Prepares the output path for saving plots.

    Args:
        methode_name: 
        output_dir (Path): Base directory for saving plots.
        campaign (str or None): Campaign name for individual plots. None for combined plots.
        line_name (str): Name of the line being plotted.
        is_combined (bool): If True, the plot is for combined campaigns; otherwise, for individual campaigns.

    Returns:
        Path: The full path for the output file.
    """
    comparison_dir = Path(output_dir) / "plot_1d_correlations" / methode_name
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

        plot_combined(comparison_data, current_line_name, output_dir, save_only, show_average=True)
        plot_combined(comparison_data, current_line_name, output_dir, save_only, show_only_average=True)


def plot_fit_results(campaign, fit_results):
    """
    Erstellt einen Plot der Fit-Daten einschließlich des Fits, markiert den Time Lag
    und zeigt die Grenzen des Fensters für den Fit an.

    Parameters:
        fit_results (list): Eine Liste von Dictionaries, die die Fit-Daten enthalten.
                            Jedes Dictionary sollte mindestens die Schlüssel "x_values", "y_values",
                            "time_lag", "fit_window_start", "fit_window_end", "amplitude",
                            "std_dev", "baseline" und "fit_success" enthalten.
    """
    for result in fit_results:
        if not result.get("fit_success", False):
            print(f"Fit für {result['continuum']} fehlgeschlagen.")
            continue

        # Extrahieren der Daten
        line_name = result["line_name"]
        x_values = np.array(result["x_values"])
        y_values = np.array(result["y_values"])
        time_lag = result["time_lag"]
        amplitude = result["amplitude"]
        std_dev = result["std_dev"]
        baseline = result["baseline"]
        fit_window_start = result["fit_window_start"]
        fit_window_end = result["fit_window_end"]

        # Erstellen der Fit-Kurve
        x_fit = np.linspace(x_values.min(), x_values.max(), 500)
        y_fit = gaussian_with_baseline(x_fit, amplitude, time_lag, std_dev, baseline)

        # Plot erstellen
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, '-', label="Data", markersize=5)
        plt.plot(x_fit, y_fit, '--', label="Gaussian Fit")
        plt.axvline(time_lag, color='red', linestyle='--', label=f"Time Lag (τ) = {time_lag:.2f}")
        plt.axvline(fit_window_start, color='blue', linestyle='--', label="Fit Window Start")
        plt.axvline(fit_window_end, color='green', linestyle='--', label="Fit Window End")

        # Achsenbeschriftungen und Titel
        plt.xlabel("Time Shift (τ)", fontsize=12)
        plt.ylabel("Correlation Coefficient", fontsize=12)
        plt.title(f"{campaign}\n\nFit for {line_name} and {result['continuum']}", fontsize=14)

        # Gitter und Legende
        plt.grid(True)
        plt.legend()

        # Plot anzeigen
        plt.show()
