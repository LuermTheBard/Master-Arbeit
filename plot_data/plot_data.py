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


def compare_plots_across_continua(
    plot_data_1d_correlation,
    line_name=None,
    output_dir=None,
    save_only=False,
    comparison_dir_name="comparisons"
):
    """
    Vergleicht gleichnamige Plots aus verschiedenen Super-Titeln nebeneinander und speichert sie optional.

    Args:
        plot_data_1d_correlation (dict): Plot-Daten, geordnet nach Kampagne (Super-Titeln).
        line_name (str, optional): Die Linie, die verglichen werden soll. Wenn None, werden alle Linien verglichen.
        output_dir (str, optional): Verzeichnis, in dem die Vergleichsplots gespeichert werden.
        save_only (bool, optional): Wenn True, werden die Plots nur gespeichert und nicht angezeigt.
        comparison_dir_name (str, optional): Unterverzeichnis für Vergleichsplots. Standard: "comparisons".
    """

    # Sammeln aller verfügbaren Linien, wenn kein spezifischer line_name angegeben ist
    all_line_names = set()
    for campaign, plot_data in plot_data_1d_correlation.items():
        all_line_names.update(plot_data.keys())

    # Wenn kein `line_name` angegeben ist, vergleiche alle Linien
    lines_to_compare = [line_name] if line_name else all_line_names

    # Iteriere über die zu vergleichenden Linien
    for current_line_name in lines_to_compare:
        comparison_data = []
        for campaign, plot_data in plot_data_1d_correlation.items():
            all_lines = {key.casefold(): key for key in plot_data.keys()}
            normalized_line_name = current_line_name.casefold()
            if normalized_line_name in all_lines:
                line_data = plot_data[all_lines[normalized_line_name]]
                comparison_data.append((campaign, line_data))

        if not comparison_data:
            print(f"Keine Daten für die Linie '{current_line_name}' in den bereitgestellten Super-Titeln gefunden.")
            continue

        # Plotten der Vergleichsdaten

        # Anzahl der Subplots für die kombinierte Figur bestimmen
        num_campaigns = len(comparison_data)

        # Kombinierte Subplots erstellen (1 Zeile, num_campaigns Spalten)
        fig_combined, axs_combined = plt.subplots(1, num_campaigns, figsize=(15, 6), sharey=True)
        if num_campaigns == 1:  # Falls nur eine Kampagne vorhanden ist, axs als Liste behandeln
            axs_combined = [axs_combined]

        # Iteration durch die Kampagnen
        for i, (campaign, data_list) in enumerate(comparison_data):
            # Einzelne Figur erstellen
            plt.figure(figsize=(10, 6))
            for data in data_list:
                x = data[1]
                y = data[2]
                plt.plot(x, y, label=f"{data[0]}")

            plt.xlabel("time shift (tau)")
            plt.ylabel("Correlation Coefficient")
            plt.title(f"Vergleich für Linie: {current_line_name} ({campaign})")
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)

            # Optional speichern
            if output_dir:
                campaign_dir = Path(output_dir) / campaign / comparison_dir_name
                campaign_dir.mkdir(parents=True, exist_ok=True)
                save_path = campaign_dir / f"{current_line_name.replace(' ', '_')}_{campaign}_comparison.png"
                plt.savefig(save_path)

            if not save_only:
                plt.show()

            # Schließen der einzelnen Figur
            plt.close()

            # Daten in kombinierte Subplots einfügen
            ax = axs_combined[i]
            for data in data_list:
                x = data[1]
                y = data[2]
                ax.plot(x, y, label=f"{data[0]}")

            ax.set_xlabel("time shift (tau)")
            if i == 0:
                ax.set_ylabel("Correlation Coefficient")  # Nur beim ersten Plot Y-Achse beschriften
            ax.set_title(f"Vergleich: {campaign}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True)

        # Gemeinsamen Titel für die kombinierte Figur setzen
        fig_combined.suptitle(f"Vergleich für Linie: {current_line_name}")

        # Layout optimieren
        fig_combined.tight_layout(rect=[0, 0, 1, 0.95])

        # Kombinierte Figur speichern
        if output_dir:
            comparison_dir = Path(output_dir) / comparison_dir_name
            comparison_dir.mkdir(parents=True, exist_ok=True)
            combined_save_path = comparison_dir / f"{current_line_name.replace(' ', '_')}_combined_comparison.png"
            fig_combined.savefig(combined_save_path)

        # Kombinierte Figur anzeigen, falls nicht nur speichern
        if not save_only:
            plt.show()

        # Schließen der kombinierten Figur
        plt.close(fig_combined)

