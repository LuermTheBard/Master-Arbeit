import numpy as np
import toml
from matplotlib import pyplot as plt

from settings import DEFAULT_OUTPUT_DIR


def plot_fit_results(campaign, fit_results, output_dir=None, save_only=False):
    """
    Erstellt einen Plot der Fit-Daten einschließlich des Fits, markiert den Time Lag
    und zeigt die Grenzen des Fensters für den Fit an, mit Achsen in 1er- und 0.1-Schritten
    und den x-Werten der Fit-Grenzen in der Legende.

    Parameters:
        save_only:
        output_dir:
        campaign:
        fit_results (list): Eine Liste von Dictionaries, die die Fit-Daten enthalten.
                            Jedes Dictionary sollte mindestens die Schlüssel "x_values", "y_values",
                            "time_lag", "fit_window_start", "fit_window_end" und "fit_success" enthalten.
    """

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / campaign
    else:
        output_dir = output_dir / campaign

    for result in fit_results:
        line_name = result["line_name"]
        if not result.get("fit_success", False):
            print(f"Fit für {campaign}/{line_name}/{result['continuum']} fehlgeschlagen.")
            continue

        # Extrahieren der Daten

        x_values = np.array(result["x_values"])
        y_values = np.array(result["y_values"])
        time_lag = result["time_lag"]

        fit_window_start = result["fit_window_start"]
        fit_window_end = result["fit_window_end"]

        # Plot erstellen
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, '-', label="Data", markersize=5)
        plt.axvline(time_lag, color='red', linestyle='--', label=f"Time Lag (τ) = {time_lag:.2f}")

        plt.axvline(fit_window_start, color='blue', linestyle='--',
                    label=f"Fit Window Start (x = {fit_window_start:.1f})")
        plt.axvline(fit_window_end, color='green', linestyle='--', label=f"Fit Window End (x = {fit_window_end:.1f})")

        # Achsenbeschriftungen und Titel
        plt.xlabel("Time Shift (τ)", fontsize=12)
        plt.ylabel("Correlation Coefficient", fontsize=12)
        plt.title(f"{campaign}\n\nFit for {line_name} and {result['continuum']}", fontsize=14)

        # Achseneinteilung
        plt.xticks(np.arange(np.floor(x_values.min()), np.ceil(x_values.max()) + 1, 1))
        plt.yticks(np.arange(np.floor(y_values.min()), np.ceil(y_values.max()) + 0.1, 0.1))

        # Gitter und Legende
        plt.grid(True)
        plt.legend()

        output_file_dir = output_dir / line_name
        output_file_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_file_dir / f"{result['continuum']}.png"
        # Plot anzeigen
        if save_only:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
            plt.close()


def plot_time_lags_from_toml(file_path):
    # Literaturwerte für H-Alpha bis H-Delta
    literature_wavelengths = {
        "HAlpha": 6563,
        "HBeta": 4861,
        "HGamma": 4341,
        "HDelta": 4102
    }

    # Datei laden
    data = toml.load(file_path)

    # Extrahieren der relevanten Daten aus der TOML-Datei
    wavelengths = []
    time_lags = []
    time_lag_errors = []

    for line_name, entry in data.items():

        if line_name == "skipped_results":
            continue

        time_lag = float(entry[0].get("time_lag"))
        time_lag_error = float(entry[0].get("time_lag_error"))

        if line_name and time_lag is not None and time_lag_error is not None:
            # Extrahiere die Wellenlänge aus den letzten 4 Zeichen des line_name
            try:
                wavelength = int(line_name[-4:])
            except ValueError:
                # Verwende Literaturwerte für bekannte Linien
                wavelength = literature_wavelengths.get(line_name, None)

            if wavelength is not None:
                wavelengths.append(wavelength)
                time_lags.append(time_lag)
                time_lag_errors.append(time_lag_error)

    # Erstellen des Plots
    plt.figure(figsize=(10, 6))
    plt.errorbar(wavelengths, time_lags, yerr=time_lag_errors, fmt='o', capsize=5, ecolor='red', markerfacecolor='blue',
                 linestyle='None')
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Time Lags')
    plt.title('Time Lag vs Wavelength with Errors')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Plot anzeigen
    plt.show()
