import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter

from handle_data.handle_data_file import calculate_standard_error_for_lightcurves, format_label
from plot_data.general_plot import prepare_data, finalize_figure, format_relative_days, format_yaxis, format_month_day
from settings import BASE_MJD, COLORCODE_CONTINUA_NORMALIZED


def plot_all_1d_lightcurves_in_groups(data_dict, campaign, output_dir, compare_cont, key_order_lines=None, key_order_conts=None, save_only=False):
    base_mjd = BASE_MJD

    xlabel = f"MJD - {base_mjd:.2f}"
    ylabel_cont = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
             r"\mathrm{\AA}^{-1}]$")
    ylabel_line = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1}]$")
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'



    x_key = 'timestamps [MJD]'
    y_key = 'fluxes [ergs/s/cm2/A]'




    save_folder = output_dir / campaign / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Plot for lines
    super_title = f"{campaign.split('_')[0]} Lines"

    try:
        compare_cont_data = {compare_cont: data_dict["continua"][compare_cont]}
        compare_cont_data.update(data_dict["lines"])
    except KeyError:
        print(f"Continuum '{compare_cont}' not found in campaign '{campaign}'. Skipping plot for lines.")
        return

    def sort_keys(key, key_order):
        for idx, prefix in enumerate(key_order):
            if key.startswith(prefix):
                return idx
        return len(key_order)

    sorted_line_data_dict = dict(sorted(compare_cont_data.items(), key=lambda item: sort_keys(item[0], key_order_lines)))

    plot_lightcurves_in_groups(sorted_line_data_dict, x_key, y_key, compare_cont, xlabel, ylabel_line, yerr_name=yerr_name, title=super_title,
                           save_only=save_only, output_dir=save_folder, line_light_curves=True)

    # Plot for continua (with custom color dictionary if needed)
    super_title = f"{campaign.split('_')[0]} Continua"

    sorted_cont_data_dict = dict(
        sorted(data_dict["continua"].items(), key=lambda item: sort_keys(item[0], key_order_conts)))

    plot_lightcurves_in_groups(sorted_cont_data_dict, x_key, y_key, compare_cont, xlabel, ylabel_cont, yerr_name=yerr_name,
                           title=super_title, save_only=save_only, output_dir=save_folder,
                           color_dict=COLORCODE_CONTINUA_NORMALIZED)


def plot_lightcurves_in_groups(data, x_key, y_key, compare_cont, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                               yerr_name=None, title=None, save_only=False,
                               output_dir=None, color_dict=None, rows=4, cols=2, line_light_curves=False):
    """
    Plots 1D-Daten in Gruppen für Lightcurves.

    Parameter:
    -----------
    data : dict
        Dictionary containing the data for the plots.
    x_key : str
        Key für die x-Daten in den Dictionaries der Daten.
    y_key : str
        Key für die y-Daten in den Dictionaries der Daten.
    xlabel : str, optional
        Label für die X-Achse (nicht immer verwendet).
    ylabel : str, optional
        Label für die Y-Achse.
    shared_y : bool, optional
        Wenn True, teilen sich alle Subplots die Y-Achse.
    yerr_name : str, optional
        Key für die Fehlerbalken im Dictionary.
    title : str, optional
        Titel für die gesamte Figure.
    save_only : bool, optional
        Ob die Abbildungen gespeichert werden sollen, ohne sie anzuzeigen.
    output_dir : str or Path, optional
        Verzeichnis, in dem die Abbildungen gespeichert werden.
    color_dict : dict, optional
        Dictionary mit Farben für jede Datenserie.
    rows : int, optional
        Anzahl der Subplot-Reihen.
    cols : int, optional
        Anzahl der Subplot-Spalten.

    Returns:
    -----------
    None
    """
    for current_data, group_index in prepare_data(data, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=True, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            if line_data:
                x_values = np.array(line_data.get(x_key, []))
                y_values = np.array(line_data.get(y_key, []))
                yerr_noise_values = np.array(line_data.get(yerr_name, [])) if yerr_name else None

                yerr_values = calculate_standard_error_for_lightcurves(y_values, yerr_noise_values)

            else:
                x_values = np.array([])
                y_values = np.array([])
                yerr_values = np.array([])

            color = color_dict.get(line_name, 'black') if color_dict else 'black'

            exponent_value = -15
            exponent = 10 ** exponent_value
            latex_exponent = f"10^{{{exponent_value}}}"

            ylabel_parts = ylabel.split("[")
            new_ylabel = ylabel_parts[0] + f"[{latex_exponent} " + ylabel_parts[1]

            y_values = y_values / exponent
            yerr_values = yerr_values / exponent

            configure_lightcurves_axis(ax, row, col, new_ylabel, color, x_values, y_values, yerr_values, line_name, line_light_curves)

        finalize_figure(fig, axes, x_label=xlabel, title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir, compare_cont=compare_cont)


def configure_lightcurves_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values, line_name, line_lightcurves=False):
    """
    Konfiguriert die Achse für Lightcurves.

    Parameter:
    -----------
    ax : matplotlib.axes.Axes
        Die jeweilige Achse, auf der geplottet wird.
    row : int
        Zeilenindex des Subplots.
    col : int
        Spaltenindex des Subplots.
    ylabel : str
        Beschriftung der Y-Achse.
    color : str
        Linienfarbe.
    x_values : np.ndarray
        X-Daten für den Plot.
    y_values : np.ndarray
        Y-Daten für den Plot.
    yerr_values : np.ndarray or None
        Fehlerbalkendaten, falls vorhanden.
    line_name : str
        Name / Label für die Datenlinie.

    Returns:
    -----------
    None
    """
    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(
                format_relative_days(x_values),
                y_values,
                yerr=yerr_values,
                fmt='.:', capsize=3, markersize=4, label=f'{format_label(line_name, as_latex=False)}', color=color
            )
        else:
            ax.plot(
                format_relative_days(x_values),
                y_values, label=f'{format_label(line_name, as_latex=False)}', color=color
            )

        ax.legend(fontsize=7.5, loc='upper right')

    if col == 0:
        if row == 0 and line_lightcurves:
            ax.set_ylabel((r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{\AA}^{-1}]$"), fontsize=12)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)


    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.set_label_coords(1.15, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=10)
