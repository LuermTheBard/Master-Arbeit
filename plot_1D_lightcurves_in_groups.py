import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter

from import_data import import_1d_lightcurve_data
from plot_utils import calculate_standard_error_for_lightcurves, format_label, ensure_output_dir
from general_plot import prepare_data, finalize_figure, format_relative_days, format_month_day
from settings import BASE_MJD, COLORCODE_CONTINUA_NORMALIZED, DEFAULT_OUTPUT_DIR, SYMBOLES_AND_COLORS_FOR_LIGHTCURVES


def plot_all_1d_lightcurves_in_groups(data_dict, campaign, output_dir, compare_cont,
                                      key_order_lines=None, key_order_conts=None, save_only=False):
    """
    Plots all available 1D lightcurves (emission lines and continua) in grouped subplots.

    Emission line lightcurves are plotted together with the selected comparison continuum.
    Continuum lightcurves are plotted separately using a custom color dictionary.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with structure {"lines": {line_name: dict}, "continua": {cont_name: dict}}.
    campaign : str
        Campaign identifier (used in output directory structure and titles).
    output_dir : str or Path
        Base output directory for saving plots.
    compare_cont : str
        Continuum name to be used for comparison in line plots.
    key_order_lines : list of str, optional
        Custom order for line lightcurve plotting.
    key_order_conts : list of str, optional
        Custom order for continuum plotting.
    save_only : bool, optional
        If True, saves the plots without displaying them.

    Returns:
    -----------
    None
    """

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
        if compare_cont == "UVW2":
            compare_cont_data = {compare_cont: data_dict["lines"][compare_cont]}
        else:
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

    if campaign == "NGC4593_optical_calibrated":
        all_cont_dict = data_dict["continua"] | {"UVW2": data_dict["lines"]["UVW2"]}
    else:
        all_cont_dict = data_dict["continua"]


    sorted_cont_data_dict = dict(
        sorted(all_cont_dict.items(), key=lambda item: sort_keys(item[0], key_order_conts)))

    plot_lightcurves_in_groups(sorted_cont_data_dict, x_key, y_key, compare_cont, xlabel, ylabel_cont, yerr_name=yerr_name,
                           title=super_title, save_only=save_only, output_dir=save_folder,
                           color_dict=COLORCODE_CONTINUA_NORMALIZED)


def plot_lightcurves_in_groups(data, x_key, y_key, compare_cont, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                               yerr_name=None, title=None, save_only=False,
                               output_dir=None, color_dict=None, rows=4, cols=2, line_light_curves=False):
    """
    Plots lightcurves grouped in subplots with optional error bars and custom layout.

    Supports both continuum and emission line lightcurves, and handles unit normalization for flux values.

    Parameters:
    -----------
    data : dict
        Dictionary with {name: lightcurve_dict}. Each lightcurve_dict must include `x_key`, `y_key`, and optionally `yerr_name`.
    x_key : str
        Key for time (x-axis) values in each lightcurve dictionary.
    y_key : str
        Key for flux (y-axis) values.
    compare_cont : str
        Reference continuum name, used for naming output and labels.
    xlabel : str, optional
        Label for the x-axis. Default is 'X-axis'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Y-axis'.
    shared_y : bool, optional
        If True, subplots share y-axis limits.
    yerr_name : str, optional
        Key for flux error values. If None, no error bars will be plotted.
    title : str, optional
        Figure title.
    save_only : bool, optional
        If True, saves the plot instead of displaying it.
    output_dir : str or Path, optional
        Output directory for plot files.
    color_dict : dict, optional
        Optional color override for individual lines.
    rows : int, optional
        Number of subplot rows per group. Default is 4.
    cols : int, optional
        Number of subplot columns per group. Default is 2.
    line_light_curves : bool, optional
        If True, adjusts axis labeling for emission line lightcurves.

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


def configure_lightcurves_axis(ax, row, col, ylabel, color, x_values, y_values, yerr_values,
                                line_name, line_lightcurves=False):
    """
    Configures a subplot axis for a single lightcurve, including error bars, labels, ticks, and legends.

    Supports both continuum and line lightcurves and applies formatting for relative time axes and LaTeX-ready labels.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to configure.
    row : int
        Row index of the subplot.
    col : int
        Column index of the subplot.
    ylabel : str
        Y-axis label (with optional unit scaling).
    color : str
        Line color.
    x_values : np.ndarray
        Time values (typically MJD).
    y_values : np.ndarray
        Normalized flux values.
    yerr_values : np.ndarray or None
        Flux uncertainty values for error bars.
    line_name : str
        Name of the plotted component (used in legend).
    line_lightcurves : bool, optional
        Whether the plotted data are emission line lightcurves (used for axis labeling).

    Returns:
    -----------
    None
    """
    # --- NEU: Stil aus Mapping holen, Parameter nur als Fallback nutzen ---
    _style = SYMBOLES_AND_COLORS_FOR_LIGHTCURVES.get(line_name, {})
    _marker = _style.get("marker", _style.get("symbole", "."))   # 'symbole' unterstützen
    _color  = _style.get("color", color)
    _ms     = _style.get("markersize", 4)
    _alpha  = _style.get("alpha", None)

    if x_values.size > 0 and y_values.size > 0:
        if yerr_values is not None:
            ax.errorbar(
                format_relative_days(x_values),
                y_values,
                yerr=yerr_values,
                fmt=f'{_marker}:',            # war '.:' → jetzt Marker aus Mapping
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
                markersize=_ms,
                label=f'{format_label(line_name, as_latex=False)}',
                color=_color,
                **({"alpha": _alpha} if _alpha is not None else {})
            )
        else:
            ax.plot(
                format_relative_days(x_values),
                y_values,
                marker=_marker,               # zusätzlich Marker setzen
                label=f'{format_label(line_name, as_latex=False)}',
                color=_color,
                markersize=_ms,
                **({"alpha": _alpha} if _alpha is not None else {})
            )

        ax.legend(fontsize=7.5, loc='upper right')

    if col == 0:
        if row == 0 and line_lightcurves:
            ylabel = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{\AA}^{-1}]$")

            exponent_value = -15
            latex_exponent = f"10^{{{exponent_value}}}"

            ylabel_parts = ylabel.split("[")
            new_ylabel = ylabel_parts[0] + f"[{latex_exponent} " + ylabel_parts[1]
            ax.set_ylabel(new_ylabel, fontsize=12)
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



def run_1d_lightcurves_groups(output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    ensure_output_dir(output_dir)
    data = import_1d_lightcurve_data()
    #for cont in ["Cont1150_not_optical_calibrated", "UVW2"]:
     #   key_order_lines = [cont, 'HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeII4685', 'OI8446']

      #  key_order_conts = ["UVW2", "Cont1150_not_optical_calibrated", "Cont4010", "Cont4440", "Cont5100", "Cont6110", "Cont6880", "Cont8015"] #, "Cont8900"]
       # for campaign, data_dict in data.items():
        #    plot_all_1d_lightcurves_in_groups(data_dict, campaign, output_dir, compare_cont=cont, key_order_lines=key_order_lines, key_order_conts=key_order_conts, save_only=save_only)
    key_order_uv_lines = ["UVW2", 'LyAlpha_not_optical_calibrated', 'NV1238_not_optical_calibrated',
                          'SiIV1393_not_optical_calibrated', 'CIV1548_not_optical_calibrated',
                          'HeII1640_not_optical_calibrated']

    plot_all_1d_lightcurves_in_groups(data["NGC4593_not_optical_calibrated"],"NGC4593_not_optical_calibrated", output_dir, compare_cont="UVW2", key_order_lines=key_order_uv_lines, key_order_conts=[], save_only=save_only)

# methods to run


def plot_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    run_1d_lightcurves_groups(output_dir)


def save_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    run_1d_lightcurves_groups(output_dir, save_only=True)


#plot_1d_lightcurves_in_groups()
save_1d_lightcurves_in_groups()