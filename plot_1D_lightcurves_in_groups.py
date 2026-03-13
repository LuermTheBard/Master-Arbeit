import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter

from import_data import import_1d_lightcurve_data
from plot_utils import calculate_standard_error_for_lightcurves, format_label, ensure_output_dir
from general_plot import prepare_data, finalize_figure, format_relative_days, format_month_day
from settings import BASE_MJD, COLORCODE_CONTINUA_NORMALIZED, DEFAULT_OUTPUT_DIR, SYMBOLES_AND_COLORS_FOR_LIGHTCURVES


def deep_merge(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def plot_all_1d_lightcurves_in_groups(data_dict, output_dir, compare_cont,
                                      key_order_lines=None, key_order_conts=None, save_only=False, file_name=None):
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

    save_folder = output_dir /  "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    def pick_only_in_order(source_dict, key_order):
        """Return dict with ONLY keys that appear in key_order, in that order (missing keys are skipped)."""
        if not key_order:
            return dict(source_dict)  # fallback: keep all
        return {k: source_dict[k] for k in key_order if k in source_dict}

    # -------------------------
    # Plot for lines (+ compare_cont)
    # -------------------------

    try:
        if compare_cont == "UVW2":
            compare_cont_data = {compare_cont: data_dict["lines"][compare_cont]}
        else:
            compare_cont_data = {compare_cont: data_dict["continua"][compare_cont]}

        # add ALL lines first (we'll filter afterwards)
        compare_cont_data.update(data_dict["lines"])
    except KeyError:
        print(f"Continuum '{compare_cont}' not found. Skipping plot for lines.")
        return

    # IMPORTANT: filter to ONLY those defined in key_order_lines
    filtered_line_data_dict = pick_only_in_order(compare_cont_data, key_order_lines)

    plot_lightcurves_in_groups(
        filtered_line_data_dict, x_key, y_key, compare_cont,
        xlabel, ylabel_line, yerr_name=yerr_name,
        save_only=save_only, output_dir=save_folder, line_light_curves=True,
        file_name=file_name
    )

    # -------------------------
    # Plot for continua
    # -------------------------

    all_cont_dict = data_dict["continua"]

    # IMPORTANT: filter to ONLY those defined in key_order_conts
    filtered_cont_data_dict = pick_only_in_order(all_cont_dict, key_order_conts)

    plot_lightcurves_in_groups(
        filtered_cont_data_dict, x_key, y_key, compare_cont,
        xlabel, ylabel_cont, yerr_name=yerr_name,
        save_only=save_only, output_dir=save_folder,
        color_dict=COLORCODE_CONTINUA_NORMALIZED, file_name=f"{file_name}_cont"
    )


def plot_1d_emission_lines_in_groups(
        data_dict, output_dir, compare_cont,
        key_order_lines=None, save_only=False, file_name=None, rows=5, cols=2, normalize_to_mean=False
):
    base_mjd = BASE_MJD

    xlabel = f"MJD - {base_mjd:.2f}"
    ylabel_line = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1}]$")
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    x_key = 'timestamps [MJD]'
    y_key = 'fluxes [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    def pick_only_in_order(source_dict, key_order):
        if not key_order:
            return dict(source_dict)
        return {k: source_dict[k] for k in key_order if k in source_dict}

    try:
        if compare_cont == "UVW2":
            compare_cont_data = {compare_cont: data_dict["lines"][compare_cont]}
        else:
            compare_cont_data = {compare_cont: data_dict["continua"][compare_cont]}

        compare_cont_data.update(data_dict["lines"])
    except KeyError:
        print(f"Continuum '{compare_cont}' not found. Skipping plot for lines.")
        return

    filtered_line_data_dict = pick_only_in_order(compare_cont_data, key_order_lines)

    plot_lightcurves_in_groups(
        filtered_line_data_dict, x_key, y_key, compare_cont,
        xlabel, ylabel_line, yerr_name=yerr_name,
        save_only=save_only, output_dir=save_folder, line_light_curves=True,
        file_name=file_name, rows=rows, cols=cols
    )


def plot_1d_continua_in_groups(
        data_dict, output_dir, compare_cont,
        key_order_conts=None, save_only=False, file_name=None, rows=5, cols=2,
):
    base_mjd = BASE_MJD

    xlabel = f"MJD - {base_mjd:.2f}"
    ylabel_cont = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                   r"\mathrm{\AA}^{-1}]$")
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    x_key = 'timestamps [MJD]'
    y_key = 'fluxes [ergs/s/cm2/A]'

    save_folder = output_dir / "plot_1d_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    def pick_only_in_order(source_dict, key_order):
        if not key_order:
            return dict(source_dict)
        return {k: source_dict[k] for k in key_order if k in source_dict}

    all_cont_dict = data_dict["continua"] | {"UVW2": data_dict["lines"]["UVW2"]}
    filtered_cont_data_dict = pick_only_in_order(all_cont_dict, key_order_conts)

    plot_lightcurves_in_groups(
        filtered_cont_data_dict, x_key, y_key, compare_cont,
        xlabel, ylabel_cont, yerr_name=yerr_name,
        save_only=save_only, output_dir=save_folder,
        color_dict=COLORCODE_CONTINUA_NORMALIZED,
        file_name=file_name,
        rows=rows, cols=cols
    )






def plot_lightcurves_in_groups(data, x_key, y_key, compare_cont, xlabel='X-axis', ylabel='Y-axis', shared_y=False,
                               yerr_name=None, title=None, save_only=False,
                               output_dir=None, color_dict=None, rows=5, cols=2, line_light_curves=False, file_name=None):
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
    # Anzahl Panels = wie viele Keys du plotten willst
    n_panels = len(data)

    # cols fest (2), rows automatisch, falls nicht explizit übergeben
    if cols is None:
        cols = 2
    if rows is None:
        rows = math.ceil(n_panels / cols)

    # -------- FIXE Panel-Größe (jedes Subplot bleibt gleich groß) --------
    panel_w = 2.1  # inch pro Panel (Breite) -> einmal einstellen
    panel_h = 1.7  # inch pro Panel (Höhe)   -> einmal einstellen

    # Extra Platz für Titel/Labels (einmal einstellen)
    pad_w = 0.8  # links/rechts
    pad_h = 1.4  # oben/unten

    fig_w = cols * panel_w + pad_w
    fig_h = rows * panel_h + pad_h
    # -------------------------------------------------------------------

    for current_data, group_index in prepare_data(data, rows, cols):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=shared_y
        )


        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        else:
            axes = axes.reshape(rows, cols)

        axes_flat = axes.ravel()

        # Panels ohne Abstand + optional fast keine Außenränder
        fig.subplots_adjust(
            left=0.06, right=0.99, bottom=0.06, top=0.93,  # tweak nach Geschmack
            wspace=0.0, hspace=0.0
        )

        # current_data ist vermutlich eine Liste von (name, dict)-Tuples
        for i, (line_name, line_data) in enumerate(current_data):
            ax = axes_flat[i]

            ax.grid(True, which="major", axis="both", alpha=0.35)
            ax.grid(True, which="minor", axis="x", alpha=0.2)

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

            if line_name != "UVW2":

                y_values = y_values / exponent
                yerr_values = yerr_values / exponent
            else:
                y_values = y_values
                yerr_values = yerr_values

            row, col = divmod(i, cols)
            configure_lightcurves_axis(
                ax, row, col, new_ylabel, color,
                x_values, y_values, yerr_values,
                line_name, line_light_curves
            )

        # Leere Panels ausblenden (falls rows*cols > len(current_data))
        for j in range(len(current_data), rows * cols):
            axes_flat[j].set_visible(False)

        finalize_figure(
            fig, axes,
            x_label=xlabel, title=title, group_index=group_index,
            save_only=save_only, output_dir=output_dir,
            compare_cont=compare_cont, file_name=file_name
        )

def one_decimal_max(x, pos):
    s = f"{x:.1f}"          # immer 1 Nachkommastelle
    return s.rstrip("0").rstrip(".")  # .0 entfernen




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
            ax.set_ylabel(new_ylabel, fontsize=9)
        else:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_label_coords(-0.2, 0.5)
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_label_coords(1.2, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(one_decimal_max))

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=9)



def run_1d_lightcurves_groups(output_dir=DEFAULT_OUTPUT_DIR, save_only=False):
    ensure_output_dir(output_dir)
    data = import_1d_lightcurve_data()

    merged_dict = deep_merge(data["NGC4593_not_optical_calibrated"], data["NGC4593_optical_calibrated"])



    key_order_lines = ["UVW2", 'HAlpha', 'HBeta', 'HGamma', 'HDelta', 'LyAlpha_not_optical_calibrated', 'OI8446']

    key_order_conts = ["UVW2", "Cont1150_not_optical_calibrated", "Cont4010", "Cont4440", "Cont5100", "Cont6110", "Cont6880", "Cont8015", "Cont8900"]

    key_order_uv_lines = ["UVW2", 'HeII1640_not_optical_calibrated',
                          'HeI5875',
                          'HeII4685',
                          'NV1238_not_optical_calibrated',
                          'SiIV1393_not_optical_calibrated',
                          'CIV1548_not_optical_calibrated']

    key_order_lines_talk = ["UVW2", 'HAlpha', 'HBeta', 'LyAlpha_not_optical_calibrated', 'HeI5875', 'SiIV1393_not_optical_calibrated', 'CIV1548_not_optical_calibrated', 'HeII4685', 'SiIV1393_not_optical_calibrated','CIV1548_not_optical_calibrated']

    plot_1d_emission_lines_in_groups(
        merged_dict, output_dir, compare_cont="UVW2",
        key_order_lines=key_order_lines, save_only=save_only,
        file_name=f"Balmer_Lyman_and_O_lines"
    )

    plot_1d_emission_lines_in_groups(
        merged_dict, output_dir, compare_cont="UVW2",
        key_order_lines=key_order_uv_lines, save_only=save_only,
        file_name=f"He_and_UV_lines"
    )

    plot_1d_continua_in_groups(
        merged_dict, output_dir, compare_cont="UVW2",
        key_order_conts=key_order_conts, save_only=save_only,
        file_name=f"Continua"
    )

    plot_1d_emission_lines_in_groups(
        merged_dict, output_dir, compare_cont="UVW2",
        key_order_lines=key_order_lines_talk, save_only=save_only,
        file_name=f"Lines_talk", cols=2, rows=3
    )


def build_latex_table(rows, sorted_mjds, key_order, caption, label, scale=1e-15):
    """Build a LaTeX table string for a given set of line/cont keys."""

    # Remove UVW2 from data columns (it's just used for ordering/reference)
    data_keys = [k for k in key_order if k != "UVW2"]

    # Clean display names for column headers
    def clean_name(k):
        return format_label(k, as_latex=True)

    col_format = "l" + "c" * len(data_keys)

    header_entries = [r"Mod. Julian Date"] + [clean_name(k) for k in data_keys]
    col_headers = " & ".join(header_entries) + r" \\"

    lines_latex = []
    lines_latex.append(r"\begin{table*}")
    lines_latex.append(r"\centering")
    lines_latex.append(
        r"\caption{" + caption + r" Integrated line fluxes in units of "
                                 r"$10^{-15}$\,erg\,cm$^{-2}$\,s$^{-1}$.}"
    )
    lines_latex.append(r"\begin{tabular}{" + col_format + "}")
    lines_latex.append(r"\hline\hline")
    lines_latex.append(col_headers)
    lines_latex.append(r"\hline")

    for mjd in sorted_mjds:
        row_data = rows[mjd]
        row_entries = [f"{mjd:.2f}"]
        for key in data_keys:
            if key in row_data:
                flux, err = row_data[key]
                # Scale flux and error
                flux_scaled = flux / scale
                err_scaled = err / scale
                row_entries.append(f"${flux_scaled:.2f} \\pm {err_scaled:.2f}$")
            else:
                row_entries.append(r"$\cdots$")
        lines_latex.append(" & ".join(row_entries) + r" \\")

    lines_latex.append(r"\hline")
    lines_latex.append(r"\end{tabular}")
    lines_latex.append(r"\label{" + label + "}")
    lines_latex.append(r"\end{table*}")

    return "\n".join(lines_latex)


def print_tables(output_dir=DEFAULT_OUTPUT_DIR):
    ensure_output_dir(output_dir)
    data = import_1d_lightcurve_data()

    merged_dict = deep_merge(
        data["NGC4593_not_optical_calibrated"],
        data["NGC4593_optical_calibrated"]
    )

    x_key     = 'timestamps [MJD]'
    y_key     = 'fluxes [ergs/s/cm2/A]'
    yerr_name = 'fluxerrs [ergs/s/cm2/A]'

    #key_order_Balmer_lines = ['HAlpha', 'HBeta', 'HGamma', 'HDelta']
    #key_order_LyO_lines    = ['LyAlpha_not_optical_calibrated', 'OI8446']
    key_order_conts        = [
        "Cont1150_not_optical_calibrated", "Cont4010", "Cont4440",
        "Cont5100", "Cont6110", "Cont6880", "Cont8015", "Cont8900"
    ]
    #key_order_Helium_lines = ['HeI5875', 'HeII1640_not_optical_calibrated', 'HeII4685']
    #key_order_UV_lines     = [
    #    'NV1238_not_optical_calibrated',
    #    'SiIV1393_not_optical_calibrated',
    #    'CIV1548_not_optical_calibrated'
    #]

    key_order_lines = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'LyAlpha_not_optical_calibrated', 'OI8446', 'HeI5875', 'HeII1640_not_optical_calibrated', 'HeII4685', 'NV1238_not_optical_calibrated','SiIV1393_not_optical_calibrated','CIV1548_not_optical_calibrated']

    # ------------------------------------------------------------------ #
    # Map each key -> correct sub-dict ("lines" or "continua")
    # ------------------------------------------------------------------ #
    #all_line_keys = (
    #    key_order_Balmer_lines +
    #    key_order_LyO_lines    +
    #    key_order_Helium_lines +
    #    key_order_UV_lines
    #)
    all_line_keys = (
        key_order_lines
    )

    all_cont_keys = key_order_conts

    key_to_subdict = {}
    for key in all_line_keys:
        if key in merged_dict.get("lines", {}):
            key_to_subdict[key] = merged_dict["lines"][key]
        else:
            print(f"Warning: '{key}' not found in merged_dict['lines']")

    for key in all_cont_keys:
        if key in merged_dict.get("continua", {}):
            key_to_subdict[key] = merged_dict["continua"][key]
        else:
            print(f"Warning: '{key}' not found in merged_dict['continua']")

    # ------------------------------------------------------------------ #
    # Build rows dict:  mjd -> { key: (flux, err) }
    # ------------------------------------------------------------------ #
    rows = {}
    for key, line_data in key_to_subdict.items():
        x_values    = np.array(line_data.get(x_key,     []))
        y_values    = np.array(line_data.get(y_key,     []))
        yerr_noise  = np.array(line_data.get(yerr_name, []))

        # Correct uncertainties
        yerr_values = calculate_standard_error_for_lightcurves(y_values, yerr_noise)

        for mjd, flux, err in zip(x_values, y_values, yerr_values):
            rows.setdefault(mjd, {})[key] = (flux, err)

    sorted_mjds = sorted(rows.keys())

    # ------------------------------------------------------------------ #
    # Build & save the five tables
    # ------------------------------------------------------------------ #
    #table_configs = [
    #    (key_order_Balmer_lines, "Balmer emission lines.",  "tab:balmer", "balmer_lines_table.tex"),
    #    (key_order_LyO_lines,    "Ly$\\alpha$ and OI lines.", "tab:lyo",  "lyo_lines_table.tex"),
    #    (key_order_conts,        "Continuum fluxes.",       "tab:conts",  "cont_flux_table.tex"),
    #    (key_order_Helium_lines, "Helium emission lines.",  "tab:helium", "helium_lines_table.tex"),
    #    (key_order_UV_lines,     "UV emission lines.",      "tab:uv",     "uv_lines_table.tex"),
    #]

    table_configs = [
        (key_order_lines, "Emission lines.", "tab:emission-lines", "emission_lines_table.tex"),
        (key_order_conts, "Continuum fluxes.", "tab:conts", "cont_flux_table.tex"),

    ]

    for key_order, caption, label, filename in table_configs:
        latex_str = build_latex_table(rows, sorted_mjds, key_order, caption, label)
        out_path  = Path(output_dir) / "lightcurve_table" / filename
        with open(out_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {out_path}")



# methods to run


def plot_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    run_1d_lightcurves_groups(output_dir)


def save_1d_lightcurves_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    run_1d_lightcurves_groups(output_dir, save_only=True)


# plot_1d_lightcurves_in_groups()
# save_1d_lightcurves_in_groups()

# print_tables()