from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from import_data import import_line_profile_data, import_fits_data, find_prime_data_folder
from plot_utils import format_label, subtract_continuum, convert_to_velocity, save_velocity_data_to_txt, \
    ensure_output_dir, cut_normalized_line_out, cut_line_out
from general_plot import prepare_data, check_for_empty_rows
from settings import DEFAULT_OUTPUT_DIR, CENTRAL_WAVELENGTH

matplotlib.use('Qt5Agg')


DEFAULT_LINE_COLORS = [
    "#000000",  # black
    "#0072B2",  # blue
    "red",
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (use thicker line / edge; can be weak on white)
    # extras (still fairly distinct)
    "#332288",  # dark indigo
    "#88CCEE",  # light cyan
    "#44AA99",  # teal
    "#AA4499",  # purple
]


def finalize_figure(fig, axes, title, group_index, save_only, output_dir, x_label, line_profile=False, file_name=None, supertitle=None):
    """
    Finalizes the layout of a Matplotlib figure: removes empty subplot rows, sets title,
    saves the figure as PDF and PNG, and optionally displays it.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to finalize.
    axes : numpy.ndarray
        2D array of Matplotlib Axes objects.
    title : str
        Title of the figure.
    group_index : int
        Index of the current group (used in filenames).
    save_only : bool
        If True, the figure will only be saved (not shown).
    output_dir : str or pathlib.Path
        Directory where the figure will be saved.
    x_label : str or tuple of str
        Label(s) for the x-axis. Tuple assigns different labels per column.
    compare_cont : str, optional
        Optional identifier for use in filenames (e.g., when comparing continua).
    line_profile : bool, optional
        Whether the plots represent line profiles (affects formatting).
    file_name : str, optional
        Custom filename prefix. If None, the title is used instead.

    Returns:
    -----------
    None
    """

    check_for_empty_rows(axes, fig, x_label=x_label, line_profile=line_profile)

    if supertitle:
        if group_index > 0:
            fig.suptitle(f'{supertitle} - Group {group_index + 1}', fontsize=14, y=0.95)
        else:
            fig.suptitle(f'{supertitle}', fontsize=14)

    if output_dir:
        if file_name:
            save_path = output_dir / f"{file_name}.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            save_path = output_dir / f"{file_name}.png"
            plt.savefig(save_path, bbox_inches='tight')
        else:
            save_path = output_dir / f"{title.replace(' ', '_')}.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            save_path = output_dir / f"{title.replace(' ', '_')}.png"
            plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if not save_only:
        plt.show()
    plt.close(fig)



def add_top_velocity_axis(ax, xlim, major=2500, labelsize=9, rotation=45):
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xlim(*xlim)
    ax_top.xaxis.set_major_locator(MultipleLocator(major))

    # gleiche Optik wie unten:
    ax_top.tick_params(axis="x", labelsize=labelsize, rotation=rotation,
                       length=3.5, width=0.8, direction="out")  # <- anpassen nach Geschmack
    return ax_top


def plot_normalized_line_profiles_in_groups(
    data,
    save_only=False,
    output_dir=None,
    rows=2,
    cols=3,
    key_order=None,
    fig_size=None,                 # <- optional lassen, wir berechnen wenn None
    title="Normalized Line Profiles",
    shared_y=False,
    components=("avg", "rms"),
    safe_file_name="normalized_profiles",
):
    x_key = 'velocity space (km/s)'
    y_key = 'normalized flux'
    ylabel = 'normalized flux'

    # Komponenten normalisieren/prüfen
    components = tuple(c.lower() for c in components)
    for c in components:
        if c not in ("avg", "rms"):
            raise ValueError(f"Unknown component '{c}'. Use 'avg' and/or 'rms'.")

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / "plot_line_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    if key_order is None:
        key_order = list(data["avg"].keys())

    # Vorbereitung der Datenstruktur für prepare_data
    plot_data = {}
    for line in key_order:
        plot_data[line] = {}

        if "avg" in components:
            plot_data[line]["avg"] = {
                "x": data["avg"][line]["data_dict"][x_key],
                "y": data["avg"][line]["data_dict"][y_key]
            }

        if "rms" in components:
            plot_data[line]["rms"] = {
                "x": data["rms"][line]["data_dict"][x_key],
                "y": data["rms"][line]["data_dict"][y_key]
            }

    # -------- FIXE Panel-Größe: Panels bleiben immer gleich groß --------
    panel_w = 2.6  # inch pro Panel (Breite) -> einmal einstellen
    panel_h = 3.4  # inch pro Panel (Höhe)   -> einmal einstellen
    pad_w = 0.8     # extra links/rechts
    pad_h = 1.4     # extra oben/unten

    if fig_size is None:
        fig_size = (cols * panel_w + pad_w, rows * panel_h + pad_h)
    # -------------------------------------------------------------------

    for current_data, group_index in prepare_data(plot_data, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=shared_y)

        # axes immer 2D (für finalize_figure / check_for_empty_rows)
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        else:
            axes = axes.reshape(rows, cols)

        axes_flat = axes.ravel()

        # Panels ohne Abstand + geringe Außenränder (tweak falls nötig)
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.93, hspace=0.0, wspace=0.0)

        for i, (line_name, line_data) in enumerate(current_data):
            ax = axes_flat[i]
            row, col = divmod(i, cols)

            if line_data is None:
                continue

            # Defaults (leere Arrays)
            avg_x = avg_y = rms_x = rms_y = np.array([])

            if "avg" in components and "avg" in line_data:
                avg_x = line_data["avg"]["x"]
                avg_y = line_data["avg"]["y"]
            if "rms" in components and "rms" in line_data:
                rms_x = line_data["rms"]["x"]
                rms_y = line_data["rms"]["y"]

            configure_line_profile_axis(
                ax,
                row=row,
                col=col,
                ylabel=ylabel,
                avg_x=avg_x,
                avg_y=avg_y,
                rms_x=rms_x,
                rms_y=rms_y,
                line_name=line_name,
                components=components,
            )

        # Optional: leere Slots komplett entfernen (falls letzte Reihe nicht voll)
        for j in range(len(current_data), rows * cols):
            fig.delaxes(axes_flat[j])

        # wie viele Panels insgesamt?
        n_panels_total = len(plot_data)
        panels_per_fig = rows * cols
        multi_page = n_panels_total > panels_per_fig

        file_name = safe_file_name if not multi_page else f"{safe_file_name}_p{group_index + 1:02d}"

        finalize_figure(
            fig=fig,
            axes=axes,
            title=title,
            group_index=group_index,
            save_only=save_only,
            output_dir=output_dir,
            x_label="Velocity (km/s)",
            line_profile=True,
            file_name=file_name,  # <- NEU
        )


def plot_overlaid_normalized_line_profiles_in_panels(
    data,
    line_groups,
    components=("rms",),
    save_only=False,
    output_dir=None,
    fig_size=None,                      # <- None = automatisch aus Panelgröße
    title="Overlaid normalized line profiles",
    xlim=(-9999, 9999),
    ylim=(-0.05, 1.05),
    show_vline_zero=True,
    legend=True,
    avg_kwargs=None,
    rms_kwargs=None,
    color_map=None,
    safe_file_name="overlay_groups",
    rows=1,
    cols=None,
):
    x_key = "velocity space (km/s)"
    y_key = "normalized flux"

    components = tuple(c.lower() for c in components)
    for c in components:
        if c not in ("avg", "rms"):
            raise ValueError(f"Unknown component '{c}'. Use 'avg' and/or 'rms'.")

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / "plot_line_profiles_overlay"
    output_dir = ensure_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if avg_kwargs is None:
        avg_kwargs = dict(linestyle="-", linewidth=1.5)
    if rms_kwargs is None:
        rms_kwargs = dict(linestyle="-", linewidth=1.5)

    # ------------------------------------------------------------------
    # globale Farbzuordnung pro Linie, wenn nichts übergeben wurde
    unique_lines = []
    seen = set()
    for g in line_groups:
        for ln in g:
            if ln not in seen:
                seen.add(ln)
                unique_lines.append(ln)

    default_cycle = DEFAULT_LINE_COLORS.copy()

    mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for c in mpl_cycle:
        if c not in default_cycle:
            default_cycle.append(c)

    auto_color_map = {}
    if not color_map:
        if len(default_cycle) == 0:
            default_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

        for i, ln in enumerate(unique_lines):
            auto_color_map[ln] = default_cycle[i % len(default_cycle)]
    # ------------------------------------------------------------------

    # Panels expandieren, wenn mehrere components getrennt geplottet werden sollen
    expanded_panels = []
    if len(components) > 1:
        for comp in components:
            for g in line_groups:
                expanded_panels.append((g, comp))
    else:
        for g in line_groups:
            expanded_panels.append((g, components[0]))

    n_panels_total = len(expanded_panels)

    if cols is None:
        cols = n_panels_total if rows == 1 else int(np.ceil(n_panels_total / rows))

    panels_per_fig = rows * cols
    if panels_per_fig <= 0:
        raise ValueError("rows * cols must be >= 1")

    # -------- FIXE Panel-Größe: Panels bleiben immer gleich groß --------
    panel_w = 2.6  # inch pro Panel (Breite) -> einmal einstellen
    panel_h = 3.4  # inch pro Panel (Höhe)   -> einmal einstellen
    pad_w = 0.8  # extra links/rechts
    pad_h = 1.4  # extra oben/unten

    if fig_size is None:
        fig_size = (cols * panel_w + pad_w, rows * panel_h + pad_h)
    # -------------------------------------------------------------------

    def _iter_pages(panels, page_size):
        for start in range(0, len(panels), page_size):
            page_index = start // page_size
            chunk = panels[start:start + page_size]
            if len(chunk) < page_size:
                chunk = chunk + [None] * (page_size - len(chunk))
            yield chunk, page_index

    multi_page = n_panels_total > panels_per_fig

    for page_panels, page_index in _iter_pages(expanded_panels, panels_per_fig):
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=True)

        # axes immer 2D (wichtig für finalize_figure)
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        else:
            axes = axes.reshape(rows, cols)

        axes_flat = axes.ravel()

        # Panels ohne Abstand + kleine Außenränder
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.93, hspace=0.0, wspace=0.0)

        for i, panel in enumerate(page_panels):
            ax = axes_flat[i]
            r, c = divmod(i, cols)

            if panel is None:
                # besser als set_visible(False): komplett rausnehmen
                fig.delaxes(ax)
                continue

            group, comp = panel

            for line in group:
                if comp not in data or line not in data[comp]:
                    continue

                x = np.asarray(data[comp][line]["data_dict"][x_key])
                y = np.asarray(data[comp][line]["data_dict"][y_key])

                tmp = format_label(line, as_latex=False)
                line_label = tmp.split("  ")[0] if "  " in tmp else tmp

                kwargs = avg_kwargs if comp == "avg" else rms_kwargs

                # Farbe: 1) user color_map  2) auto_color_map
                if color_map:
                    if line in color_map:
                        if isinstance(color_map[line], dict) and "color" in color_map[line]:
                            kwargs = {**kwargs, "color": color_map[line]["color"]}
                        elif isinstance(color_map[line], str):
                            kwargs = {**kwargs, "color": color_map[line]}
                else:
                    if line in auto_color_map:
                        kwargs = {**kwargs, "color": auto_color_map[line]}

                ax.plot(x, y, label=line_label, **kwargs)

            ax.text(
                0.90, 0.98, comp.upper(),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=11,
            )

            if show_vline_zero:
                ax.vlines(0, ylim[0], ylim[1], linestyles="dashed", color="black", linewidth=0.5)
                ax.hlines(0, xlim[0], xlim[1], linestyles="dashed", color="black", linewidth=0.5)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(MultipleLocator(2500))
            ax.tick_params(axis="both", labelsize=9)

            # Top-X-Achse in der ersten Row
            if r == 0:
                add_top_velocity_axis(ax, xlim=xlim, major=2500, labelsize=9, rotation=45)

            if r == rows - 1:
                ax.set_xlabel("Velocity (km/s)", fontsize=12)
            else:
                ax.set_xticklabels([])

            if c == 0:
                ax.set_ylabel("normalized flux", fontsize=12)
                ax.tick_params(axis="y", which="both", left=True, labelleft=True)
            elif c == cols - 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("normalized flux", fontsize=12)
                ax.tick_params(
                    axis="y", which="both",
                    right=True, labelright=True,
                    left=False, labelleft=False,
                    direction="out",
                )
            else:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False, right=False, labelright=False)

            if legend:
                ax.legend(loc="upper left", fontsize=9)

        file_name = safe_file_name if not multi_page else f"{safe_file_name}_p{page_index+1:02d}"

        finalize_figure(
            fig=fig,
            axes=axes,
            title=title,
            group_index=page_index,
            save_only=save_only,
            output_dir=output_dir,
            x_label=("Velocity (km/s)",) * cols,
            line_profile=True,
            file_name=file_name,
        )


def configure_line_profile_axis(ax, row, col, ylabel, avg_x, avg_y, rms_x, rms_y, line_name,
                                 line_lightcurves=False, components=("avg","rms")):
    """
    Configures a subplot axis to display normalized AVG and RMS line profiles.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis object of the subplot.
    row : int
        Row index in the subplot grid.
    col : int
        Column index in the subplot grid.
    ylabel : str
        Label for the y-axis.
    avg_x, avg_y : array-like
        Velocity and normalized flux arrays for the AVG spectrum.
    rms_x, rms_y : array-like
        Velocity and normalized flux arrays for the RMS spectrum.
    line_name : str
        Name of the line (used in subplot label).
    line_lightcurves : bool
        If True, adjusts the y-axis label for lightcurves (optional/legacy).

    Returns:
    --------
    None
    """

    if "avg" in components:
        ax.plot(avg_x, avg_y, label=f'AVG', color='black')
    if "rms" in components:
        ax.plot(rms_x, rms_y, label=f'RMS', color='red')
    ax.vlines(0,-0.1, 1.5, linestyles="dashed", color="black", linewidth=0.5)
    ax.hlines(0, -10000, 10000, linestyles="dashed", color="black", linewidth=0.5)
    label = format_label(line_name, as_latex=False).split("  ")[0] if "  " in format_label(line_name, as_latex=False) else format_label(line_name, as_latex=False)
    ax.text(0.9, 0.97, f'{label}', transform=ax.transAxes,
            ha='right', va='top', fontsize=11)

    ax.set_xlim(-9999, 9999)
    ax.set_ylim(-0.1, 1.05)
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(loc='upper left')

    if col == 0:
        if row == 0 and line_lightcurves:
            ax.set_ylabel(
                (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{\AA}^{-1}]$"),
                fontsize=12)
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

    ax.xaxis.set_major_locator(MultipleLocator(2500))
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(2500))
        #ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=9)


pseudo_conts_for_line_avg = {

    'LyAlpha_not_optical_calibrated': {'blue': (1151, 1161), 'red': (1270, 1285)},
    'NV1238_not_optical_calibrated': {'blue': (1155, 1165), 'red': (1270, 1285)},
    'SiIV1393_not_optical_calibrated': {'blue': (1350, 1360), 'red': (1430,1440)},
    'CIV1548_not_optical_calibrated': {'blue': (1461, 1468), 'red': (1679, 1685)},
    'HeII1640_not_optical_calibrated': {'blue': (1461, 1468), 'red': (1679, 1685)},

    'HAlpha': {'blue': (6194, 6216), 'red': (6861, 6900)},
    'HBeta': {'blue': (4762, 4774), 'red': (5085, 5112)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (4026, 4033), 'red': (4197, 4220)},
    'HeI5875': {'blue': (5645, 5653), 'red': (6044, 6057)},
    'HeI7065': {'blue': (6934, 6941), 'red': (7331, 7357)},
    'HeI4471': {'blue': (4210, 4225), 'red': (4762, 4774)},
    'HeI5015': {'blue': (4976, 4981), 'red': (5085, 5112)},
    'HeII4685': {'blue': (4198, 4225), 'red': (4762, 4774)},
    'OI8446': {'blue': (7999, 8025), 'red': (8775, 8798)},
    'OIII5007': {'blue': (4976, 4987), 'red': (5085, 5112)},
}
pseudo_conts_for_line_rms = {

    'LyAlpha_not_optical_calibrated': {'blue': (1151, 1161), 'red': (1340, 1355)},
    'NV1238_not_optical_calibrated': {'blue': (1155, 1165), 'red': (1270, 1285)},
    'SiIV1393_not_optical_calibrated': {'blue': (1340, 1355), 'red': (1430,1440)},
    'CIV1548_not_optical_calibrated': {'blue': (1461, 1468), 'red': (1679, 1685)},
    'HeII1640_not_optical_calibrated': {'blue': (1461, 1468), 'red': (1679, 1685)},


    'HAlpha': {'blue': (6279, 6301), 'red': (6742, 6781)},
    'HBeta': {'blue': (4762, 4774), 'red': (4967, 4984)},
    'HGamma': {'blue': (4197, 4220), 'red': (4417, 4429)},
    'HDelta': {'blue': (4006, 4016), 'red': (4197, 4211)},
    'HeI5875': {'blue': (5773, 5790), 'red': (5952, 5961)},
    'HeI7065': {'blue': (6934, 6941), 'red': (7335, 7349)},
    'HeI4471': {'blue': (4210, 4225), 'red': (4762, 4774)},
    'HeI5015': {'blue': (4976, 4981), 'red': (5119, 5133)},
    'HeII4685': {'blue': (4543, 4554), 'red': (4762, 4774)},
    'OI8446': {'blue': (8222, 8238), 'red': (8748, 8767)},
    'OIII5007': {'blue': (4976, 4987), 'red': (5085, 5112)},
}


one_sided_FWHM_lines = {'NV1238_not_optical_calibrated': {"avg": "right", "rms": "right"},
                        'HeII4685': {"avg": "right"},
                        'HeII1640_not_optical_calibrated': {"rms": "left"},
                        }


DELTA_W = {

'HeII4685': 10,
'NV1238_not_optical_calibrated':5
}


import numpy as np

def compute_width_velocity_windowed_with_positions(
    velocity: np.ndarray,
    flux: np.ndarray,
    center_v: float = 0.0,
    window: float = 15000.0,
    peak_search_window: float = 3000.0,
    level: float = 0.5,
    level_mode: str = "relative",  # "relative" or "absolute"
    one_sided: str | None = None,  # None, "left", "right"
    one_sided_ref: str = "center", # "center" or "peak"
):
    """
    Compute line width at a specified height in velocity space.

    Returns (width_kms, v_left, v_right, y_level, peak).

    Enforces:
      - v_left  < center_v
      - v_right > center_v
    by selecting the closest level-crossings on each side of center_v.

    one_sided:
      - None   : width = v_right - v_left
      - "left" : width = 2 * |v_ref - v_left|
      - "right": width = 2 * |v_right - v_ref|

    one_sided_ref:
      - "center": v_ref = center_v
      - "peak"  : v_ref = v_at_peak
    """
    v = np.asarray(velocity, dtype=float)
    f = np.asarray(flux, dtype=float)

    if v.size < 5 or f.size != v.size:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    # sort by velocity
    if np.any(np.diff(v) < 0):
        s = np.argsort(v)
        v, f = v[s], f[s]

    # limit to window around the expected center
    mask = (v >= center_v - window) & (v <= center_v + window)
    if np.count_nonzero(mask) < 10:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    vw = v[mask]
    fw = f[mask]

    # find peak near center
    mask_peak = (vw >= center_v - peak_search_window) & (vw <= center_v + peak_search_window)
    if np.count_nonzero(mask_peak) < 5:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    i_peak_local = np.nanargmax(fw[mask_peak])
    peak = fw[mask_peak][i_peak_local]
    if not np.isfinite(peak) or peak <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    peak_indices = np.where(mask_peak)[0]
    i_peak = peak_indices[i_peak_local]
    v_peak = vw[i_peak]

    # define y-level
    if level_mode == "relative":
        if not (0.0 < level < 1.0):
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        y_level = level * peak
    elif level_mode == "absolute":
        y_level = float(level)
    else:
        raise ValueError("level_mode must be 'relative' or 'absolute'")

    if y_level >= peak:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    # --- find ALL crossings of (fw - y_level) via sign changes ---
    g = fw - y_level
    # indices where consecutive samples straddle 0 (sign change); ignore exact zeros safely
    sgn = np.sign(g)
    # replace zeros by nearest nonzero sign (simple fallback)
    if np.any(sgn == 0):
        # forward fill then backward fill
        for k in range(1, sgn.size):
            if sgn[k] == 0:
                sgn[k] = sgn[k-1]
        for k in range(sgn.size-2, -1, -1):
            if sgn[k] == 0:
                sgn[k] = sgn[k+1]

    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]  # sign change between i and i+1

    if idx.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    crossings = []
    for i in idx:
        v1, v2 = vw[i], vw[i+1]
        g1, g2 = g[i], g[i+1]
        if g2 == g1:
            continue
        vc = v1 + (0.0 - g1) * (v2 - v1) / (g2 - g1)  # linear interp to g=0
        crossings.append(vc)

    if len(crossings) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    crossings = np.array(crossings, dtype=float)

    # enforce left < center_v, right > center_v (closest to center on each side)
    left_candidates = crossings[crossings < center_v]
    right_candidates = crossings[crossings > center_v]

    v_left = np.nan if left_candidates.size == 0 else np.max(left_candidates)
    v_right = np.nan if right_candidates.size == 0 else np.min(right_candidates)

    # choose reference for one-sided doubling
    if one_sided_ref not in ("center", "peak"):
        raise ValueError("one_sided_ref must be 'center' or 'peak'")
    v_ref = center_v if one_sided_ref == "center" else v_peak

    # --- width computation ---
    if one_sided is None:
        if not (np.isfinite(v_left) and np.isfinite(v_right)):
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        width = v_right - v_left
    else:
        one_sided = one_sided.lower()
        if one_sided == "left":
            if not np.isfinite(v_left):
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            width = 2.0 * abs(v_ref - v_left)
        elif one_sided == "right":
            if not np.isfinite(v_right):
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            width = 2.0 * abs(v_right - v_ref)
        else:
            raise ValueError("one_sided must be None, 'left', or 'right'")

    if not np.isfinite(width) or width <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    return (float(width),
            float(v_left) if np.isfinite(v_left) else np.nan,
            float(v_right) if np.isfinite(v_right) else np.nan,
            float(y_level), float(peak))




def process_spectrum(
    wavelength,
    intensity,
    line_name,
    spec_type="rms",
    output_dir=DEFAULT_OUTPUT_DIR,
    plot=False,
    width_level=0.5,               # <- neu: z.B. 0.7 oder 0.8
    width_level_mode="relative",   # <- "relative" (0..1 vom Peak) oder "absolute"
    peak_search_window=1000.0
):
    """
    Processes a spectrum: subtracts pseudo-continuum, normalizes the line,
    converts to velocity space, saves outputs, and optionally plots.

    Additionally computes a line width at a user-defined height:
      - width_level=0.5 corresponds to classic FWHM (if mode="relative").

    Parameters:
    -----------
    wavelength : np.ndarray
        Array of wavelength values (Å).
    intensity : np.ndarray
        Array of flux values.
    line_name : str
        Name of the emission line (e.g. 'HAlpha').
    spec_type : str
        Either 'avg' or 'rms' to choose pseudo-continuum ranges.
    output_dir : Path
        Directory to save the processed spectrum and line profile.
    plot : bool
        If True, show diagnostic plots for the spectrum and line profile.
    width_level : float
        Level at which the width is measured.
        If width_level_mode="relative": fraction of peak (0 < level < 1).
        If width_level_mode="absolute": absolute flux value.
    width_level_mode : str
        "relative" or "absolute".

    Returns:
    --------
    None
    """

    line_wavelength = CENTRAL_WAVELENGTH[line_name]
    delta_w = DELTA_W.get(line_name, 50)

    if spec_type == "avg":
        pseudo_conts_for_line = pseudo_conts_for_line_avg
    else:
        pseudo_conts_for_line = pseudo_conts_for_line_rms

    blue_pseudo_cont = pseudo_conts_for_line[line_name]['blue']
    red_pseudo_cont = pseudo_conts_for_line[line_name]['red']

    wavelength_x_lim = (blue_pseudo_cont[0] - 100, red_pseudo_cont[1] + 100)

    mask_x_lim = (wavelength >= wavelength_x_lim[0]) & (wavelength <= wavelength_x_lim[1])
    max_intensity = np.max(intensity[mask_x_lim])
    min_intensity = np.min(intensity[mask_x_lim])

    y_lim_original = (min_intensity * 0.9, max_intensity * 1.1)

    corrected_intensity, continuum = subtract_continuum(
        wavelength, intensity, line_wavelength,
        blue_pseudo_cont, red_pseudo_cont, delta_w
    )

    velocity = convert_to_velocity(wavelength, line_wavelength)

    cut_intensity, cut_velocity = cut_normalized_line_out(
        corrected_intensity, velocity, [-20000, 20000]
    )

    one_sided = None

    if line_name in one_sided_FWHM_lines.keys():
        if spec_type in one_sided_FWHM_lines[line_name].keys():
            one_sided = one_sided_FWHM_lines[line_name][spec_type]

    # --- Width at chosen level (FWHM if width_level=0.5 & mode="relative") ---
    width_kms, v_left, v_right, y_level, peak = compute_width_velocity_windowed_with_positions(
        cut_velocity, cut_intensity,
        center_v=0.0,
        window=10000.0,
        peak_search_window=peak_search_window,
        level=width_level,
        level_mode=width_level_mode,
        one_sided=one_sided,
    )

    if width_level_mode == "relative":
        label = f"W@{width_level:.2f}*peak"
    else:
        label = f"W@flux={width_level:.2f}"

    print(f"[{line_name} | {spec_type}] {label} (km/s) = {width_kms}")

    np.savetxt(
        str(output_dir / f"{line_name}_{spec_type}_corrected_spectrum.txt"),
        np.column_stack((wavelength, corrected_intensity, continuum)),
        header="Wavelength (Å)  Intensity  Continuum"
    )
    np.savetxt(
        str(output_dir / f"{line_name}_normalized_{spec_type}_line_profile-{blue_pseudo_cont}-{red_pseudo_cont}.txt"),
        np.column_stack((cut_velocity, cut_intensity)),
        header="velocity space (km/s) \t normalized flux"
    )

    if plot is True:
        plt.figure(figsize=(8, 5))
        plt.plot(wavelength, intensity, label="Originalspektrum")
        plt.plot(wavelength, continuum, label="Interpoliertes Kontinuum", linestyle="dashed")
        plt.axvline(line_wavelength, color="r", linestyle=":", label="Linienzentrum")
        plt.axvspan(blue_pseudo_cont[0], blue_pseudo_cont[1], alpha=0.2, color="r")
        plt.axvspan(red_pseudo_cont[0], red_pseudo_cont[1], alpha=0.2, color="r")
        plt.xlim(wavelength_x_lim)
        plt.ylim(y_lim_original)
        plt.legend()
        plt.xlabel("Wellenlänge (Å)")
        plt.ylabel("Intensität")
        plt.title(line_name + " AVG")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(cut_velocity, cut_intensity, label="Linienprofil im Geschwindigkeitsraum")
        plt.axvline(0, color="r", linestyle=":", label="v = 0 km/s (Zentrum)")
        plt.xlim(-20000, 20000)
        plt.ylim(-0.1, 1.2)
        plt.axhline(0, color="black", linestyle=":")

        # --- Width-Positionen einzeichnen ---
        if np.isfinite(width_kms):
            plt.axhline(y_level, linestyle="--", label=f"Level = {y_level:.3f}")
            plt.axvline(v_left, linestyle="--", label=f"v_left = {v_left:.0f} km/s")
            plt.axvline(v_right, linestyle="--", label=f"v_right = {v_right:.0f} km/s")
            plt.axvspan(v_left, v_right, alpha=0.15, label=f"{label} = {width_kms:.0f} km/s")
        else:
            plt.text(
                0.02, 0.95,
                f"{label}: NaN (overlap / no crossing)",
                transform=plt.gca().transAxes, va="top"
            )

        plt.legend()
        plt.xlabel("Geschwindigkeit (km/s)")
        plt.ylabel("Intensität")
        plt.title(line_name + " RMS")
        plt.show()



def plot_cut_out_line_profile(cut_out_range, intensity_avg, intensity_rms, line_name, output_path, plot, velocity_avg, velocity_rms):
    """
    Plots and saves a velocity-space cut-out of a line profile (AVG and RMS).

    Parameters:
    -----------
    cut_out_range : tuple of float
        Velocity range (min, max) for the cut-out.
    intensity_avg : np.ndarray
        Normalized AVG profile intensities.
    intensity_rms : np.ndarray
        Normalized RMS profile intensities.
    line_name : str
        Name of the emission line.
    output_path : Path
        Directory to save the .txt output files.
    plot : bool
        If True, display a plot of both AVG and RMS profiles.
    velocity_avg : np.ndarray
        Velocity array for AVG profile.
    velocity_rms : np.ndarray
        Velocity array for RMS profile.

    Returns:
    --------
    None
    """

    range_str = f"{cut_out_range[0]}_{cut_out_range[1]}"
    save_velocity_data_to_txt(output_path / f"{line_name}_avg_profile_{range_str}.txt", velocity_avg, intensity_avg)
    save_velocity_data_to_txt(output_path / f"{line_name}_rms_profile_{range_str}.txt", velocity_rms, intensity_rms)
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        axs[0].plot(velocity_avg, intensity_avg, linestyle='-', marker='o', color='b')
        axs[0].set_ylabel("Avg Profile")
        axs[0].grid(True)

        axs[1].plot(velocity_rms, intensity_rms, linestyle='--', marker='s', color='r')
        axs[1].set_xlabel("Velocity (km/s)")
        axs[1].set_ylabel("RMS Profile")
        axs[1].grid(True)

        plt.suptitle(f"Line Profile: {line_name}")
        plt.show()

#methodes to run

def run_normalized_profiles_together_in_groups(output_dir=DEFAULT_OUTPUT_DIR):
    ensure_output_dir(output_dir)

    profile_data = import_line_profile_data(normalized=True)

    # key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', "LyAlpha_not_optical_calibrated", 'HeI5875', 'HeII4685', 'OI8446']
    key_order_all = ['HAlpha',
                     'HBeta',
                     'HGamma',
                     'HDelta',
                     'LyAlpha_not_optical_calibrated',
                     'OI8446',
                     'HeI5875',
                     'HeII1640_not_optical_calibrated',
                     'HeII4685',
                     'NV1238_not_optical_calibrated',
                     'SiIV1393_not_optical_calibrated',
                     'CIV1548_not_optical_calibrated']

    key_order_balmer = ['HAlpha', 'HBeta', 'HGamma']
    key_order_helium = ['HeI5875',  'HeII4685']
    key_order_Ly_O= ["LyAlpha_not_optical_calibrated", 'OI8446']
    key_order_UV = ['LyAlpha_not_optical_calibrated',
                    'NV1238_not_optical_calibrated',
                    'SiIV1393_not_optical_calibrated',
                    'CIV1548_not_optical_calibrated']

    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_balmer, title="Normalized Balmer Line Profiles", fig_size=(6, 8))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_helium, title="Normalized Helium Line Profiles", fig_size=(6, 4))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_Ly_O,
    #                                        title="Normalized Lyman and Oxygen Line Profiles", fig_size=(10, 5))

    plot_normalized_line_profiles_in_groups(profile_data, rows=4, cols=3, key_order=key_order_all,
                                            safe_file_name="Normalized Line Profiles")

    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_UV,
    #                                        safe_file_name="Normalized Line Profiles UV")

    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_balmer,
    #                                        title="Normalized AVG Balmer Line Profiles", fig_size=(6, 8), components=("avg",))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_helium,
    #                                       title="Normalized AVG Helium Line Profiles", fig_size=(6, 4), components=("avg",))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_balmer,
    #                                       title="Normalized RMS Balmer Line Profiles", fig_size=(6, 8), components=("rms",))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_helium,
    #                                        title="Normalized RMS Helium Line Profiles", fig_size=(6, 4), components=("rms",))

    plot_overlaid_normalized_line_profiles_in_panels(
        data=profile_data,
        line_groups=[["HAlpha", "HBeta"], ["HAlpha", "HGamma"], ["HBeta", "HGamma"], ["HAlpha", "HDelta"],["HBeta", "HDelta"], ["HGamma", "HDelta"]],
        components=("avg","rms"),
        title="AVG overlay Balmer",
        safe_file_name="AVG_RMS_overlay_Balmer",
        xlim=(-9999, 9999),
        rows=4,
        cols=3,
    )
    """
    plot_overlaid_normalized_line_profiles_in_panels(
        data=profile_data,
        line_groups=[["HAlpha", "HBeta"], ["HAlpha", "HGamma"], ["HBeta", "HGamma"], ["HAlpha", "HDelta"],
                     ["HBeta", "HDelta"], ["HGamma", "HDelta"]],
        components=("rms",),
        title="RMS overlay Balmer",
        safe_file_name="RMS_overlay_Balmer",
        xlim=(-9999, 9999),
        rows=2,
        cols=3,
    )
    """

    plot_overlaid_normalized_line_profiles_in_panels(
        data=profile_data,
        line_groups=[['HeI5875', 'HeII4685'], ['HeI5875', 'HeII1640_not_optical_calibrated'],['HeII4685', 'HeII1640_not_optical_calibrated']],
        components=("avg", "rms"),
        title="AVG and RMS overlay Helium",
        safe_file_name="AVG_RMS_overlay_Helium",
        xlim=(-9999, 9999),
        rows=2,
        cols=3,
    )
    """
    plot_overlaid_normalized_line_profiles_in_panels(
        data=profile_data,
        line_groups=[['HeI5875', 'HeII4685'], ['HeI5875', 'HeII1640_not_optical_calibrated'],
                     ['HeII4685', 'HeII1640_not_optical_calibrated']],
        components=("rms",),
        title="AVG and RMS overlay Helium",
        safe_file_name="RMS_overlay_Helium",
        xlim=(-9999, 9999),
        rows=2,
        cols=2,
    )
    """


def substract_pseudo_continua_from_spectra(plot=False, output_dir=DEFAULT_OUTPUT_DIR):

    data_path = find_prime_data_folder()

    output_dir = data_path / "LineProfiles"

    ensure_output_dir(output_dir)

    fits_data = import_fits_data()

    wavelenghts = np.array(fits_data['NGC4593_avg.fits']['x_axis'][0])
    avg_data = np.array(fits_data['NGC4593_avg.fits']['data'][0])
    rms_data = np.array(fits_data['NGC4593_rms.fits']['data'][0])

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeII4685', "OIII5007"]

    for line in key_order:
        process_spectrum(wavelenghts, avg_data, line, spec_type="avg", output_dir=output_dir, plot=plot, width_level=0.5)
        process_spectrum(wavelenghts, rms_data, line, spec_type="rms", output_dir=output_dir, plot=plot, width_level=0.5)

    uncalibrated_fits_data = import_fits_data(Path("fits") / "uncalibrated_AVG_RMS")

    uncalibrated_wavelenghts = np.array(uncalibrated_fits_data['avg.fits']['x_axis'][0])
    uncalibrated_avg_data = np.array(uncalibrated_fits_data['avg.fits']['data'][0])
    uncalibrated_rms_data = np.array(uncalibrated_fits_data['rms.fits']['data'][0])

    uv_lines_peak_windows = {
        "LyAlpha_not_optical_calibrated": 2000,
        "SiIV1393_not_optical_calibrated": 2000,
        "NV1238_not_optical_calibrated": 2000, # gecho
        "CIV1548_not_optical_calibrated": 2000,
        "HeII1640_not_optical_calibrated": 2000,
    }

    for line, peak_window in uv_lines_peak_windows.items():
        process_spectrum(
            uncalibrated_wavelenghts, uncalibrated_avg_data, line,
            spec_type="avg", output_dir=output_dir, plot=plot,
            peak_search_window=peak_window
        )
        process_spectrum(
            uncalibrated_wavelenghts, uncalibrated_rms_data, line,
            spec_type="rms", output_dir=output_dir, plot=plot,
            peak_search_window=peak_window
        )




def cut_line_profile(
    line_name,
    cut_out_range=(-1000, 1000),
    normalized=True,
    plot=False,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    def str_to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return False  # fallback

    normalized = str_to_bool(normalized)
    plot = str_to_bool(plot)

    if isinstance(cut_out_range, str):
        cut_out_range = tuple(map(int, cut_out_range.strip("() ").split(',')))

    # Sicherstellen, dass Hauptausgabeverzeichnis existiert
    output_dir_path = ensure_output_dir(output_dir)

    # Sicherstellen, dass Unterordner existiert
    output_path = output_dir_path / "Line_Profiles_cut_out"
    output_path.mkdir(parents=True, exist_ok=True)

    # Daten importieren
    line_profile_dict = import_line_profile_data(normalized=normalized)
    line_data_avg = line_profile_dict["avg"][line_name]["data_dict"]
    line_data_rms = line_profile_dict["rms"][line_name]["data_dict"]

    if normalized is True:
        velocity = line_data_avg.get("velocity space (km/s)")
        intensities_avg = line_data_avg.get("normalized flux")
        intensities_rms = line_data_rms.get("normalized flux")
        intensity_avg, velocity_avg = cut_normalized_line_out(intensities_avg, velocity, cut_out_range)
        intensity_rms, velocity_rms = cut_normalized_line_out(intensities_rms, velocity, cut_out_range)
    else:
        central_wave_length = CENTRAL_WAVELENGTH[line_name]
        cut_range_absolute = (
            central_wave_length + cut_out_range[0],
            central_wave_length + cut_out_range[1]
        )
        wavelengths = line_data_avg.get("velocity space (km/s)")
        intensities_avg = line_data_avg.get('flux ergs/s/cm2/A')
        intensities_rms = line_data_rms.get('flux ergs/s/cm2/A')
        intensity_avg, velocity_avg = cut_line_out(intensities_avg, wavelengths, cut_range_absolute)
        intensity_rms, velocity_rms = cut_line_out(intensities_rms, wavelengths, cut_range_absolute)

    # Plotten/Speichern
    plot_cut_out_line_profile(
        cut_out_range, intensity_avg, intensity_rms, line_name,
        output_path, plot, velocity_avg, velocity_rms
    )

substract_pseudo_continua_from_spectra(plot=True)

run_normalized_profiles_together_in_groups()
