from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from import_data import import_line_profile_data, import_fits_data
from plot_utils import format_label, subtract_continuum, convert_to_velocity, save_velocity_data_to_txt, \
    ensure_output_dir, cut_normalized_line_out, cut_line_out
from general_plot import finalize_figure, prepare_data
from settings import DEFAULT_OUTPUT_DIR, CENTRAL_WAVELENGTH, SYMBOLES_AND_COLORS_FOR_LIGHTCURVES

matplotlib.use('Qt5Agg')

def plot_normalized_line_profiles_in_groups(
    data,
    save_only=False,
    output_dir=None,
    rows=2,
    cols=3,
    key_order=None,
    fig_size=(8, 8),
    title="Normalized Line Profiles",
    shared_y=False,
    components=("avg", "rms"),   # <- NEU: ("avg",) oder ("rms",) oder ("avg","rms")
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

    for current_data, group_index in prepare_data(plot_data, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=shared_y)
        axes = np.array(axes).reshape(rows, cols)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, (line_name, line_data) in enumerate(current_data):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            if line_data is None:
                continue

            # Defaults (leere Arrays)
            avg_x = avg_y = rms_x = rms_y = np.array([])

            if line_data is not None:
                if "avg" in components and "avg" in line_data:
                    avg_x = line_data["avg"]["x"]
                    avg_y = line_data["avg"]["y"]
                if "rms" in components and "rms" in line_data:
                    rms_x = line_data["rms"]["x"]
                    rms_y = line_data["rms"]["y"]

            # -> Wichtig: configure_line_profile_axis muss wissen, was geplottet werden soll
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
                components=components,   # <- NEU (siehe unten)
            )

        finalize_figure(
            fig=fig,
            axes=axes,
            title=title,
            group_index=group_index,
            save_only=save_only,
            output_dir=output_dir,
            x_label="Velocity (km/s)",
            line_profile=True
        )


def plot_overlaid_normalized_line_profiles_in_panels(
    data,
    line_groups,                       # z.B. [["HAlpha","HBeta"], ["HeI5875","HeII4685"]]
    components=("rms",),
    save_only=False,
    output_dir=None,
    fig_size=(12, 6),
    title="Overlaid normalized line profiles",
    xlim=(-9000, 8999),
    ylim=(-0.1, 1.2),
    show_vline_zero=True,
    legend=True,
    avg_kwargs=None,
    rms_kwargs=None,
    color_map=None,                    # optional: Dict -> nur "color" wird genutzt
    safe_file_name="overlay_groups",   # Windows-safe!

    # NEU:
    rows=1,
    cols=None,                         # wenn None: wie vorher -> 1 x n_panels
):
    x_key = "velocity space (km/s)"
    y_key = "normalized flux"

    # Komponenten normalisieren/prüfen
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

    n_panels_total = len(line_groups)

    # Backward-compatible Default: 1 x n_panels
    if cols is None:
        cols = n_panels_total if rows == 1 else int(np.ceil(n_panels_total / rows))

    panels_per_fig = rows * cols
    if panels_per_fig <= 0:
        raise ValueError("rows * cols must be >= 1")

    # Helper: chunking/paging wie prepare_data, aber für line_groups-Listen
    def _iter_pages(groups, page_size):
        for start in range(0, len(groups), page_size):
            page_index = start // page_size
            chunk = groups[start:start + page_size]
            # auf volle Gridgröße auffüllen
            if len(chunk) < page_size:
                chunk = chunk + [None] * (page_size - len(chunk))
            yield chunk, page_index

    multi_page = n_panels_total > panels_per_fig

    for page_groups, page_index in _iter_pages(line_groups, panels_per_fig):
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharex=True, sharey=True)
        axes = np.array(axes).reshape(rows, cols)
        fig.subplots_adjust(hspace=0, wspace=0)

        for i, group in enumerate(page_groups):
            r, c = divmod(i, cols)
            ax = axes[r, c]

            # Leere Panels (Padding) -> ausblenden
            if group is None:
                ax.set_visible(False)
                continue

            # Plot: pro Panel mehrere Lines überlagert, ggf. mehrere components
            for comp in components:
                for line in group:
                    if comp not in data or line not in data[comp]:
                        continue

                    x = np.asarray(data[comp][line]["data_dict"][x_key])
                    y = np.asarray(data[comp][line]["data_dict"][y_key])

                    line_label = format_label(line, as_latex=False)
                    label = f"{line_label} ({comp.upper()})" if len(components) > 1 else line_label

                    kwargs = avg_kwargs if comp == "avg" else rms_kwargs

                    # Farbe aus color_map (falls vorhanden)
                    if color_map and line in color_map and "color" in color_map[line]:
                        kwargs = {**kwargs, "color": color_map[line]["color"]}

                    ax.plot(x, y, label=label, **kwargs)

            # Achsen-Setup
            if show_vline_zero:
                ax.vlines(0, ylim[0], ylim[1], linestyles="dashed")

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(MultipleLocator(2500))
            ax.tick_params(axis="both", labelsize=9)

            # X-Labels nur in der untersten Zeile
            if r == rows - 1:
                ax.set_xlabel("Velocity (km/s)", fontsize=12)
            else:
                ax.set_xticklabels([])

            # Y-Handling:
            # - linke Spalte: links ticks + ylabel
            # - rechte Spalte: rechts ticks + ylabel
            # - mittlere Spalten: keine y-ticks/labels
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

        # Datei-Name pro Seite (wenn nötig)
        file_name = safe_file_name if not multi_page else f"{safe_file_name}_p{page_index+1:02d}"

        finalize_figure(
            fig=fig,
            axes=axes,
            title=title,
            group_index=page_index,
            save_only=save_only,
            output_dir=output_dir,
            x_label=("Velocity (km/s)",) * cols,   # tuple pro Spalte
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
    ax.vlines(0, -0.1, 1.5, linestyles='dashed', color='black')
    ax.text(0.95, 0.95, f'{format_label(line_name, as_latex=False)}', transform=ax.transAxes,
            ha='right', va='top', fontsize=11)

    ax.set_xlim(-9000, 8999)
    ax.set_ylim(-0.1, 1.2)
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

    'LyAlpha_not_optical_calibrated': {'blue': (1155, 1165), 'red': (1270, 1285)},

    'HAlpha': {'blue': (6107, 6129), 'red': (6861, 6900)},
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

    'LyAlpha_not_optical_calibrated': {'blue': (1155, 1165), 'red': (1300, 1315)},


    'HAlpha': {'blue': (6201, 6223), 'red': (6970, 6994)},
    'HBeta': {'blue': (4762, 4774), 'red': (4970, 4990)},
    'HGamma': {'blue': (4197, 4220), 'red': (4435, 4450)},
    'HDelta': {'blue': (3939, 3950), 'red': (4197, 4220)},
    'HeI5875': {'blue': (5649, 5660), 'red': (6068, 6081)},
    'HeI7065': {'blue': (6934, 6941), 'red': (7335, 7349)},
    'HeI4471': {'blue': (4210, 4225), 'red': (4762, 4774)},
    'HeI5015': {'blue': (4976, 4981), 'red': (5119, 5133)},
    'HeII4685': {'blue': (4198, 4225), 'red': (4770, 4787)},
    'OI8446': {'blue': (8222, 8238), 'red': (8748, 8767)},
    'OIII5007': {'blue': (4976, 4987), 'red': (5085, 5112)},
}


def process_spectrum(wavelength, intensity, line_name, spec_type="rms", output_dir=DEFAULT_OUTPUT_DIR, plot=False):
    """
    Processes a spectrum: subtracts pseudo-continuum, normalizes the line,
    converts to velocity space, saves outputs, and optionally plots.

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

    Returns:
    --------
    None
    """

    line_wavelength = CENTRAL_WAVELENGTH[line_name]

    if spec_type == "avg":
        pseudo_conts_for_line = pseudo_conts_for_line_avg
    else:
        pseudo_conts_for_line = pseudo_conts_for_line_rms


    blue_pseudo_cont = pseudo_conts_for_line[line_name]['blue']
    red_pseudo_cont = pseudo_conts_for_line[line_name]['red']

    wavelength_x_lim = (blue_pseudo_cont[0]-100, red_pseudo_cont[1]+100)

    mask_x_lim = (wavelength >= wavelength_x_lim[0]) & (wavelength <= wavelength_x_lim[1])
    max_intensity = np.max(intensity[mask_x_lim])
    min_intensity = np.min(intensity[mask_x_lim])

    y_lim_original = (min_intensity * 0.9, max_intensity * 1.1)  # Optional: 10% Puffer nach oben

    corrected_intensity, continuum = subtract_continuum(wavelength, intensity, line_wavelength, blue_pseudo_cont, red_pseudo_cont)

    velocity = convert_to_velocity(wavelength, line_wavelength)

    output_dir = output_dir / "substracted_pseudocont_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(str(output_dir / f"{line_name}_{spec_type}_corrected_spectrum.txt"), np.column_stack((wavelength, corrected_intensity, continuum)),
               header="Wavelength (Å)  Intensity  Continuum")
    np.savetxt(str(output_dir /f"{line_name}_normalized_{spec_type}_line_profile-{blue_pseudo_cont}-{red_pseudo_cont}.txt"), np.column_stack((velocity, corrected_intensity)),
               header="velocity space (km/s) \t normalized flux")

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
        plt.plot(velocity, corrected_intensity, label="Linienprofil im Geschwindigkeitsraum")
        plt.axvline(0, color="r", linestyle=":", label="v = 0 km/s (Zentrum)")
        plt.xlim(-10000,10000)
        plt.ylim(-0.1, 1.2)
        plt.axhline(0, color="black", linestyle=":")
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
    key_order_all = ['HAlpha', 'HBeta', 'HGamma', "LyAlpha_not_optical_calibrated", 'HeI5875', 'HeII4685', 'OI8446']

    key_order_balmer = ['HAlpha', 'HBeta', 'HGamma']
    key_order_helium = ['HeI5875',  'HeII4685']
    key_order_Ly_O= ["LyAlpha_not_optical_calibrated", 'OI8446']

    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_balmer, title="Normalized Balmer Line Profiles", fig_size=(8, 8))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_helium, title="Normalized Helium Line Profiles", fig_size=(8, 4))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_Ly_O,
    #                                        title="Normalized Lyman and Oxygen Line Profiles", fig_size=(10, 5))

    #plot_normalized_line_profiles_in_groups(profile_data, rows=4, cols=2, key_order=key_order_all,
    #                                        title="Normalized Line Profiles", fig_size=(8, 12))

    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_balmer,
    #                                        title="Normalized AVG Balmer Line Profiles", fig_size=(8, 8), components=("avg",))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_helium,
    #                                        title="Normalized AVG Helium Line Profiles", fig_size=(8, 4), components=("avg",))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=2, cols=2, key_order=key_order_balmer,
    #                                        title="Normalized RMS Balmer Line Profiles", fig_size=(8, 8), components=("rms",))
    #plot_normalized_line_profiles_in_groups(profile_data, rows=1, cols=2, key_order=key_order_helium,
    #                                        title="Normalized RMS Helium Line Profiles", fig_size=(8, 4), components=("rms",))


    plot_overlaid_normalized_line_profiles_in_panels(
        data=profile_data,
        line_groups=[["HAlpha", "HBeta"], ["HAlpha", "HGamma"], ['HeI5875', 'HeII4685']],
        components=("rms",),
        title="RMS overlay Balmer",
        color_map=SYMBOLES_AND_COLORS_FOR_LIGHTCURVES,
        safe_file_name="RMS_overlay_Balmer",
        xlim=(-4999, 5000),
        rows=2,
        cols=2,
        fig_size=(10, 10)
    )



def substract_pseudo_continua_from_spectra(plot=False, output_dir=DEFAULT_OUTPUT_DIR):
    ensure_output_dir(output_dir)

    fits_data = import_fits_data()

    wavelenghts = np.array(fits_data['NGC4593_avg.fits']['x_axis'][0])
    avg_data = np.array(fits_data['NGC4593_avg.fits']['data'][0])
    rms_data = np.array(fits_data['NGC4593_rms.fits']['data'][0])

    key_order = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685',
                 'OI8446', "OIII5007"]

    for line in key_order:
        process_spectrum(wavelenghts, avg_data, line, spec_type="avg", output_dir=output_dir, plot=plot)
        process_spectrum(wavelenghts, rms_data, line, spec_type="rms", output_dir=output_dir, plot=plot)

    uncalibrated_fits_data = import_fits_data(Path("fits") / "uncalibrated_AVG_RMS")

    uncalibrated_wavelenghts = np.array(uncalibrated_fits_data['avg.fits']['x_axis'][0])
    uncalibrated_avg_data = np.array(uncalibrated_fits_data['avg.fits']['data'][0])
    uncalibrated_rms_data = np.array(uncalibrated_fits_data['rms.fits']['data'][0])

    process_spectrum(uncalibrated_wavelenghts, uncalibrated_avg_data, "LyAlpha_not_optical_calibrated", spec_type="avg", output_dir=output_dir, plot=plot)
    process_spectrum(uncalibrated_wavelenghts, uncalibrated_rms_data, "LyAlpha_not_optical_calibrated", spec_type="rms", output_dir=output_dir, plot=plot)




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

# substract_pseudo_continua_from_spectra()

run_normalized_profiles_together_in_groups()
