import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator

from handle_data.handle_data_file import format_label, calculate_standard_error_for_lightcurves
from plot_data.general_plot import finalize_figure, format_yaxis, format_month_day
from settings import BASE_MJD


def plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccf_data_dict, campaign, output_dir, key_orders, save_only=False, file_name=None, final_key_order=None):
    base_mjd = BASE_MJD
    xlabel_ccfs = "Time Lag $\\tau$ [d]"
    ylabel_ccfs = "Correlation Coefficient"

    xlabel_lightcurves = f"MJD - {base_mjd:.2f}"
    ylabel_cont_lightcurves = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, "
                   r"\mathrm{\AA}^{-1}]$")
    ylabel_line_lightcurves = (r"$F_{\lambda}$ $[\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1}]$")
    yerr_name_lightcurves = 'fluxerrs [ergs/s/cm2/A]'




    save_folder = output_dir / campaign / "plot_1d_ccfs" / "corr_and_lightcurves"
    save_folder.mkdir(parents=True, exist_ok=True)

    all_sorted_data_dict = dict()

    first = True
    for reference_lightcurve, key_order in key_orders.items():

        def sort_keys(key):
            for idx, prefix in enumerate(key_order):
                if key == prefix:
                    return idx
            return len(key_order)

        try:
            sorted_data_dict = dict(
                sorted(lightcurves_ccf_data_dict["ccfs"][reference_lightcurve].items(),
                       key=lambda item: sort_keys(item[0])))


            keys_to_keep = key_order[1:]
            if first:
                keys_to_keep = ["time shift (tau)"] + keys_to_keep
                first = False

            sorted_data_dict = {
                (k if k == "time shift (tau)" else k + "_ref_" + reference_lightcurve): v
                for k, v in sorted_data_dict.items()
                if k in keys_to_keep
            }

        except KeyError:
            print(f"[Warning] Continuum name '{reference_lightcurve}' not found in campaign '{campaign}'. Skipping.")
            continue

        all_sorted_data_dict.update(sorted_data_dict)

    final_sorted_data_dict=dict()

    for key, value in all_sorted_data_dict.items():

        if key == "time shift (tau)":
            final_sorted_data_dict.update({key:value})
            continue

        line, reference = key.split("_ref_")

        if "Cont" in reference:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][reference]
        else:
            lightcurve_reference_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][reference]

        if "Cont" in line:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["continua"][line]
        else:
            lightcurve_data = lightcurves_ccf_data_dict["lightcurves"]["lines"][line]

        final_sorted_data_dict.update({key:{"ccfs":value,"lightcurves":lightcurve_data,"lightcurves_ref":lightcurve_reference_data}})

    def final_sort_keys(key):
        for idx, prefix in enumerate(final_key_order):
            if prefix == key.split("_ref_")[0]:
                return idx
        return len(key_order)

    final_sorted_data_dict = dict(
        sorted(final_sorted_data_dict.items(),
               key=lambda item: final_sort_keys(item[0])))


    plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs, xlabel_lightcurves, ylabel_line_lightcurves, ylabel_cont_lightcurves, yerr_name_lightcurves, title=f"CCFs and reference lightcurves",save_only=save_only, output_dir=save_folder, shared_y=False, file_name=file_name)


def plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs,
                                                  xlabel_lightcurves, ylabel_line_lightcurves, ylabel_cont_lightcurves,
                                                  yerr_name_lightcurves, title, save_only, output_dir, shared_y,
                                                  file_name, color_dict=None, rows=4, cols=2,):
    x_values_ccfs = final_sorted_data_dict['time shift (tau)']
    final_sorted_data_dict.pop('time shift (tau)')


    for current_data, group_index in prepare_ccfs_references_data(final_sorted_data_dict, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(8, 12), sharex=False, sharey=shared_y)
        fig.subplots_adjust(hspace=0, wspace=0)


        for i, (line_name, line_data) in enumerate(current_data):

            row, col = divmod(i, cols)

            ax = axes[row, col]

            if line_data is not None:
                if i % 2 == 0:
                    color = ("blue", "orange")
                    yerr = True
                else:
                    color = "black"
                    yerr = None
            else:
                yerr = None
                line_data = np.array([])
                color = "black"

            configure_ccfs_and_reference_axis(ax, row, col, ylabel_ccfs, color, x_values_ccfs, line_data, yerr, line_name)


        finalize_figure(fig, axes, x_label=(xlabel_lightcurves, xlabel_ccfs), title=title, group_index=group_index,
                        save_only=save_only, output_dir=output_dir, file_name=file_name)


def prepare_ccfs_references_data(data, rows, cols):
    # Jedes Item zweimal hintereinander einfügen
    data_items = []
    for key, value in data.items():
        data_items.append((key, value))
        data_items.append((key, value))  # direkt danach nochmal

    total_plots = len(data_items)
    plots_per_group = rows * cols
    num_groups = (total_plots + plots_per_group - 1) // plots_per_group

    for group_index in range(num_groups):
        start_index = group_index * plots_per_group
        end_index = min(start_index + plots_per_group, total_plots)
        current_data = data_items[start_index:end_index]

        # Mit Platzhaltern auffüllen, falls unvollständig
        while len(current_data) < plots_per_group:
            current_data.append((
                f'Empty {len(current_data) + 1}',
                None
            ))

        yield current_data, group_index



def configure_ccfs_and_reference_axis(ax, row, col, ylabel_ccfs, color, x_values_ccfs, line_data, yerr,
                                      line_name_and_ref_name):
    line_name, reference_name = line_name_and_ref_name.split("_ref_")

    if len(line_data) == 0:
        return

    if yerr is not None:
        x_key = 'timestamps [MJD]'
        y_key = 'fluxes [ergs/s/cm2/A]'
        yerr_key = 'fluxerrs [ergs/s/cm2/A]'

        y_norm, yerr_norm = normalize_lightcurve(line_data["lightcurves"][y_key], line_data["lightcurves"][yerr_key])
        y_ref_norm, yerr_ref_norm = normalize_lightcurve(line_data["lightcurves_ref"][y_key],
                                                          line_data["lightcurves_ref"][yerr_key])

        plot_normalized_lightcurve(ax, line_data["lightcurves"][x_key], y_norm, yerr_norm,
                                   format_label(line_name, as_latex=False), color[0])
        plot_normalized_lightcurve(ax, line_data["lightcurves_ref"][x_key], y_ref_norm, yerr_ref_norm,
                                   format_label(reference_name, as_latex=False), color[1])

        configure_axes_for_lightcurves(ax, row, col)
    else:
        ax.plot(x_values_ccfs, line_data["ccfs"], label=format_label(line_name, as_latex=False), color=color)
        configure_axes_for_ccfs(ax, row, col, ylabel_ccfs)

    ax.legend(fontsize=8, loc='upper right')



def normalize_lightcurve(y, yerr_vals):
    yerr_vals = calculate_standard_error_for_lightcurves(y, yerr_vals)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std
    yerr_norm = yerr_vals / y_std
    return y_norm, yerr_norm


def plot_normalized_lightcurve(ax, x, y_norm, yerr_norm, label, color):
    ax.errorbar(x, y_norm, yerr=yerr_norm, fmt='.:', capsize=3, markersize=4, label=label, color=color)


def configure_axes_for_lightcurves(ax, row, col):
    if col == 0:
        ax.set_ylabel("Normalized Lightcurves", fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)

    if row < 3:
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(-3, 3.5)

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.xaxis.set_major_formatter(FuncFormatter(format_month_day))
        ax_top.tick_params(axis='x', rotation=45, labelsize=10)


def configure_axes_for_ccfs(ax, row, col, ylabel_ccfs):
    ax.axvline(x=0, color='black', linestyle=':', linewidth=0.5)
    ax.set_xlim(-9.999, 14.999)
    ax.set_ylim(-0.1, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    if col == 0:
        ax.set_ylabel("Lightcurves", fontsize=12)
        ax.yaxis.set_label_coords(-0.15, 0.5)
    else:
        ax.yaxis.tick_right()
        ax.set_ylabel(ylabel_ccfs, fontsize=12)
        ax.yaxis.set_label_position("right")
        ax_right = ax.secondary_yaxis('right')
        ax_right.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_right.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

    if row < 3:
        ax.set_xticklabels([])

    if row == 0:
        ax_top = ax.secondary_xaxis('top')
        ax_top.xaxis.set_major_locator(MultipleLocator(5))
        ax_top.tick_params(axis='x')
