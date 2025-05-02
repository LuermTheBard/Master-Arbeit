# todo: methode should give the data to an equivilant methode to plot_ccfs_in_groups, which uses prepare data
# todo: write a fitting configure_..._axis methode


from settings import BASE_MJD


def plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs,
                                                  xlabel_lightcurves, ylabel_line_lightcurves, ylabel_cont_lightcurves,
                                                  yerr_name_lightcurves, title, save_only, output_dir, shared_y,
                                                  file_name):
    pass


def plot_1d_corr_and_lightcurves_in_groups(lightcurves_ccf_data_dict, campaign, output_dir, key_orders, save_only=True, file_name=None, only_key_order=False):
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

            if only_key_order is True:
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

    plot_ccfs_and_reference_lightcurves_in_groups(final_sorted_data_dict, xlabel_ccfs, ylabel_ccfs, xlabel_lightcurves, ylabel_line_lightcurves, ylabel_cont_lightcurves, yerr_name_lightcurves,title=f"CCFs and reference lightcurves",save_only=save_only, output_dir=save_folder, shared_y=True, file_name=file_name)