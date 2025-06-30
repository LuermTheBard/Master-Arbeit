from pathlib import Path

import numpy as np

from Malte_get_BH_mass import Line
from plot_utils import print_table_for_one_reference, print_table_for_multiple_reference, save_centroid_as_txt
from import_data import find_prime_data_folder
from settings import FWHM_RMS, FWHM_ERR



def import_centroid_and_mc_data(campaign, continuum, lines):
    data_folder = find_prime_data_folder()
    base_path = Path(
        rf'{data_folder}\campaigns\{campaign}\calc_time_lag_ccfs\{continuum}'
    )

    # Lade die lineCorrelations-Daten
    line_correlations_path = base_path / "lineCorrelations_ICCF.txt"
    lightcurve_correlations_path = base_path / "lightcurveCorrelations_ICCF.txt"
    try:

        with open(str(line_correlations_path), "r") as file:
            header_line = file.readline().strip().split(" ")
            line_correlation_data = np.loadtxt(line_correlations_path).T

        correlation_header = ["time shift (tau)"] + header_line[5:]

        correlation_data_dict = dict()

        for i, name in enumerate(correlation_header):
            correlation_data_dict[name] = line_correlation_data[i]

    except Exception as e:
        print(f"❌ Fehler beim Laden der Datei {line_correlations_path}: {e}")
        return None

    if lightcurve_correlations_path.exists():
        try:
            with open(str(lightcurve_correlations_path), "r") as file:
                header_line = file.readline().strip().split(" ")
                line_correlation_data = np.loadtxt(lightcurve_correlations_path).T

            correlation_header = ["time shift (tau)"] + header_line[5:]

            for i, name in enumerate(correlation_header):
                if name not in correlation_data_dict.keys():
                    correlation_data_dict[name] = line_correlation_data[i]
        except Exception as e:
            print(f"⚠️ Datei {lightcurve_correlations_path} konnte nicht verarbeitet werden: {e}")
            print("➡️ Verwende nur lineCorrelations.")
    else:
        print(f"ℹ️ Datei {lightcurve_correlations_path} nicht gefunden. Verwende nur lineCorrelations.")

    # Lade alle Zentroid- und Peak-Daten in ein Dictionary
    mc_data = {}
    for line in lines:
        try:
            cents_path = base_path / f"calculatedCentroids{continuum}_{line}_ICCF.txt"
            peaks_path = base_path / f"peakDistribution_{continuum}_{line}_ICCF.txt"

            # Falls eine Datei fehlt, wird sie übersprungen
            mc_data[line] = {
                "centroids": np.loadtxt(cents_path) if cents_path.exists() else None,
                "peaks": np.loadtxt(peaks_path) if peaks_path.exists() else None
            }
        except Exception as e:
            print(f"⚠️ Warnung: Fehler beim Laden der Dateien für {line}: {e}")
            continue  # Statt `return`, damit andere Linien geladen werden

    return correlation_data_dict, mc_data


def calc_centroid_malte_code(campaign, continuum, lines, include_mass=True, create_tex_file=True):


    correlation_data_dict, mc_data = import_centroid_and_mc_data(campaign, continuum, lines=lines)

    # Erstelle die Line-Objekte
    line_objects = []
    for line in lines:
        if line in correlation_data_dict.keys() and mc_data[line]["centroids"] is not None and mc_data[line]["peaks"] is not None:
            line_obj = Line(
                line,  # Name der Linie
                FWHM_RMS.get(line, 0.0),  # FWHM (rms)
                FWHM_ERR.get(line, 0.0),  # Sigma (rms)
                np.vstack((correlation_data_dict['time shift (tau)'], correlation_data_dict[line])).T,
                mc_data[line]["centroids"],
                mc_data[line]["peaks"]
            )
            line_objects.append(line_obj)

    # Speichere die Ergebnisse mit dynamischem Dateinamen
    if create_tex_file:
        output_filename = f'CCF_lags_{campaign}_{continuum}.tex'
        print(f"✅ Speichere Ergebnisse in Datei: {output_filename}")
        save_centroid_as_txt(line_objects, f"CCF_lags_{campaign}_{continuum}.txt")
        print_table_for_one_reference(output_filename, line_objects, continuum, include_mass=include_mass)
    return line_objects


def get_fluoreszenz_table():
    output_filename = f'CCF_lags_fluoreszenz.tex'

    campaign = "NGC4593_optical_calibrated"
    reference_light_curve_lines_dict = {
        #"Cont1150": ["HAlpha",
        #             "HBeta",
        #             "LyAlpha",
        #             "OI8446",
        #             "HBeta_not_optical_calibrated",
        #             "LyAlpha_not_optical_calibrated",
                     # "OI8446_not_optical_calibrated"
         #            ],
        "Cont1150_not_optical_calibrated": ["HAlpha",
                                            "HBeta",
                                            #"LyAlpha",
                                            "OI8446",
                                            #"HBeta_not_optical_calibrated",
                                            "LyAlpha_not_optical_calibrated",
                                            # "OI8446_not_optical_calibrated"
                                            ],
       # "Cont1460": ["HAlpha",
       #              "HBeta",
       #              "LyAlpha",
       #              "OI8446",
                     #"HBeta_not_optical_calibrated",
        #             "LyAlpha_not_optical_calibrated",
                     #"OI8446_not_optical_calibrated"
        #             ],
        #"Cont1460_not_optical_calibrated": ["HAlpha",
        #                                    "HBeta",
                                            #"LyAlpha",
        #                                   "OI8446",
                                            #"HBeta_not_optical_calibrated",
        #                                    "LyAlpha_not_optical_calibrated",
                                            #"OI8446_not_optical_calibrated"
        #                                    ],
        #"LyAlpha": ["HAlpha",
        #            "HBeta",
        #            "OI8446",
        #            "HBeta_not_optical_calibrated",
                    # "OI8446_not_optical_calibrated"
        #            ],
        "LyAlpha_not_optical_calibrated": ["HAlpha",
                                           "HBeta",
                                           "OI8446",
                                           # "HBeta_not_optical_calibrated",
                                           # "OI8446_not_optical_calibrated"
                                           ],
        "HBeta": ["OI8446",
                  #"OI8446_not_optical_calibrated"
                  ],
        "HAlpha": ["OI8446",
                   ]
        #"HBeta_not_optical_calibrated": ["OI8446",
                                         #"OI8446_not_optical_calibrated"
         #                                ]
    }

    data_light_curve_lines_dict = {}

    for reference, lines in reference_light_curve_lines_dict.items():
        data_light_curve_lines_dict[reference] = calc_centroid_malte_code(campaign, reference, lines, include_mass=False, create_tex_file=True)

    print_table_for_multiple_reference(output_filename, data_light_curve_lines_dict, include_mass=False)


#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150_not_optical_calibrated", lines=["HBeta", "LyAlpha", "OI8446", "HBeta_not_optical_calibrated", "LyAlpha_not_optical_calibrated", "OI8446_not_optical_calibrated"], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1450_not_optical_calibrated", lines=["HBeta", "LyAlpha", "OI8446", "HBeta_not_optical_calibrated", "LyAlpha_not_optical_calibrated", "OI8446_not_optical_calibrated"], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "LyAlpha_not_optical_calibrated", lines=["HBeta", "OI8446", "HBeta_not_optical_calibrated", "OI8446_not_optical_calibrated"], include_mass=False, create_tex_file=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "LyAlpha", lines=["HBeta", "OI8446"], include_mass=False)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150", lines=['HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685'], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150_not_optical_calibrated", lines=['HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685'], include_mass=True, create_tex_file=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "HBeta", lines=["OI8446", "OI8446_not_optical_calibrated"], include_mass=True, create_tex_file=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "HAlpha", lines=["OI8446", "OI8446_not_optical_calibrated"], include_mass=True, create_tex_file=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "UVW2", lines=['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'LyAlpha','HeI5875', 'HeII4685', 'OI8446'], include_mass=True, create_tex_file=True)
#calc_centroid_malte_code("NGC4593_not_optical_calibrated", "UVW2",
 #                        lines=["LyAlpha_not_optical_calibrated", "SiIV1393_not_optical_calibrated", "NV1238_not_optical_calibrated",
 #                               "CIV1548_not_optical_calibrated", "HeII1640_not_optical_calibrated"], index_map="UV", include_mass=False, create_tex_file=True)


get_fluoreszenz_table()