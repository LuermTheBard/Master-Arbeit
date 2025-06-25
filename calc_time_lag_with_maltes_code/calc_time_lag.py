from pathlib import Path

import numpy as np

from Malte_get_BH_mass import Line
from plot_utils import print_table_for_one_reference, print_table_for_multiple_reference, save_centroid_as_txt
from import_data import find_prime_data_folder
from settings import FWHM_RMS, FWHM_ERR


def calc_centroid_malte_code(campaign, continuum, lines=None, include_mass=True, create_tex_file=True, index_map='optical'):

    data_folder = find_prime_data_folder()
    base_path = Path(
        rf'{data_folder}\campaigns\{campaign}\calc_time_lag_ccfs\{continuum}'
    )
    if lines is None:
        # Definiere die Liniennamen
        lines = ['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685', 'OI8446']

    # Lade die lineCorrelations-Daten
    line_correlations_path = base_path / "lineCorrelations_ICCF.txt"
    lightcurve_correlations_path = base_path / "lightcurveCorrelations_ICCF.txt"
    try:
        lineCorrelations = np.loadtxt(line_correlations_path)
    except Exception as e:
        print(f"❌ Fehler beim Laden der Datei {line_correlations_path}: {e}")
        return None

    if lightcurve_correlations_path.exists():
        try:
            lightcurveCorrelations = np.loadtxt(lightcurve_correlations_path)
            combined = np.hstack((lineCorrelations, lightcurveCorrelations[:, 1:]))
        except Exception as e:
            print(f"⚠️ Datei {lightcurve_correlations_path} konnte nicht verarbeitet werden: {e}")
            print("➡️ Verwende nur lineCorrelations.")
            combined = lineCorrelations
    else:
        print(f"ℹ️ Datei {lightcurve_correlations_path} nicht gefunden. Verwende nur lineCorrelations.")
        combined = lineCorrelations

    # Lade alle Zentroid- und Peak-Daten in ein Dictionary
    data = {}
    for line in lines:
        try:
            cents_path = base_path / f"calculatedCentroids{continuum}_{line}_ICCF.txt"
            peaks_path = base_path / f"peakDistribution_{continuum}_{line}_ICCF.txt"

            # Falls eine Datei fehlt, wird sie übersprungen
            data[line] = {
                "centroids": np.loadtxt(cents_path) if cents_path.exists() else None,
                "peaks": np.loadtxt(peaks_path) if peaks_path.exists() else None
            }
        except Exception as e:
            print(f"⚠️ Warnung: Fehler beim Laden der Dateien für {line}: {e}")
            continue  # Statt `return`, damit andere Linien geladen werden

    # Index-Mapping für die Spalten von combined
    if index_map == 'optical':
        index_map = {
            'HDelta': 1, 'HGamma': 2, 'HeII4685': 3, 'HBeta': 4,
            'HeI5875': 5, 'HAlpha': 6, 'HeI5015': 7, 'OI8446': 8,
            'HeI4471': 9, 'HeI7065': 10, 'OIII5007': 11, "LyAlpha": 12,
            "UVW2": 13, "Cont1150_not_optical_calibrated": 14,
            "LyAlpha_not_optical_calibrated": 15,
            "HBeta_not_optical_calibrated": 16,
            "OI8446_not_optical_calibrated": 17
        }
    elif index_map == 'UV':
        index_map = {
            "LyAlpha_not_optical_calibrated":1,
            "SiIV1393_not_optical_calibrated":2,
            "NV1238_not_optical_calibrated":3,
            "CIV1548_not_optical_calibrated": 4,
            "HeII1640_not_optical_calibrated": 5,
            "OIII]1660_not_optical_calibrated":6
        }
    else:
        print(f"⚠️ Warnung: please define index_map. Options:'optical' or 'UV'.")
        return None

    # Erstelle die Line-Objekte
    line_objects = []
    for line in lines:
        if line in index_map and data[line]["centroids"] is not None and data[line]["peaks"] is not None:
            line_obj = Line(
                line,  # Name der Linie
                FWHM_RMS.get(line, 0.0),  # FWHM (rms)
                FWHM_ERR.get(line, 0.0),  # Sigma (rms)
                np.vstack((combined[:, 0], combined[:, index_map[line]])).T,
                data[line]["centroids"],
                data[line]["peaks"]
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
        data_light_curve_lines_dict[reference] = calc_centroid_malte_code(campaign, reference, lines, include_mass=True, create_tex_file=True)

    print_table_for_multiple_reference(output_filename, data_light_curve_lines_dict, include_mass=True)


#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150_not_optical_calibrated", lines=["HBeta", "LyAlpha", "OI8446", "HBeta_not_optical_calibrated", "LyAlpha_not_optical_calibrated", "OI8446_not_optical_calibrated"], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1450_not_optical_calibrated", lines=["HBeta", "LyAlpha", "OI8446", "HBeta_not_optical_calibrated", "LyAlpha_not_optical_calibrated", "OI8446_not_optical_calibrated"], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "LyAlpha_not_optical_calibrated", lines=["HBeta", "OI8446", "HBeta_not_optical_calibrated", "OI8446_not_optical_calibrated"], include_mass=False)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "LyAlpha", lines=["HBeta", "OI8446"], include_mass=False)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150", lines=['HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685'], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "Cont1150_not_optical_calibrated", lines=['HeI5875', 'HeI7065', 'HeI4471', 'HeI5015', 'HeII4685'], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "HBeta", lines=["OI8446", "OI8446_not_optical_calibrated"], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "HBeta_not_optical_calibrated", lines=["OI8446", "OI8446_not_optical_calibrated"], include_mass=True)
#calc_centroid_malte_code("NGC4593_optical_calibrated", "UVW2", lines=['HAlpha', 'HBeta', 'HGamma', 'HDelta', 'LyAlpha','HeI5875', 'HeII4685', 'OI8446'], include_mass=True, create_tex_file=True)
calc_centroid_malte_code("NGC4593_not_optical_calibrated", "UVW2",
                         lines=["LyAlpha_not_optical_calibrated", "SiIV1393_not_optical_calibrated", "NV1238_not_optical_calibrated",
                                "CIV1548_not_optical_calibrated", "HeII1640_not_optical_calibrated"], index_map="UV", include_mass=False, create_tex_file=True)


#get_fluoreszenz_table()