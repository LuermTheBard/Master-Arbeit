from pathlib import Path

import numpy as np

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "default_output"
COLORCODE_CONTINUA = {
    'Cont4010': (100, 0, 128, 255),
    'Cont4200': (80, 0, 255, 255),
    'Cont4440': (0, 34, 255, 255),
    'Cont4765': 'dodgerblue',
    'Cont5100': 'tab:green',
    'Cont5600': 'gold',
    'Cont6045': (255, 180, 0, 255),
    'Cont6110': (255, 99, 0, 255),
    'Cont6880': (222, 0, 0, 255),
    'Cont7390': (190, 0, 0, 255),
    'Cont8015': (165, 0, 0, 255),
    'Cont8900': (139, 0, 0, 255)
}

BASE_MJD = 57581.66

F_MEAN = 1.114 * 10 ** (-13)

F_SIGMA = 1.983 * 10 ** (-15)

F_VAR = 1.429 * 10 ** (-2)

F_REL = 0.0472

VALUES_CONTINUA = {'Cont1150': (1140, 1160),
                   'Cont4010': (4026, 4033),
                   'Cont4200': (4197, 4220),
                   'Cont4440': (4435, 4450),
                   'Cont4765': (4762, 4774),
                   'Cont5100': (5085, 5112),
                   'Cont5600': (5645, 5653),
                   'Cont6045': (6044, 6057),
                   'Cont6110': (6107, 6129),
                   'Cont6880': (6861, 6900),
                   'Cont7390': (7382, 7405),
                   'Cont8015': (8005, 8031),
                   'Cont8900': (8864, 8955)}


def normalize_color_values(colorcode_dict):
    """
    Normalisiert RGBA-Tupel im Bereich 0-255 zu RGBA-Tupeln im Bereich 0-1,
    falls erforderlich.
    """
    normalized_dict = {}
    for key, color in colorcode_dict.items():
        if isinstance(color, tuple) and len(color) == 4:
            # Normiere die Farbwerte in den Bereich 0-1
            normalized_dict[key] = tuple(c / 255.0 for c in color)
        else:
            # Wenn es sich um einen gültigen Farbstring handelt, behalte ihn bei
            normalized_dict[key] = color
    return normalized_dict


COLORCODE_CONTINUA_NORMALIZED = normalize_color_values(COLORCODE_CONTINUA)
All_LINES = {
    # UV
    r"$\mathrm{Ly}\alpha$": {
        "position": 1215.67, "text_vertical_shift": -10, "slanted": False,
        "show_no_tick_avg": True, "text_shift": -15, "tick_shift_rms": 3
    },

    r"$\mathrm{N\,IV]}\,\lambda\,1486$": {
        "position": 1486.49, "text_vertical_shift": 3, "slanted": False,
        "tick_shift_avg": 1.5, "text_shift": 0, "tick_shift_rms": 1
    },

    r"$\mathrm{He\,II}\,\lambda\,1640$": {
        "position": 1640.42, "text_vertical_shift": 3, "slanted": False,
        "tick_shift_avg": 2.5, "text_shift": 0, "tick_shift_rms": 3
    },

    # Balmer
    r"$\mathrm{H}\alpha$": {
        "position": 6562.82, "text_vertical_shift": -10, "slanted": False,
        "text_shift": 90, "show_no_tick_avg": True, "tick_shift_rms": 1
    },
    r"$\mathrm{H}\beta$": {
        "position": 4861.33, "text_vertical_shift": -0.7, "slanted": False,
        "show_no_tick_avg": True, "tick_shift_rms": 0.6
    },
    r"$\mathrm{H}\gamma$": {
        "position": 4340.47, "text_vertical_shift": 0.1, "tick_shift_avg": 0.31,
        "text_shift": -30, "slanted": False, "tick_shift_rms": 0.57
    },
    r"$\mathrm{H}\delta$": {"position": 4101.74, "text_vertical_shift": 0.1, "slanted": False},
    r"$\mathrm{H}\epsilon$": {"position": 3970.08, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.6},

    # Forbidden / nebular etc.
    r"$[\mathrm{Ne\,III}]\,\lambda\,3868$": {
        "position": 3868.76, "text_vertical_shift": 0.1, "slanted": False,
        "show_no_tick_rms": True
    },

    # He I series
    r"$\mathrm{He\,I}\,\lambda\,4471$": {"position": 4471.48, "text_vertical_shift": 0.1, "slanted": False, "text_shift": -20},
    r"$\mathrm{He\,I}\,\lambda\,5876$": {"position": 5875.6, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.3},
    r"$\mathrm{He\,I}\,\lambda\,5016$": {"position": 5015.67, "text_vertical_shift": -1, "slanted": False, "tick_shift_avg": 1, "text_shift": 70, "show_no_tick_rms": True},
    r"$\mathrm{He\,I}\,\lambda\,7065$": {"position": 7065.2, "text_vertical_shift": 0.1, "slanted": False, "text_shift": -20},

    # He II optical
    r"$\mathrm{He\,II}\,\lambda\,4686$": {"position": 4685.7, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.4},

    # Oxygen
    r"$[\mathrm{O\,III}]\,\lambda\,4363$": {"position": 4363.21, "text_vertical_shift": 0.1, "slanted": False, "text_shift": 40},
    r"$[\mathrm{O\,III}]\,\lambda\,4958$": {"position": 4958.91, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},
    r"$[\mathrm{O\,III}]\,\lambda\,5006$": {
        "position": 5006.84, "text_vertical_shift": -2.5, "text_shift": 80, "slanted": False,
        "show_no_tick_avg": True, "show_no_tick_rms": True
    },

    r"$[\mathrm{O\,II}]\,\lambda\,7320,7330$": {
        "position": 7325, "slanted": False, "show_no_tick_rms": True
    },

    # Iron high-ionization
    r"$[\mathrm{Fe\,VII}]\,\lambda\,5721$": {"position": 5721.7, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},
    r"$[\mathrm{Fe\,VII}]\,\lambda\,6087$": {"position": 6087, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},
    r"$[\mathrm{Fe\,XI}]\,\lambda\,7892$": {"position": 7892, "text_vertical_shift": 0.1, "text_shift": 20, "tick_shift_avg": 0.37, "slanted": False, "show_no_tick_rms": True},

    # Argon
    r"$[\mathrm{Ar\,III}]\,\lambda\,7135$": {"position": 7135.79, "text_vertical_shift": 0.1, "text_shift": 20, "tick_shift_avg": 0.37, "slanted": False, "show_no_tick_rms": True},
    r"$[\mathrm{Ar\,III}]\,\lambda\,7751$": {"position": 7751, "text_vertical_shift": 0.1, "text_shift": 20, "tick_shift_avg": 0.37, "slanted": False, "show_no_tick_rms": True},

    # Oxygen I lines
    r"$\mathrm{O\,I}\,\lambda\,8446$": {"position": 8446.35, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.29},
    r"$[\mathrm{O\,I}]\,\lambda\,6300$": {"position": 6300.30, "text_vertical_shift": 0.1, "tick_shift_avg": 0.43, "slanted": False, "show_no_tick_rms": True},
}


All_LINE_GROUPS = {
    r"$\mathrm{N\,V}\,\lambda\lambda\,1238,\,1242$": {
        "position": [1238.82, 1242.8],
        "tick_vertical_shift_avg": 2,
        "tick_vertical_shift_rms": 7,
        "show_in_rms": True,
        "text_vertical_shift": 2
    },

    r"$\mathrm{Si\,IV}\,\lambda\lambda\,1393,\,1402,\,\mathrm{O\,IV]}$": {
        "position": [1393.75, 1402.7],
        "tick_vertical_shift_avg": 2,
        "tick_vertical_shift_rms": 2.5,
        "show_in_rms": True,
        "text_vertical_shift": 2
    },

    r"$\mathrm{C\,IV}\,\lambda\lambda\,1548,\,1550$": {
        "position": [1548.19, 1550.77],
        "tick_vertical_shift_avg": 2,
        "tick_vertical_shift_rms": 10,
        "show_in_rms": True,
        "show_in_avg": False,
        "text_vertical_shift": -15,
        "text_horizontal_shift": -15
    },

    r"$\mathrm{O\,III]}\,\lambda\lambda\,1660,\,1666$": {
        "position": [1660.80, 1666.15],
        "tick_vertical_shift_avg": 1,
        "tick_vertical_shift_rms": 1,
        "show_in_rms": True,
        "text_vertical_shift": 2
    },

    r"$\mathrm{Fe\,II}$": {
        "position": [4489, 4629.33],
        "tick_vertical_shift_avg": 0.8
    },

    r"$\mathrm{Fe\,II}\,$": {
        "position": [5169.03, 5336.18],
        "tick_vertical_shift_avg": 0.5
    },

    r"$[\mathrm{S\,II}]\,6716,\,6730$": {
        "position": [6716.44, 6730.81],
        "tick_vertical_shift_avg": 0.1,
        "show_in_rms": False
    },

    r"$[\mathrm{O\,I}]\,6364,\,[\mathrm{Fe\,X}]\,6375$": {
        "position": [6363.77, 6374.51],
        "tick_vertical_shift_avg": 0.2,
        "show_in_rms": False,
        "text_horizontal_shift": 15
    },

    r"$\mathrm{Ca\,II}$": {
        "position": [8498.02, 8542.09, 8662.14],
        "tick_vertical_shift_avg": 0.34,
        "tick_vertical_shift_rms": 0.23,
        "all_lines": True,
        "show_in_rms": True
    },
}

def calculate_velocity_errors(lines_wavelengths):
    """
    Gibt ein Dictionary mit Liniennamen und geschätztem Fehler im Velocity Space (km/s) zurück.
    """
    c = 299792.458
    velocity_errors = {}

    for line, central_wavelength in lines_wavelengths.items():
        # Grating auswählen
        if 1119 <= central_wavelength <= 1715:
            R = 1000  # G140L   # https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-3-gratings/first-order-grating-g140l
        elif 2888 <= central_wavelength <= 5697:
            R = 750   # G430L   # https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-3-gratings/first-order-grating-g430l
        elif 5245 <= central_wavelength <= 10233:
            R = 500   # G750L   # https://hst-docs.stsci.edu/stisihb/chapter-13-spectroscopic-reference-material/13-3-gratings/first-order-grating-g750l
        else:
            R = None

        if R:
            delta_lambda = central_wavelength / R
            delta_v = c * (delta_lambda / central_wavelength)


            velocity_errors[line] = round(delta_v, 0)
        else:
            velocity_errors[line] = None
            print(f"Warnung: Keine passende Grating-Auflösung für {line} (λ = {central_wavelength} Å) gefunden.")

    return velocity_errors






CENTRAL_WAVELENGTH = {
    'HAlpha': 6562.82,
    'HBeta': 4861.33,
    'HGamma': 4340.47,
    'HDelta': 4101.74,
    'HeI5875': 5875.6,
    'HeI7065': 7065.2,
    'HeI4471': 4471.48,
    'HeI5015': 5015.67,
    'HeII4685': 4685.7,
    'OI8446': 8446.35,
    'OIII5007': 5006.84,
    'LyAlpha_not_optical_calibrated': 1215.67,
    "SiIV1393_not_optical_calibrated": 1402,
    'NV1238_not_optical_calibrated': 1242,
    'CIV1548_not_optical_calibrated': 1550,
    'HeII1640_not_optical_calibrated': 1640.42,
    'LyAlpha': 1215.67
}

IONIZATION_POTENTIAL = {
    'HAlpha': 13.6,       # H I → H II (Wasserstoff neutral)
    'HBeta': 13.6,        # H I → H II
    'HGamma': 13.6,       # H I → H II
    'HDelta': 13.6,       # H I → H II
    'HeI5875': 24.6,      # He I → He II (Helium neutral)
    'HeI7065': 24.6,      # He I → He II
    'HeI4471': 24.6,      # He I → He II
    'HeI5015': 24.6,      # He I → He II
    'HeII4685': 54.4,     # He II → He III (Helium einfach ionisiert)
    'OI8446': 13.6,       # O I → O II (Sauerstoff neutral)
    'OIII5007': 35.1,     # O III → O IV (Sauerstoff zweifach ionisiert)
    'LyAlpha': 13.6,      # H I → H II (Lyman-Alpha)
    "SiIV1393_not_optical_calibrated": 33.5,   # Si IV → Si V (Silizium dreifach ionisiert)
    "NV1238_not_optical_calibrated": 77.5,     # N V → N VI (Stickstoff vierfach ionisiert)
    "CIV1548_not_optical_calibrated": 47.9,    # C IV → C V (Kohlenstoff dreifach ionisiert)
    "HeII1640_not_optical_calibrated": 54.4,   # He II → He III (Helium einfach ionisiert)
    "OIII]1660_not_optical_calibrated": 35.1   # O III → O IV (Sauerstoff zweifach ionisiert)
}


FWHM_RMS = {
    'HAlpha': 3111,
    'HBeta': 3437,
    'HGamma': 3852,
    'HDelta': 4893,
    'HeI5875': 3952,
    'HeI7065': 2542,
    'HeI4471': 999,
    'HeI5015': np.nan,
    'HeII4685': 5971,
    'OI8446': 3800,
    "LyAlpha_not_optical_calibrated": 4566,
    "NV1238_not_optical_calibrated": 3383,
    "SiIV1393_not_optical_calibrated": 10005,
    "CIV1548_not_optical_calibrated":  8428,
    "HeII1640_not_optical_calibrated": 6891
}

#DELTA_V = calculate_velocity_errors(CENTRAL_WAVELENGTH)



KM_PER_S_PER_PIXEL ={'CIV1548_not_optical_calibrated': 113.1,
                     'HAlpha': 222.9,
                     'HBeta': 169.3,
                     'HDelta': 200.6,
                     'HGamma': 189.6,
                     'HeI4471': 184.1,
                     'HeI5015': 164.1,
                     'HeI5875': 249.0,
                     'HeI7065': 207.1,
                     'HeII1640_not_optical_calibrated': 106.7,
                     'HeII4685': 175.6,
                     'LyAlpha_not_optical_calibrated': 144.0,
                     'NV1238_not_optical_calibrated': 141.3,
                     'OI8446': 173.2,
                     'OIII5007': 164.4,
                     'SiIV1393_not_optical_calibrated': 125.6}

FWHM_ERR = {
    'HAlpha': 250,
    'HBeta': 200,
    'HGamma': 300,
    'HDelta': 400,
    'HeI5875': 400,
    'HeII4685': 400.0,
    'OI8446': 400.0,
    'HeII1640_not_optical_calibrated':600,
    'SiIV1393_not_optical_calibrated': 1000,
    'CIV1548_not_optical_calibrated': 500,
    'LyAlpha_not_optical_calibrated': 350,
    'NV1238_not_optical_calibrated': 1000
}



DISPERSION_SIGMA_RMS = {
    'HAlpha': 1175.0, 'HBeta': 1190.0, 'HGamma': 1240.0, 'HDelta': 1560.0,
    'HeI5875': 1218, 'HeI7065': np.nan, 'HeI4471': 155, 'HeI5015': np.nan,
    'HeII4685': 1998, 'OI8446': 846, 'LyAlpha': 1752
}



SYMBOLES_AND_COLORS_FOR_LIGHTCURVES = {
    "UVW2": {"symbole": ".", "color": "darkblue", "markersize": 3, "alpha": 0.6},
    #"Cont1150_not_optical_calibrated": {"symbole": ".", "color": "#1f77b4", "markersize": 6},

    "HAlpha": {"symbole": "^", "color": "#d62728", "markersize": 4},
    "HBeta": {"symbole": ">", "color": "darkgoldenrod", "markersize": 3},
    "HGamma": {"symbole": "v", "color": "#cc6600", "markersize": 4},
    "HDelta": {"symbole": "<", "color": "#996633", "markersize": 4},

    "HeI5875": {"symbole": "s", "color": "#0059b3", "markersize": 3},           # dunkleres, satteres Blau
    "HeI7065": {"symbole": "p", "color": "#17becf", "markersize": 3},
    "HeI4471": {"symbole": "h", "color": "#aec7e8", "markersize": 3},
    "HeI5015": {"symbole": "P", "color": "#2ca02c", "markersize": 3},

    "HeII4685": {"symbole": "x", "color": "#9467bd", "markersize": 3},
    "HeII1640_not_optical_calibrated": {"symbole": "x", "color": "#7f5fbf", "markersize": 3},

    "OI8446": {"symbole": "d", "color": "#000000", "markersize": 4},
    "OIII5007": {"symbole": "*", "color": "#8c564b", "markersize": 4},
    "OIII]1660_not_optical_calibrated": {"symbole": "*", "color": "#b08a7b", "markersize": 3},

    "LyAlpha_not_optical_calibrated": {"symbole": "s", "color": "#ff7f0e", "markersize": 3},

    "SiIV1393_not_optical_calibrated": {"symbole": ">", "color": "#17becf", "markersize": 3},
    "NV1238_not_optical_calibrated": {"symbole": "^", "color": "#bcbd22", "markersize": 3},
    "CIV1548_not_optical_calibrated": {"symbole": "P", "color": "#ff9896", "markersize": 3}
}


# in prozent
ERR_CORRECTION = {"LyAlpha_not_optical_calibrated" : 50, "OI8446" : 20}

ERR_SET = {"OI8446" : 2.5}

NUMBER_MAPPING = {
 1: 'a', 2: 'b', 3: 'c', 4: 'd',
 5: 'e', 6: 'f', 7: 'g', 8: 'h',
 9: 'i', 10: 'j', 11: 'k', 12: 'l',
 13: 'm', 14: 'n', 15: 'o', 16: 'p'
}

