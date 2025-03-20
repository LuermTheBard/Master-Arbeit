from pathlib import Path

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

F_MEAN = 13.15 * 10 ** (-15)

F_SIGMA = 0.23 * 10 ** (-15)

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
    "Hα": {"position": 6562.82, "text_vertical_shift": -10, "slanted": False, "text_shift": -60, "show_no_tick_avg": True, "tick_shift_rms": 1},
    "Hβ": {"position": 4861.33, "text_vertical_shift": -0.7, "slanted": False, "show_no_tick_avg": True, "tick_shift_rms": 0.6},
    "Hγ": {"position": 4340.47, "text_vertical_shift": 0.1, "tick_shift_avg": 0.31, "text_shift": -20, "slanted": False, "tick_shift_rms": 0.57},
    "Hδ": {"position": 4101.74, "text_vertical_shift": 0.1, "slanted": False},
    "Hε": {"position": 3970.08, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.6},

    # "[Ne V] 3345": {"position": 3345.82, "text_vertical_shift": 0.01, "slanted": False},
    # "[Ne V] 3425": {"position": 3425.88, "text_vertical_shift": 0.01, "slanted": False},
    "[Ne III] 3868": {"position": 3868.76, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},

    # "He I 3487": {"position": 3487.72, "text_vertical_shift": 0.1, "slanted": True},
    "He I 4471": {"position": 4471.48, "text_vertical_shift": 0.1, "slanted": False, "text_shift": -20},
    "He I 5875": {"position": 5875.6, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.3},
    "He I 5015": {"position": 5015.67, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_avg": 0.2,"text_shift": 40, "show_no_tick_rms": True},
    "He I 7065": {"position": 7065.2, "text_vertical_shift": 0.1, "slanted": False},
    "He II 4685": {"position": 4685.7, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.4},
    # "He I 3187": {"position": 3187.74, "text_vertical_shift": 0.2, "slanted": True},
    # "He II 3203": {"position": 3203.1, "text_vertical_shift": 0.1, "slanted": True},

    "[O III] 4363": {"position": 4363.21, "text_vertical_shift": 0.1, "slanted": False, "text_shift": 30},
    "[O III] 4958": {"position": 4958.91, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},
    "[O III] 5006": {"position": 5006.84, "text_vertical_shift": -2, "text_shift": -40, "slanted": False, "show_no_tick_avg": True, "show_no_tick_rms": True},

    # "[O I] 6364": {"position": 6363.77, "text_vertical_shift": 0.1, "slanted": False},
    # "[Fe X] 6375": {"position": 6374.51, "text_vertical_shift": 0.1, "slanted": False},
    "Fe II 6516": {"position": 6516.08, "text_vertical_shift": 0.1, "slanted": False, "text_shift": -30, "show_no_tick_rms": True, "tick_shift_avg": 1.5},
    "[Fe VII] 5721": {"position": 5721.7, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},
    "[Fe VII] 6087": {"position": 6087, "text_vertical_shift": 0.1, "slanted": False, "show_no_tick_rms": True},
    # "[Fe VII] 3586": {"position": 3586.32, "text_vertical_shift": 0.1, "slanted": True},

    # "[S II] 6716": {"position": 6716.44, "text_vertical_shift": 0.1, "slanted": False},
    # "[S II] 6731": {"position": 6730.81, "text_vertical_shift": 0.1, "slanted": False},

    # "[N II] 6548": {"position": 6548.05, "text_vertical_shift": 0.1, "slanted": False, "text_shift": -40},
    # "[N II] 6584": {"position": 6583.46, "text_vertical_shift": 0.1, "slanted": False, "text_shift": 40},

    "[Ar III] 7135": {"position": 7135.79, "text_vertical_shift": 0.1, "tick_shift_avg": 0.37, "slanted": False, "show_no_tick_rms": True},

    # "O III 3132": {"position": 3132.79, "text_vertical_shift": 0.1, "slanted": True},
    "O I 8446": {"position": 8446.35, "text_vertical_shift": 0.1, "slanted": False, "tick_shift_rms": 0.29},
    "[O I] 6300": {"position": 6300.30, "text_vertical_shift": 0.1, "tick_shift_avg": 0.43, "slanted": False, "show_no_tick_rms": True},

    # "Ca II 8498": {"position": 8498.02, "text_vertical_shift": 0.5, "slanted": False},
    # "Ca II 8542": {"position": 8542.09, "text_vertical_shift": 0, "slanted": False},
    # "Ca II 8662": {"position": 8662.14, "text_vertical_shift": 0.1, "slanted": False},
}
All_LINE_GROUPS = {
    "Fe II": {"position": [4489,
                           4629.33],
              "tick_vertical_shift_avg": 0.8},

    "Fe II ": {"position": [5169.03,
                            5336.18],
               "tick_vertical_shift_avg": 0.5},

    "[S II] 6716,6730 ": {"position": [6716.44,
                                       6730.81],
                          "tick_vertical_shift_avg": 0.1,
                          "show_in_rms": False},

    "[O I] 6364, [Fe X] 6375": {"position": [6363.77,
                                             6374.51],
                                "tick_vertical_shift_avg": 0.2,
                                "show_in_rms": False},
    "Ca II": {"position": [8498.02,
                           8542.09,
                           8662.14],
              "tick_vertical_shift_avg": 0.34,
              "tick_vertical_shift_rms": 0.23,
              "all_lines": True,
              "show_in_rms": True},

}
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
    'OIII5007': 5007,
}
