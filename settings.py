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
