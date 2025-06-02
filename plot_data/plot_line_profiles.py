import numpy as np
from astropy.constants.codata2018 import c
from scipy.interpolate import interp1d

from settings import CENTRAL_WAVELENGTH

line_mapping = {
    'HAlpha': 'Hα',
    'HBeta': 'Hβ',
    'HGamma': 'Hγ',
    'HDelta': 'Hδ',
    'HeI5875': 'He I 5875',
    'HeI7065': 'He I 7065',
    'HeI4471': 'He I 4471',
    'HeI5015': 'He I 5015',
    'HeII4685': 'He II 4685',
    'OI8446': 'O I 8446'
}


def subtract_continuum(wavelength, intensity, line_wavelength, left_range, right_range):
    """
    Subtrahiert das Pseudokontinuum von einer Emissionslinie in einem Spektrum und normalisiert auf das Maximum in einer gegebenen Umgebung.

    :param wavelength: Array der Wellenlängen
    :param intensity: Array der Intensitätswerte
    :param line_wavelength: Zentrale Wellenlänge der Linie
    :param left_range: Tupel (min, max) des linken Bereichs für die Kontinuumsschätzung
    :param right_range: Tupel (min, max) des rechten Bereichs für die Kontinuumsschätzung
    :return: (korrigierte Intensität, Pseudokontinuum)
    """
    left_mask = (wavelength > left_range[0]) & (wavelength < left_range[1])
    right_mask = (wavelength > right_range[0]) & (wavelength < right_range[1])

    left_mean = (np.mean(wavelength[left_mask]), np.mean(intensity[left_mask]))
    right_mean = (np.mean(wavelength[right_mask]), np.mean(intensity[right_mask]))

    continuum_fit = interp1d([left_mean[0], right_mean[0]], [left_mean[1], right_mean[1]], kind="linear",
                             fill_value="extrapolate")
    continuum = continuum_fit(wavelength)

    corrected_intensity = intensity - continuum

    # Bestimme das Maximum innerhalb des Bereichs ±50 um line_wavelength
    norm_range = (line_wavelength - 50, line_wavelength + 50)
    norm_mask = (wavelength > norm_range[0]) & (wavelength < norm_range[1])
    if np.any(norm_mask):
        max_value = np.max(corrected_intensity[norm_mask])
    else:
        max_value = np.max(corrected_intensity)  # Falls der Bereich leer ist, fallback auf das globale Maximum

    corrected_intensity /= max_value

    return corrected_intensity, continuum


def convert_to_velocity(wavelength, line_wavelength):
    """
    Wandelt Wellenlängenwerte in Geschwindigkeitswerte um.
    """
    c_km_s = c.to('km/s').value  # Lichtgeschwindigkeit in km/s
    return (wavelength - line_wavelength) / line_wavelength * c_km_s


def transform_wavelength_to_velocity_and_cut(wavelength, intensity, line_name, velocity_range=None, filename=None):
    """
    Transformiert die Wellenlängenachse in den Geschwindigkeitsraum, normalisiert die Intensität
    auf das Maximum und schneidet optional die Daten auf einen Bereich um 0.

    :param wavelength: Array der Wellenlängen
    :param intensity: Array der Intensitätswerte
    :param line_name: Zentrale Wellenlänge der Linie
    :param velocity_range: Tupel (min, max) zur Begrenzung des Geschwindigkeitsbereichs; min muss negativ, max positiv sein
    :param filename: Falls definiert, wird das Ergebnis in eine Datei gespeichert
    :return: (Geschwindigkeiten, normalisierte Intensität)
    """

    line_wavelength = CENTRAL_WAVELENGTH[line_name]

    # Transformation der Wellenlängen in den Geschwindigkeitsraum
    velocity = convert_to_velocity(wavelength, line_wavelength)

    intensity = normalize_to_maximum(intensity, line_wavelength, wavelength)

    intensity, velocity = cut_normalized_line_out(intensity, velocity, velocity_range)

    # Falls eine Datei angegeben ist, speichern
    if filename is not None:
        with open(str(filename), 'w') as file:
            file.write("# velocity space (km/s) \t normalized flux\n")
            for v, i in zip(velocity, intensity):
                file.write(f"{v}\t{i}\n")

    return velocity, intensity


def normalize_to_maximum(intensity, line_wavelength, wavelength):
    # Normalisierung der Intensität auf das Maximum
    if np.max(intensity) != 0:
        # Bestimme das Maximum innerhalb des Bereichs ±10 um line_wavelength
        norm_range = (line_wavelength - 10, line_wavelength + 10)
        norm_mask = (wavelength > norm_range[0]) & (wavelength < norm_range[1])
        if np.any(norm_mask):
            max_value = np.max(intensity[norm_mask])
        else:
            max_value = np.max(intensity)  # Falls der Bereich leer ist, fallback auf das globale Maximum

        intensity /= max_value
    return intensity


def cut_normalized_line_out(intensity, velocity, velocity_range):
    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if velocity_range is not None:
        min_v, max_v = velocity_range
        min_v = min(min_v, -abs(min_v))  # Sicherstellen, dass min negativ ist
        max_v = max(max_v, abs(max_v))  # Sicherstellen, dass max positiv ist
        mask = (velocity >= min_v) & (velocity <= max_v)
        velocity = velocity[mask]
        intensity = intensity[mask]
    return intensity, velocity


def cut_line_out(intensity, wavelength, wavelength_range):
    # Falls ein Bereich gegeben ist, schneide die Daten zurecht
    if wavelength_range is not None:
        min_v, max_v = wavelength_range
        mask = (wavelength >= min_v) & (wavelength <= max_v)
        wavelength = wavelength[mask]
        intensity = intensity[mask]
    return intensity, wavelength


def save_velocity_data_to_txt(filename, velocity, intensity):
    """
    Speichert die Geschwindigkeits- und Intensitätswerte in eine TXT-Datei im gewünschten Format.

    :param filename: Name der Datei, in die gespeichert werden soll
    :param velocity: Array der Geschwindigkeitswerte
    :param intensity: Array der Intensitätswerte
    """
    with open(filename, 'w') as file:
        file.write("# velocity space (km/s) \t normalized flux\n")
        for v, i in zip(velocity, intensity):
            file.write(f"{v}\t{i}\n")


