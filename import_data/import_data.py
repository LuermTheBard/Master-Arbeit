import warnings
from pathlib import Path
import numpy as np

from astropy.io import fits

from plot_data.plot_line_profiles import line_mapping

try:
    import tomllib
except ImportError:
    import tomli as tomllib

DATA_PATH = Path.cwd().parent / "data"


def find_prime_data_folder():
    current_working_directory = Path.cwd()

    # Suche in den aktuellen und zwei übergeordneten Verzeichnissen
    for i in range(3):
        potential_data_folder = current_working_directory / "data"
        if potential_data_folder.exists() and potential_data_folder.is_dir():
            print(f"'data' folder found at: {potential_data_folder}")
            return potential_data_folder
        current_working_directory = current_working_directory.parent

    # Falls 'data' nicht in den ersten drei Ebenen gefunden wurde, rekursive Suche im Baum
    for folder in Path.cwd().rglob("data"):
        if folder.is_dir():
            print(f"'data' folder found at: {folder}")
            return folder

    print("No 'data' folder found.")
    return None


def import_1d_correlation_data():
    data_path = find_prime_data_folder()

    campains_path = data_path / "campaigns"

    galaxie_campaigns = [f.name for f in campains_path.iterdir() if f.is_dir()]
    galaxie_campaigns_dict = dict()
    for campaign in galaxie_campaigns:
        continua_dict = dict()
        campaign_path = (campains_path / campaign)
        if "1DCorrelations" in [f.name for f in campaign_path.iterdir()]:

            one_dim_correlation_path = (campaign_path / "1DCorrelations")

            continua = [f.name for f in one_dim_correlation_path.iterdir() if f.is_dir()]

            for continuum in continua:

                line_correlation_file_path = one_dim_correlation_path / continuum / "lineCorrelations_ICCF.txt"

                lightcurve_correlation_file_path = one_dim_correlation_path / continuum / "lightcurveCorrelations_ICCF.txt"

                with open(str(line_correlation_file_path), "r") as file:
                    header_line = file.readline().strip().split(" ")
                    line_correlation_data = np.loadtxt(line_correlation_file_path).T


                if lightcurve_correlation_file_path.exists():

                    with open(str(lightcurve_correlation_file_path), "r") as file:
                        header_lightcurve = file.readline().strip().split(" ")
                        lightcurve_correlation_data = np.loadtxt(lightcurve_correlation_file_path).T
                else:
                    print(f"SKIPPED: lightcurveCorrelations file not found: {str(lightcurve_correlation_file_path)}")

                continua_lines = ["time shift (tau)"] + header_line[5:]

                continua_lightcurves = header_lightcurve[4:]

                correlation_data_dict = dict()

                for i, name in enumerate(continua_lines):
                    correlation_data_dict[name] = line_correlation_data[i]

                for i, name in enumerate(continua_lightcurves):
                    if i == 0:
                        continue
                    correlation_data_dict[name] = lightcurve_correlation_data[i]



                continua_dict[str(continuum)] = correlation_data_dict

        galaxie_campaigns_dict[str(campaign)] = continua_dict

    return galaxie_campaigns_dict


def load_dopplershift_data_from_toml(toml_name, toml_path=None):
    if toml_path is None:
        data_path = find_prime_data_folder()

        toml_path = data_path / "dopplershift_calc_data" / toml_name
    else:
        toml_path = toml_path / toml_name

    with open(str(toml_path), "rb") as file:
        dopplershift_data = tomllib.load(file)

    return dopplershift_data


def process_light_curves(light_curves, one_dim_lightcurves_path):
    """
    Verarbeitet Light-Curve-Dateien und erstellt ein Dictionary mit den Daten.

    Args:
        light_curves (list): Liste der Dateinamen der Light-Curves.
        one_dim_lightcurves_path (Path): Pfad zum Verzeichnis der Light-Curve-Dateien.

    Returns:
        dict: Ein Dictionary mit den verarbeiteten Light-Curve-Daten.
    """
    result_dict = {}

    for curve in light_curves:
        text_file_path = str(one_dim_lightcurves_path / curve)

        with open(text_file_path, "r") as file:
            header = file.readline().strip().split(",")
            light_curve_data = np.loadtxt(text_file_path).T

        curve_name = header[0].split(" ")[1]
        array_names = ['timestamps [MJD]', 'fluxes [ergs/s/cm2/A]', 'fluxerrs [ergs/s/cm2/A]']

        light_curve_data_dict = {name: light_curve_data[i] for i, name in enumerate(array_names)}
        result_dict[curve_name] = light_curve_data_dict

    return result_dict


def import_1d_lightcurve_data():
    """
    Importiert 1D-Light-Curve-Daten aus den Kampagnenverzeichnissen und erstellt ein verschachteltes Dictionary.

    Returns:
        dict: Ein Dictionary mit den Daten aller Kampagnen, geordnet nach Lines und Continua.
    """
    data_path = find_prime_data_folder()
    campaigns_path = data_path / "campaigns"

    galaxie_campaigns = [f.name for f in campaigns_path.iterdir() if f.is_dir()]
    galaxie_campaigns_dict = {}

    for campaign in galaxie_campaigns:
        continua_dict = {}
        line_dict = {}
        campaign_path = campaigns_path / campaign

        if "1DLightCurves" in [f.name for f in campaign_path.iterdir()]:
            one_dim_lightcurves_path = campaign_path / "1DLightCurves"

            # Light-Curves identifizieren
            continua_light_curves = [f.name for f in one_dim_lightcurves_path.glob('Cont*.txt')]
            line_light_curves = [f.name for f in one_dim_lightcurves_path.glob("*.txt") if "Cont" not in f.name]

            # Daten verarbeiten
            continua_dict = process_light_curves(continua_light_curves, one_dim_lightcurves_path)
            line_dict = process_light_curves(line_light_curves, one_dim_lightcurves_path)

        # Ergebnisse der aktuellen Kampagne speichern
        galaxie_campaigns_dict[campaign] = {"lines": line_dict, "continua": continua_dict}

    return galaxie_campaigns_dict


def import_fits_data(fits_folder=None):
    """
    Importiert FITS-Daten aus einem Verzeichnis und erstellt ein verschachteltes Dictionary.
    Gibt Warnungen aus, falls Daten oder notwendige Header-Parameter fehlen.

    Returns:
        dict: Ein Dictionary mit den Daten, Headern und x-Achsen-Werten aller FITS-Dateien im Verzeichnis.
    """

    data_path = find_prime_data_folder()

    if fits_folder is None:

        fits_folder = data_path / "fits"

    else:

        fits_folder = data_path / Path(fits_folder)

    fits_files = list(fits_folder.glob("*.fits"))

    fits_data_dict = {}

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            # Ein Dictionary für die aktuelle FITS-Datei erstellen
            file_data = {
                "data": [],
                "headers": [],
                "x_axis": []  # Hier speichern wir die x-Achsen-Werte
            }

            # Jede HDU der Datei verarbeiten
            for i, hdu in enumerate(hdul):
                if hdu.data is None:
                    warnings.warn(
                        f"Keine Daten in HDU-{i} der Datei '{fits_file.name}'.",
                        UserWarning
                    )
                    file_data["data"].append(None)
                else:
                    file_data["data"].append(hdu.data)

                # Header speichern
                file_data["headers"].append(dict(hdu.header))

                # X-Achsen-Werte berechnen, falls WCS-Parameter vorhanden sind
                header = hdu.header
                if "CRVAL1" in header and "CD1_1" in header and "CRPIX1" in header:
                    crval1 = header.get("CRVAL1", 0.0)
                    cd1_1 = header.get("CD1_1", 1.0)
                    crpix1 = header.get("CRPIX1", 1.0)

                    if hdu.data is not None:
                        n_pixels = hdu.data.shape[-1]  # Anzahl der Pixel entlang der x-Achse
                        x_axis = (np.arange(n_pixels) - (crpix1 - 1)) * cd1_1 + crval1
                        file_data["x_axis"].append(x_axis.tolist())
                    else:
                        warnings.warn(
                            f"Keine Daten verfügbar, um x-Achse für HDU-{i} in Datei '{fits_file.name}' zu berechnen.",
                            UserWarning
                        )
                        file_data["x_axis"].append(None)
                else:
                    warnings.warn(
                        f"Fehlende WCS-Parameter (CRVAL1, CD1_1, CRPIX1) in HDU-{i} der Datei '{fits_file.name}'.",
                        UserWarning
                    )
                    file_data["x_axis"].append(None)

            # Ergebnisse speichern, Schlüssel ist der Dateiname
            fits_data_dict[fits_file.name] = file_data

    return fits_data_dict


def import_line_profile_data(normalized=None):

    data_path = find_prime_data_folder()

    avg_data_dict = dict()
    rms_data_dict = dict()

    if "LineProfiles" in [f.name for f in data_path.iterdir()]:
        line_profile_path = data_path / "LineProfiles"

        avg_line_profiles = [f.name for f in line_profile_path.glob('*_avg_*')]
        rms_line_profiles = [f.name for f in line_profile_path.glob('*_rms_*')]

        if normalized:
            sorted_avg_line_profiles = [f for f in avg_line_profiles if "normalized" in f]
            sorted_rms_line_profiles = [f for f in rms_line_profiles if "normalized" in f]
        else:
            sorted_avg_line_profiles = [f for f in avg_line_profiles if "normalized" not in f]
            sorted_rms_line_profiles = [f for f in rms_line_profiles if "normalized" not in f]

        avg_data_dict = process_line_profile_data(sorted_avg_line_profiles, line_profile_path)
        rms_data_dict = process_line_profile_data(sorted_rms_line_profiles, line_profile_path)

    galaxie_campaigns_dict = {"avg": avg_data_dict, "rms": rms_data_dict}

    return galaxie_campaigns_dict


def process_line_profile_data(line_profiles_list, line_profile_path):
    result_dict = dict()

    for item in line_profiles_list:

        line_name = item.split("_", 1)[0]

        if "-" in item:
            base, cont1, cont2 = item.rsplit("-", 2)
            pseudo_conts = (cont1, cont2.removesuffix(".txt"))
        else:
            pseudo_conts = None

        file_path = str(line_profile_path / item)

        with open(file_path, "r") as file:
            header = file.readline().strip().split(" \t ")
            line_profile_data = np.loadtxt(file_path).T

        array_keys = [header[0].split("# ")[1], header[1]]

        line_profile_data_dict = {name: line_profile_data[i] for i, name in enumerate(array_keys)}

        local_data_dict = {"pseudo_conts": pseudo_conts, "data_dict": line_profile_data_dict}
        result_dict[line_name] = local_data_dict

    return result_dict

