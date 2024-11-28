from pathlib import Path
import numpy as np

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

                text_file_path = str(one_dim_correlation_path / continuum / "lineCorrelations_ICCF.txt")
                # Line and Continuum names should not have spaces
                with open(text_file_path, "r") as file:
                    header = file.readline().strip().split(" ")
                    correlation_data = np.loadtxt(text_file_path).T

                continua_lines = ["time shift (tau)"] + header[5:]

                correlation_data_dict = dict()

                for i, name in enumerate(continua_lines):
                    correlation_data_dict[name] = correlation_data[i]

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


import_1d_lightcurve_data()