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
