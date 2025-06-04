import warnings
from pathlib import Path
import numpy as np

from astropy.io import fits


try:
    import tomllib
except ImportError:
    import tomli as tomllib

DATA_PATH = Path.cwd().parent / "data"


def find_prime_data_folder():
    """
    Locates the main 'data' directory within the current working directory or its ancestors.

    This function searches for a directory named 'data' in the current working directory
    and up to two parent directories. If not found, it performs a recursive search starting
    from the current working directory.

    Returns:
        Path or None: A Path object pointing to the found 'data' directory, or None if
        no such directory could be located.
    """
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
    """
    Imports 1D correlation data (ICCF results) from all galaxy campaigns.

    The function searches for the 'data/campaigns' directory structure, iterates through
    available campaigns, and loads correlation data for each continuum, including both
    line-to-continuum and lightcurve-to-continuum ICCF results.

    The expected structure inside each campaign directory is:
        campaigns/<campaign_name>/1DCorrelations/<continuum_name>/lineCorrelations_ICCF.txt
        campaigns/<campaign_name>/1DCorrelations/<continuum_name>/lightcurveCorrelations_ICCF.txt

    Returns:
        dict: A nested dictionary structured as:
              {
                  <campaign_name>: {
                      <continuum_name>: {
                          "time shift (tau)": [...],
                          <line_or_lightcurve_name>: [...],
                          ...
                      },
                      ...
                  },
                  ...
              }
    """

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
                    header_lightcurve = list()
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


def process_light_curves(light_curves, one_dim_lightcurves_path):
    """
    Processes light curve text files and returns a dictionary of their contents.

    Each file is assumed to have a header and three columns:
    timestamps, fluxes, and flux errors.

    Args:
        light_curves (list): List of filenames of the light curve text files.
        one_dim_lightcurves_path (Path): Path to the directory containing the light curve files.

    Returns:
        dict: A dictionary where each key is the curve name and the value is another
              dictionary with keys: 'timestamps [MJD]', 'fluxes [ergs/s/cm2/A]', and
              'fluxerrs [ergs/s/cm2/A]'.
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
    Imports 1D light curve data from all campaign directories and organizes them into a nested dictionary.

    For each campaign, separates and loads continuum and emission line light curves
    from the '1DLightCurves' subdirectory.

    Returns:
        dict: A dictionary where each key is a campaign name and the value is a dictionary
              with keys 'lines' and 'continua', each containing their respective light curve data.
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
    Imports FITS files from a specified directory and organizes them into a nested dictionary.
    Calculates the x-axis values from WCS header keywords if available.
    Issues warnings if data or required header parameters are missing.

    Args:
        fits_folder (str or Path, optional): Relative or absolute path to the directory containing FITS files.
                                             If not provided, defaults to the 'fits' subdirectory under the main data folder.

    Returns:
        dict: A dictionary where each key is a FITS filename and the value is a dictionary containing:
              - "data": List of data arrays (one per HDU)
              - "headers": List of header dictionaries (one per HDU)
              - "x_axis": List of x-axis arrays computed from WCS keywords if present
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
    """
    Imports AVG and RMS line profile data from the 'LineProfiles' directory.

    If the `normalized` flag is set, only normalized profiles will be imported;
    otherwise, unnormalized profiles are imported.

    Args:
        normalized (bool or None): Whether to import normalized profiles (True),
                                   unnormalized profiles (False), or all (None).

    Returns:
        dict: A dictionary with keys "avg" and "rms", each mapping to a dictionary
              of line names and their corresponding profile data.
    """

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
    """
    Processes a list of line profile files and returns a structured dictionary.

    Each file is expected to have a header with axis labels and two data columns.
    Optionally extracts pseudo-continuum ranges from the filename.

    Args:
        line_profiles_list (list): List of line profile filenames to process.
        line_profile_path (Path): Path to the directory containing the profile files.

    Returns:
        dict: A dictionary mapping line names to:
              - 'pseudo_conts': a tuple of (blue, red) pseudo-continuum bounds or None
              - 'data_dict': a dictionary with axis labels and corresponding data arrays
    """

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


def load_centroid_data_as_dict():
    """
    Lädt alle .txt-Dateien mit Zeitverzögerungs- und Massenwerten aus einem Ordner
    in ein gemeinsames Dictionary mit den Liniennamen als Keys. Doppelte Namen werden ignoriert.

    Returns:
    --------
    dict
        Dictionary mit dem Namen der Emissionslinie als Key.
        Jeder Value ist ein Dictionary mit den zugehörigen Werten.
    """

    data_path = Path(find_prime_data_folder()) / "centroids"
    data_dict = {}

    for file in data_path.glob("*.txt"):
        with open(file, "r") as f:
            header = f.readline().strip().split()
            data = np.loadtxt(file, skiprows=1, dtype=str)

        for row in data:
            name = row[0]
            if name in data_dict:
                continue  # bereits vorhanden, überspringen

            values = row[1:].astype(float)
            entry = dict(zip(header[1:], values))
            data_dict[str(name)] = entry

    return data_dict