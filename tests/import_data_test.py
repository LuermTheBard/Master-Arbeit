import pytest


from import_data.import_data import load_dopplershift_data_from_toml


def test_load_dopplershift_data_from_toml_custom_path(tmp_path):
    """
    Testet die Funktion mit einem benutzerdefinierten Pfad.
    """
    # Erstelle ein benutzerdefiniertes Verzeichnis
    custom_path = tmp_path / "custom_path"
    custom_path.mkdir(parents=True)

    # Erstelle eine Beispiel-TOML-Datei
    toml_file = custom_path / "test.toml"
    toml_content = """
    example_key = "example_value"

    [nested_key]
    sub_key = 42
    """
    toml_file.write_text(toml_content)

    # Aufruf der Funktion
    result = load_dopplershift_data_from_toml("test.toml", custom_path)

    # Assertions
    assert result["example_key"] == "example_value"
    assert result["nested_key"]["sub_key"] == 42


def test_load_dopplershift_data_from_toml_file_not_found(tmp_path):
    """
    Testet, ob die Funktion eine Ausnahme auslöst, wenn die Datei nicht existiert.
    """
    # Kein Verzeichnis oder Datei wird erstellt

    with pytest.raises(FileNotFoundError):
        load_dopplershift_data_from_toml("non_existent.toml", tmp_path)