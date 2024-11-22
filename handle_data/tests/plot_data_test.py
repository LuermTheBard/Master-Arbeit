from plot_data.plot_data import get_all_line_names, collect_comparison_data, prepare_output_path, \
    compare_plots_across_continua, plot_individual, plot_combined


def test_get_all_line_names():
    # Beispiel-Daten
    plot_data_1d_correlation = {
        "campaign_1": {"LineA": [], "LineB": []},
        "campaign_2": {"LineB": [], "LineC": []},
    }

    # Erwartetes Ergebnis
    expected_line_names = {"LineA", "LineB", "LineC"}

    # Test
    result = get_all_line_names(plot_data_1d_correlation)
    assert result == expected_line_names


def test_collect_comparison_data():
    # Beispiel-Daten
    plot_data_1d_correlation = {
        "campaign_1": {"LineA": ["data1"], "LineB": ["data2"]},
        "campaign_2": {"LineB": ["data3"], "LineC": ["data4"]},
    }
    line_name = "LineB"

    # Erwartetes Ergebnis
    expected_comparison_data = [
        ("campaign_1", ["data2"]),
        ("campaign_2", ["data3"]),
    ]

    # Test
    result = collect_comparison_data(plot_data_1d_correlation, line_name)
    assert result == expected_comparison_data


def test_prepare_output_path(tmp_path):
    # Beispiel-Daten
    output_dir = tmp_path / "output"
    campaign = "campaign_1"
    line_name = "LineA"
    is_combined = False

    # Erwarteter Pfad
    expected_path = output_dir / "plot_data" / "compare_plots_across_continua" / campaign / "LineA_campaign_1_comparison.png"

    # Test
    result_path = prepare_output_path(output_dir, campaign, line_name, is_combined)

    # Überprüfen, ob der Pfad korrekt erstellt wurde
    assert result_path == expected_path
    assert result_path.parent.exists()  # Der Verzeichnisbaum sollte existieren


def test_compare_plots_across_continua(mocker, tmp_path):
    # Beispiel-Daten
    plot_data_1d_correlation = {
        "campaign_1": {"LineA": [("label1", [0, 1], [0.1, 0.2])]},
        "campaign_2": {"LineA": [("label2", [0, 1], [0.2, 0.3])]},
    }

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Mock für plot_individual und plot_combined
    plot_individual_mock = mocker.patch("plot_data.plot_data.plot_individual")
    plot_combined_mock = mocker.patch("plot_data.plot_data.plot_combined")

    # Test
    compare_plots_across_continua(plot_data_1d_correlation, line_name="LineA", output_dir=output_dir, save_only=True)

    # Assertions
    plot_individual_mock.assert_called()
    plot_combined_mock.assert_called_once()


def test_plot_individual(tmp_path):
    data_list = [("label1", [0, 1], [0.1, 0.2])]
    current_line_name = "LineA"
    campaign = "campaign_1"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    plot_individual(data_list, current_line_name, campaign, output_dir, save_only=True)

    file_path = (output_dir / "plot_data" / "compare_plots_across_continua" / "campaign_1" /
                 "LineA_campaign_1_comparison.png")

    assert file_path.is_file()


def test_plot_combined(tmp_path):
    comparison_data = [
        ("campaign_1", [("label1", [0, 1], [0.1, 0.2])]),
        ("campaign_2", [("label2", [0, 1], [0.2, 0.3])]),
    ]
    current_line_name = "LineA"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    plot_combined(comparison_data, current_line_name, output_dir, save_only=True)

    file_path = (output_dir / "plot_data" / "compare_plots_across_continua" / "combined" /
                 "LineA_combined_comparison.png")

    assert file_path.is_file()
