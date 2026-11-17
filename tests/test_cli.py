"""Tests of the command-line interface."""

import pytest

from viral.cli import main


def test_theory_command_prints_known_values(capsys):
    assert main(["theory"]) == 0
    output = capsys.readouterr().out
    assert "8.000" in output
    assert "0.1458" in output


def test_simulate_command_is_reproducible(capsys):
    main(["simulate", "--runs", "300", "--seed", "7"])
    first = capsys.readouterr().out
    main(["simulate", "--runs", "300", "--seed", "7"])
    assert capsys.readouterr().out == first


def test_figures_command_writes_two_images(tmp_path):
    pytest.importorskip("matplotlib")
    assert main(["figures", "--results", "results/viral.json", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dose_response.png", "viral_load.png"]


def test_neutralization_outside_range_is_rejected():
    with pytest.raises(SystemExit):
        main(["theory", "--neutralization", "1.5"])
