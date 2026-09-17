"""Command-line interface: exit codes and stream discipline."""

from __future__ import annotations

import json

import pytest

from nkstate.cli import (
    EXIT_BAD_INPUT, EXIT_EXTERNAL_TOOL, EXIT_OK, build_parser, main,
)


def _write_config(path, extra: str = "") -> str:
    path.write_text(
        "name: cli_run\n"
        "output_dir: out\n"
        "data:\n"
        "  source: synthetic\n"
        "  min_genes: 0\n"
        "  min_counts: 0\n"
        "  max_mito_fraction: 1.0\n"
        "  min_cells_per_gene: 0\n"
        "  n_top_genes: 200\n"
        "  label_key: true_state\n"
        "split:\n"
        "  kind: donor\n"
        "  test_fraction: 0.25\n"
        "  val_fraction: 0.2\n"
        "labels:\n"
        "  source: annotation\n"
        "models:\n"
        "  baselines: [logistic]\n"
        "synthetic:\n"
        "  n_donors: 8\n"
        "  cells_per_donor: 60\n"
        "  state_effect: 0.8\n"
        "  n_background_genes: 300\n"
        "interpret:\n"
        "  top_n_genes: 20\n" + extra
    )
    return str(path)


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_emits_json_on_stdout_only(tmp_path, capsys):
    """Status goes to stderr so `nkstate run | jq` works."""
    config = _write_config(tmp_path / "run.yaml")
    code = main(["run", "-c", config, "-o", str(tmp_path / "out")])
    captured = capsys.readouterr()

    assert code == EXIT_OK
    payload = json.loads(captured.out)
    assert payload["name"] == "cli_run"
    assert "[nkstate]" in captured.err
    assert "[nkstate]" not in captured.out


def test_run_reports_metrics_on_stderr(tmp_path, capsys):
    config = _write_config(tmp_path / "run.yaml")
    main(["run", "-c", config, "-o", str(tmp_path / "out")])
    assert "macro-F1" in capsys.readouterr().err


def test_run_rejects_a_bad_config(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  kind: nonsense\n")
    assert main(["run", "-c", str(path)]) == EXIT_BAD_INPUT
    assert "config error" in capsys.readouterr().err


def test_run_reports_a_missing_config(tmp_path, capsys):
    assert main(["run", "-c", str(tmp_path / "absent.yaml")]) == EXIT_BAD_INPUT
    assert "not found" in capsys.readouterr().err


def test_markers_command_lists_panels(capsys):
    assert main(["markers"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "cytotoxic" in payload["panels"]
    assert payload["panels"]["cytotoxic"]["reference"]


def test_markers_command_reports_dataset_coverage(tmp_path, capsys):
    config = _write_config(tmp_path / "run.yaml")
    assert main(["markers", "-c", config]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "coverage" in payload
    assert payload["dataset"]["n_cells"] > 0


def test_scgpt_command_reports_an_absent_checkpoint(tmp_path, capsys):
    config = _write_config(tmp_path / "run.yaml")
    assert main(["scgpt", "-c", config, "--dry-run"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"]["available"] is False
    assert payload["executed"] is False


def test_scgpt_command_without_dry_run_exits_external_tool(tmp_path, capsys):
    """No checkpoint is an external-tool condition, not a crash."""
    config = _write_config(tmp_path / "run.yaml")
    assert main(["scgpt", "-c", config]) == EXIT_OK
    # With no checkpoint the command falls back to dry-run and still reports.
    assert json.loads(capsys.readouterr().out)["executed"] is False


def test_tokenise_without_a_vocabulary_exits_external_tool(tmp_path, capsys):
    config = _write_config(tmp_path / "run.yaml")
    assert main(["tokenise", "-c", config]) == EXIT_EXTERNAL_TOOL
    assert "vocabulary_path" in capsys.readouterr().err


def test_tokenise_with_a_vocabulary_emits_a_record(tmp_path, capsys):
    vocabulary = tmp_path / "vocab.json"
    from nkstate.markers import all_state_genes

    vocabulary.write_text(
        json.dumps({g: i + 3 for i, g in enumerate(all_state_genes())})
    )
    config = _write_config(
        tmp_path / "run.yaml",
        extra=f"scgpt:\n  vocabulary_path: {vocabulary}\n  max_seq_len: 64\n",
    )
    assert main(["tokenise", "-c", config]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["seq_len"] <= 64
    assert payload["n_cells"] > 0


def test_validate_command_runs_the_controls(capsys):
    assert main(["validate"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_passed"] is True
    assert payload["n_checks"] >= 7


def test_validate_command_logs_each_control(capsys):
    main(["validate"])
    err = capsys.readouterr().err
    assert err.count("PASS") >= 7
