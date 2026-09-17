"""scGPT integration layer: the parts that are validatable without weights."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nkstate.models.scgpt import (
    DEFAULT_MAX_SEQ_LEN, DEFAULT_N_BINS, ScgptUnavailable, TokenisationError,
    bin_expression, build_finetune_command, checkpoint_report, load_vocabulary,
    map_to_vocabulary, tokenise,
)


@pytest.fixture
def vocabulary():
    return {f"G{i}": i + 3 for i in range(100)} | {"GNLY": 200, "PRF1": 201}


def test_mapping_keeps_known_genes_and_drops_the_rest(vocabulary):
    mapping = map_to_vocabulary(["G0", "G1", "ABSENT"], vocabulary)
    assert mapping.kept_genes == ["G0", "G1"]
    assert mapping.dropped_genes == ["ABSENT"]
    assert mapping.token_ids == [3, 4]


def test_mapping_reports_coverage(vocabulary):
    mapping = map_to_vocabulary(["G0", "G1", "X", "Y"], vocabulary)
    assert mapping.coverage == pytest.approx(0.5)


def test_mapping_separates_dropped_marker_genes(vocabulary):
    """Losing GNLY is not equivalent to losing a background gene."""
    mapping = map_to_vocabulary(
        ["G0", "GZMB", "GNLY"], vocabulary, marker_genes=["GZMB", "GNLY"]
    )
    assert mapping.dropped_marker_genes == ["GZMB"]
    assert "GNLY" in mapping.kept_genes


def test_mapping_coverage_of_an_empty_gene_list(vocabulary):
    assert map_to_vocabulary([], vocabulary).coverage == 0.0


def test_mapping_serialises_with_examples(vocabulary):
    record = map_to_vocabulary(["G0", "X"], vocabulary).to_dict()
    assert record["n_kept"] == 1
    assert record["dropped_examples"] == ["X"]


def test_binning_puts_zeros_in_bin_zero():
    expression = np.array([[0.0, 1.0, 2.0, 3.0]])
    binned = bin_expression(expression, n_bins=4)
    assert binned[0, 0] == 0
    assert np.all(binned[0, 1:] >= 1)


def test_binning_is_per_cell_not_global():
    """Two cells with the same ranking must bin identically despite scale."""
    low = np.array([[1.0, 2.0, 3.0, 4.0]])
    high = low * 1000
    assert np.array_equal(
        bin_expression(low, n_bins=5), bin_expression(high, n_bins=5)
    )


def test_binning_respects_the_bin_count():
    rng = np.random.default_rng(0)
    expression = rng.random((5, 200))
    binned = bin_expression(expression, n_bins=10)
    assert int(binned.max()) <= 9
    assert int(binned.min()) >= 0


def test_binning_preserves_monotonic_order_within_a_cell():
    expression = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    binned = bin_expression(expression, n_bins=7)[0]
    assert list(binned) == sorted(binned)


def test_an_all_zero_cell_bins_to_zero():
    assert int(bin_expression(np.zeros((1, 20)), n_bins=10).max()) == 0


def test_binning_rejects_negative_values():
    """Scaled matrices have negatives and cannot be binned."""
    with pytest.raises(TokenisationError, match="negative"):
        bin_expression(np.array([[-1.0, 1.0]]), n_bins=5)


def test_binning_rejects_a_one_dimensional_input():
    with pytest.raises(TokenisationError, match="cells, genes"):
        bin_expression(np.array([1.0, 2.0]), n_bins=5)


def test_binning_rejects_too_few_bins():
    with pytest.raises(TokenisationError, match="at least 2"):
        bin_expression(np.array([[1.0]]), n_bins=1)


def test_tokenisation_truncates_to_the_sequence_length(vocabulary):
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(100)]
    tokenised = tokenise(
        rng.random((10, 100)), genes, vocabulary, max_seq_len=20
    )
    assert tokenised.gene_ids.shape == (10, 20)
    assert tokenised.expression_bins.shape == (10, 20)


def test_tokenisation_pins_marker_genes_regardless_of_rank(vocabulary):
    """A marker that is lowly expressed must survive truncation."""
    genes = [f"G{i}" for i in range(100)] + ["GNLY"]
    expression = np.ones((5, 101))
    expression[:, -1] = 0.001  # GNLY is the lowest-expressed gene
    tokenised = tokenise(
        expression, genes, vocabulary, max_seq_len=10, pin_genes=["GNLY"]
    )
    assert "GNLY" in tokenised.gene_names
    assert tokenised.pinned_genes == ["GNLY"]


def test_tokenisation_without_pinning_drops_the_lowly_expressed_marker(vocabulary):
    genes = [f"G{i}" for i in range(100)] + ["GNLY"]
    expression = np.ones((5, 101))
    expression[:, -1] = 0.001
    tokenised = tokenise(expression, genes, vocabulary, max_seq_len=10)
    assert "GNLY" not in tokenised.gene_names


def test_tokenisation_rejects_a_gene_name_mismatch(vocabulary):
    with pytest.raises(TokenisationError, match="names"):
        tokenise(np.ones((3, 5)), ["G0", "G1"], vocabulary)


def test_tokenisation_rejects_a_disjoint_vocabulary():
    with pytest.raises(TokenisationError, match="vocabulary"):
        tokenise(np.ones((3, 2)), ["X", "Y"], {"A": 1, "B": 2})


def test_tokenised_record_reports_the_bin_range(vocabulary):
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(100)]
    record = tokenise(
        rng.random((5, 100)), genes, vocabulary, max_seq_len=30
    ).to_dict()
    assert record["seq_len"] == 30
    assert record["bin_range"][0] >= 0


def test_vocabulary_loads_from_json(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"GNLY": 1, "PRF1": 2}))
    assert load_vocabulary(path) == {"GNLY": 1, "PRF1": 2}


def test_missing_vocabulary_explains_where_to_get_one(tmp_path):
    with pytest.raises(ScgptUnavailable, match="scGPT"):
        load_vocabulary(tmp_path / "absent.json")


def test_empty_vocabulary_is_rejected(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text("{}")
    with pytest.raises(TokenisationError, match="non-empty"):
        load_vocabulary(path)


def test_absent_checkpoint_is_reported_not_assumed():
    report = checkpoint_report(None)
    assert report["available"] is False
    assert "args.json" in report["required_files"]
    assert "github.com/bowang-lab/scGPT" in report["source"]


def test_incomplete_checkpoint_lists_missing_files(tmp_path):
    (tmp_path / "vocab.json").write_text("{}")
    report = checkpoint_report(tmp_path)
    assert report["available"] is False
    assert report["present_files"] == ["vocab.json"]
    assert set(report["missing_files"]) == {"args.json", "best_model.pt"}


def test_complete_checkpoint_is_reported_available(tmp_path):
    for name in ("args.json", "vocab.json", "best_model.pt"):
        (tmp_path / name).write_text("{}")
    assert checkpoint_report(tmp_path)["available"] is True


def test_finetuning_without_a_checkpoint_is_refused():
    """No weights means no transformer result, and no pretending otherwise."""
    with pytest.raises(ScgptUnavailable, match="not redistributed"):
        build_finetune_command(None, "data.h5ad", "out", n_classes=4)


def test_dry_run_builds_a_command_without_a_checkpoint():
    command = build_finetune_command(
        None, "data.h5ad", "out", n_classes=4, dry_run=True
    )
    assert "--n-cls=4" in command.command_string
    assert "cell_annotation" in command.command_string
    assert command.settings["dry_run"] is True


def test_command_records_the_settings_that_change_results():
    command = build_finetune_command(
        None, "d.h5ad", "out", n_classes=4, epochs=3, batch_size=8,
        learning_rate=1e-3, max_seq_len=512, n_bins=21, freeze_encoder=True,
        dry_run=True,
    )
    assert command.settings["epochs"] == 3
    assert command.settings["freeze_encoder"] is True
    assert "--freeze-encoder=True" in command.command_string
    assert "--max-seq-len=512" in command.command_string


def test_freeze_encoder_is_omitted_when_false():
    command = build_finetune_command(
        None, "d.h5ad", "out", n_classes=4, freeze_encoder=False, dry_run=True
    )
    assert "--freeze-encoder" not in command.command_string


def test_command_rejects_fewer_than_two_classes():
    with pytest.raises(TokenisationError, match="at least 2"):
        build_finetune_command(None, "d.h5ad", "out", n_classes=1, dry_run=True)


def test_defaults_match_the_published_task():
    assert DEFAULT_N_BINS == 51
    assert DEFAULT_MAX_SEQ_LEN == 1200
