"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from nkstate.config import ConfigError, RunConfig, config_from_dict, load_config


def test_defaults_load():
    config = config_from_dict({})
    assert isinstance(config, RunConfig)
    assert config.split.kind == "donor"


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level section"):
        config_from_dict({"modelz": {}})


def test_unknown_key_is_rejected():
    """A silently ignored typo produces a run that contradicts its own config."""
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"data": {"n_top_gene": 100}})


def test_rejection_message_lists_the_valid_keys():
    with pytest.raises(ConfigError) as info:
        config_from_dict({"data": {"n_top_gene": 100}})
    assert "n_top_genes" in str(info.value)


def test_non_mapping_section_is_rejected():
    with pytest.raises(ConfigError, match="must be a mapping"):
        config_from_dict({"data": ["not", "a", "mapping"]})


def test_non_mapping_top_level_is_rejected():
    with pytest.raises(ConfigError, match="mapping at the top level"):
        config_from_dict(["nope"])


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"source": "parquet"}},
        {"data": {"source": "h5ad"}},
        {"data": {"max_mito_fraction": 20.0}},
        {"data": {"n_top_genes": 10}},
        {"data": {"target_sum": 0}},
        {"data": {"min_genes": -1}},
        {"split": {"kind": "random"}},
        {"split": {"test_fraction": 1.5}},
        {"split": {"test_fraction": 0.8, "val_fraction": 0.3}},
        {"split": {"n_folds": 1}},
        {"labels": {"source": "guess"}},
        {"labels": {"states": ["resting"]}},
        {"labels": {"margin": -0.5}},
        {"models": {"baselines": ["deep_magic"]}},
        {"models": {"baselines": []}},
        {"scgpt": {"max_seq_len": 8}},
        {"scgpt": {"n_bins": 1}},
        {"scgpt": {"epochs": 0}},
        {"scgpt": {"learning_rate": 5.0}},
        {"interpret": {"top_n_genes": 2}},
        {"synthetic": {"n_donors": 0}},
        {"synthetic": {"cells_per_donor": 2}},
        {"synthetic": {"state_effect": 1.5}},
    ],
)
def test_invalid_values_are_rejected(payload):
    with pytest.raises(ConfigError):
        config_from_dict(payload)


def test_mito_fraction_error_explains_the_unit():
    """The most likely mistake is passing a percentage, so say so."""
    with pytest.raises(ConfigError, match="percentage"):
        config_from_dict({"data": {"max_mito_fraction": 15.0}})


def test_h5ad_source_requires_a_path():
    with pytest.raises(ConfigError, match="requires a path"):
        config_from_dict({"data": {"source": "h5ad"}})


def test_lineage_panels_are_valid_label_states():
    config = config_from_dict(
        {"labels": {"source": "marker_score", "states": ["NK_cell", "T_cell"]}}
    )
    assert config.labels.states == ["NK_cell", "T_cell"]


def test_annotation_source_does_not_require_panels():
    """With curated labels there is no marker panel requirement."""
    config = config_from_dict(
        {"labels": {"source": "annotation", "states": ["anything", "goes"]}}
    )
    assert config.labels.source == "annotation"


def test_paths_resolve_relative_to_the_config_file(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "run.yaml").write_text(
        "output_dir: ../results\ndata:\n  source: h5ad\n  path: ../data/cells.h5ad\n"
    )
    config = load_config(directory / "run.yaml")
    assert config.output_dir == str((tmp_path / "results").resolve())
    assert config.data.path == str((tmp_path / "data" / "cells.h5ad").resolve())


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_empty_yaml_loads_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path).split.kind == "donor"


def test_shipped_configs_are_valid():
    """The configs in the repository must load, or the quickstart is broken."""
    root = Path(__file__).resolve().parent.parent / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files, "expected shipped configs"
    for path in files:
        load_config(path)


def test_synthetic_config_builds_a_spec():
    config = config_from_dict({"synthetic": {"n_donors": 4, "state_effect": 0.5}})
    spec = config.synthetic.to_spec()
    assert spec.n_donors == 4
    assert spec.state_effect == 0.5


def test_to_dict_round_trips():
    config = config_from_dict({"data": {"n_top_genes": 321}})
    assert config_from_dict(config.to_dict()).data.n_top_genes == 321
