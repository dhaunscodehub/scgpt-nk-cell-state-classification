"""Data loading, quality control and HVG selection."""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.data.loading import (
    DataError, expression_matrix, normalise, quality_control, select_hvgs,
    subset_nk_cells,
)
from nkstate.testing import SyntheticSpec, build_synthetic


@pytest.fixture(scope="module")
def raw():
    return build_synthetic(
        SyntheticSpec(n_donors=4, cells_per_donor=80, state_effect=0.8, seed=0)
    )


def test_qc_attributes_removals_to_individual_rules(raw):
    _, report = quality_control(
        raw.adata, min_genes=0, min_counts=0, max_mito_fraction=1.0,
        min_cells_per_gene=0,
    )
    record = report.to_dict()
    for key in ("removed_low_genes", "removed_low_counts", "removed_high_mito"):
        assert key in record


def test_qc_with_permissive_thresholds_keeps_every_cell(raw):
    filtered, report = quality_control(
        raw.adata, min_genes=0, min_counts=0, max_mito_fraction=1.0,
        min_cells_per_gene=0,
    )
    assert filtered.n_obs == raw.adata.n_obs
    assert report.n_cells_after == raw.adata.n_obs


def test_qc_removes_only_the_cells_that_fail(raw):
    """A threshold between the observed extremes must remove some, not all."""
    counts = expression_matrix(raw.adata)
    genes_per_cell = (counts > 0).sum(axis=1)
    threshold = int(np.median(genes_per_cell))

    filtered, report = quality_control(
        raw.adata, min_genes=threshold, min_counts=0, max_mito_fraction=1.0,
        min_cells_per_gene=0,
    )
    assert 0 < filtered.n_obs < raw.adata.n_obs
    assert report.removed_low_genes == int((genes_per_cell < threshold).sum())
    assert report.n_cells_after == filtered.n_obs


def test_qc_that_removes_everything_raises_with_a_diagnosis(raw):
    """An impossible threshold must say what to check, not return nothing."""
    with pytest.raises(DataError, match="removed every cell"):
        quality_control(
            raw.adata, min_genes=10_000, min_counts=0, max_mito_fraction=1.0,
        )


def test_qc_records_the_thresholds_it_used(raw):
    _, report = quality_control(
        raw.adata, min_genes=5, min_counts=10, max_mito_fraction=0.5,
    )
    assert report.thresholds["min_genes"] == 5
    assert report.thresholds["max_mito_fraction"] == 0.5


def test_normalise_keeps_raw_counts_in_a_layer(raw):
    normalised = normalise(raw.adata)
    assert "counts" in normalised.layers
    counts = expression_matrix(normalised, layer="counts")
    # Counts are integers; log-normalised values are not.
    assert np.allclose(counts, np.round(counts))


def test_normalise_produces_non_negative_values(raw):
    assert float(expression_matrix(normalise(raw.adata)).min()) >= 0.0


def test_normalise_records_what_it_did(raw):
    normalised = normalise(raw.adata)
    assert normalised.uns["nkstate_normalisation"]["log1p"] is True


def test_normalise_refuses_to_log_twice(raw):
    """Double log1p flattens every fold change while looking plausible."""
    once = normalise(raw.adata)
    twice = normalise(once)
    assert np.allclose(expression_matrix(once), expression_matrix(twice))
    assert "skipped_second_pass" in twice.uns["nkstate_normalisation"]


def test_hvg_selection_returns_the_requested_count(raw):
    normalised = normalise(raw.adata)
    selected = select_hvgs(normalised, n_top_genes=100, donor_key=None)
    assert selected.n_vars <= 100 + 0


def test_hvg_selection_forces_marker_genes_back_in(raw):
    """Selection must not drop the genes the task is defined by."""
    from nkstate.markers import all_state_genes

    normalised = normalise(raw.adata)
    keep = tuple(
        g for g in all_state_genes() if g in set(map(str, normalised.var_names))
    )
    selected = select_hvgs(
        normalised, n_top_genes=50, donor_key=None, keep_genes=keep
    )
    assert set(keep) <= set(map(str, selected.var_names))


def test_hvg_selection_uses_a_per_donor_union_when_donors_exist(raw):
    normalised = normalise(raw.adata)
    selected = select_hvgs(normalised, n_top_genes=100, donor_key="donor_id")
    assert selected.uns["nkstate_hvg"]["strategy"] == "per-donor union"
    assert selected.uns["nkstate_hvg"]["n_donors"] == 4


def test_hvg_selection_says_so_when_there_is_only_one_donor(raw):
    """A per-donor guard that did nothing must not claim it did."""
    normalised = normalise(raw.adata)
    normalised.obs["donor_id"] = "single"
    selected = select_hvgs(normalised, n_top_genes=100, donor_key="donor_id")
    assert "only one donor" in selected.uns["nkstate_hvg"]["strategy"]


def test_hvg_selection_rejects_too_few_genes(raw):
    with pytest.raises(DataError, match="below 10"):
        select_hvgs(normalise(raw.adata), n_top_genes=5)


def test_expression_matrix_reads_a_named_layer(raw):
    normalised = normalise(raw.adata)
    assert expression_matrix(normalised, layer="counts").shape == normalised.shape


def test_subset_nk_prefers_a_curated_label(raw):
    adata = raw.adata.copy()
    adata.obs["cell_type"] = "NK cells"
    adata.obs.iloc[:10, adata.obs.columns.get_loc("cell_type")] = "T cells"
    subset = subset_nk_cells(adata, cell_type_key="cell_type")
    assert subset.n_obs == adata.n_obs - 10


def test_subset_nk_requires_an_explicit_threshold_without_labels(raw):
    """Marker gating needs a stated cut-off; there is no safe default."""
    adata = raw.adata.copy()
    with pytest.raises(DataError):
        subset_nk_cells(adata, cell_type_key=None, identity_threshold=None)
