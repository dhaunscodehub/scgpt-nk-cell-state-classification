"""Differential expression and marker enrichment."""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.interpret import (
    InterpretationError, benjamini_hochberg, differential_expression,
    interpret_states, marker_enrichment, model_gene_enrichment,
)


def test_bh_leaves_a_single_p_value_unchanged():
    assert benjamini_hochberg(np.array([0.03]))[0] == pytest.approx(0.03)


def test_bh_is_monotone_in_the_input_order():
    adjusted = benjamini_hochberg(np.array([0.001, 0.01, 0.2, 0.5, 0.9]))
    assert list(adjusted) == sorted(adjusted)


def test_bh_never_exceeds_one():
    adjusted = benjamini_hochberg(np.array([0.9, 0.95, 0.99]))
    assert float(adjusted.max()) <= 1.0


def test_bh_is_never_smaller_than_the_raw_p_value():
    raw = np.array([0.01, 0.02, 0.03, 0.5])
    assert np.all(benjamini_hochberg(raw) >= raw - 1e-12)


def test_bh_matches_the_closed_form_on_a_known_case():
    """Smallest of n p-values is multiplied by n/1 when it stays the minimum."""
    raw = np.array([0.01, 0.4, 0.6, 0.8])
    assert benjamini_hochberg(raw)[0] == pytest.approx(0.04)


def test_bh_preserves_input_ordering_of_positions():
    raw = np.array([0.5, 0.001])
    adjusted = benjamini_hochberg(raw)
    assert adjusted[1] < adjusted[0]


def test_bh_handles_an_empty_input():
    assert benjamini_hochberg(np.array([])).size == 0


def test_bh_rejects_out_of_range_values():
    with pytest.raises(InterpretationError, match=r"\[0, 1\]"):
        benjamini_hochberg(np.array([1.5]))


def test_de_ranks_the_planted_genes_first(strong_signal):
    from nkstate.markers import STATE_PANELS

    genes = differential_expression(
        strong_signal.features, strong_signal.gene_names,
        strong_signal.true_labels, "cytotoxic", top_n=20,
    )
    panel = set(STATE_PANELS["cytotoxic"].genes)
    top_ten = {g.gene for g in genes[:10]}
    assert len(top_ten & panel) >= 5, f"expected cytotoxic markers, got {top_ten}"


def test_de_fold_change_is_positive_for_an_upregulated_gene(strong_signal):
    genes = differential_expression(
        strong_signal.features, strong_signal.gene_names,
        strong_signal.true_labels, "cytotoxic", top_n=5,
    )
    assert genes[0].log2_fold_change > 0


def test_de_reports_the_expressing_fraction(strong_signal):
    genes = differential_expression(
        strong_signal.features, strong_signal.gene_names,
        strong_signal.true_labels, "cytotoxic", top_n=5,
    )
    assert all(0.0 <= g.fraction_expressing_in <= 1.0 for g in genes)


def test_de_flags_technical_genes():
    from nkstate.interpret import DEGene

    gene = DEGene("MT-CO1", 1.0, 1.0, 0.01, 0.02, 1.0, 0.5, 0.9)
    assert gene.is_technical
    assert not DEGene("GNLY", 1.0, 1.0, 0.01, 0.02, 1.0, 0.5, 0.9).is_technical


def test_de_rejects_a_group_with_too_few_cells(strong_signal):
    labels = strong_signal.true_labels.copy()
    labels[:] = "bulk"
    labels[:3] = "tiny"
    with fewer_cells_error():
        differential_expression(
            strong_signal.features, strong_signal.gene_names, labels, "tiny",
        )


def fewer_cells_error():
    return pytest.raises(InterpretationError, match="fewer than the minimum")


def test_de_rejects_a_group_with_no_comparison(strong_signal):
    labels = np.array(["all"] * strong_signal.features.shape[0])
    with pytest.raises(InterpretationError, match="comparison group"):
        differential_expression(
            strong_signal.features, strong_signal.gene_names, labels, "all",
        )


def test_de_rejects_mismatched_shapes(strong_signal):
    with pytest.raises(InterpretationError, match="labels"):
        differential_expression(
            strong_signal.features, strong_signal.gene_names,
            strong_signal.true_labels[:5], "cytotoxic",
        )


def test_enrichment_detects_a_planted_panel(strong_signal):
    from nkstate.markers import STATE_PANELS

    ranked = list(STATE_PANELS["cytotoxic"].genes) + strong_signal.gene_names[:40]
    result = marker_enrichment(
        ranked, "cytotoxic", strong_signal.gene_names, top_n=20
    )
    assert result.is_enriched
    assert result.fold_enrichment > 1.0
    assert result.p_value < 0.05


def test_enrichment_is_absent_for_an_unrelated_ranking(strong_signal):
    """Background genes must not appear enriched for a marker panel."""
    from nkstate.markers import all_state_genes

    panel_genes = set(all_state_genes())
    background = [g for g in strong_signal.gene_names if g not in panel_genes]
    result = marker_enrichment(
        background, "cytotoxic", strong_signal.gene_names, top_n=30
    )
    assert result.n_overlap == 0
    assert not result.is_enriched


def test_enrichment_resolves_lineage_labels(strong_signal):
    """A curated cell-type label must map to its lineage panel."""
    from nkstate.markers import ALL_PANELS

    universe = list(ALL_PANELS["NK_cell"].genes) + strong_signal.gene_names[:100]
    result = marker_enrichment(
        list(ALL_PANELS["NK_cell"].genes), "NK cells", universe, top_n=10
    )
    assert result.n_overlap > 0


def test_enrichment_rejects_an_unknown_label(strong_signal):
    with pytest.raises(InterpretationError, match="no marker panel"):
        marker_enrichment(
            strong_signal.gene_names, "erythrocyte", strong_signal.gene_names
        )


def test_enrichment_rejects_an_empty_universe():
    with pytest.raises(InterpretationError, match="universe is empty"):
        marker_enrichment(["GNLY"], "cytotoxic", [])


def test_enrichment_rejects_a_universe_without_panel_genes():
    with pytest.raises(InterpretationError, match="in the measured universe"):
        marker_enrichment(["FAKE1"], "cytotoxic", ["FAKE1", "FAKE2"])


def test_enrichment_uses_the_measured_universe_not_the_transcriptome():
    """A larger universe must not make the same overlap look more significant."""
    from nkstate.markers import STATE_PANELS

    panel = list(STATE_PANELS["cytotoxic"].genes)
    small = panel + [f"BG{i}" for i in range(50)]
    large = panel + [f"BG{i}" for i in range(5000)]
    p_small = marker_enrichment(panel, "cytotoxic", small, top_n=10).p_value
    p_large = marker_enrichment(panel, "cytotoxic", large, top_n=10).p_value
    assert p_large < p_small


def test_enrichment_reports_technical_genes_in_the_top_list():
    from nkstate.markers import STATE_PANELS

    panel = list(STATE_PANELS["cytotoxic"].genes)
    universe = panel + ["MT-CO1", "RPS6"]
    result = marker_enrichment(["MT-CO1", "RPS6", *panel], "cytotoxic", universe)
    assert set(result.technical_genes_in_top) == {"MT-CO1", "RPS6"}


def test_interpret_states_covers_every_state(strong_signal):
    report = interpret_states(
        strong_signal.features, strong_signal.gene_names,
        strong_signal.true_labels, top_n=25,
    )
    assert report["all_states_enriched"]
    assert not report["states_not_enriched"]
    assert len(report["states_tested"]) == 4


def test_interpret_states_lists_classes_without_a_panel(strong_signal):
    # object dtype, not the fixture's fixed-width unicode array: assigning a
    # longer label into a '<U9' array truncates it silently.
    labels = strong_signal.true_labels.astype(object)
    labels[:100] = "erythrocyte"
    report = interpret_states(
        strong_signal.features, strong_signal.gene_names, labels, top_n=25,
    )
    assert "erythrocyte" in report["classes_without_marker_panel"]


def test_interpret_states_rejects_labels_with_no_panels(strong_signal):
    labels = np.array(["x", "y"] * (strong_signal.features.shape[0] // 2))
    with pytest.raises(InterpretationError, match="no label"):
        interpret_states(
            strong_signal.features, strong_signal.gene_names, labels,
        )


def test_model_gene_enrichment_tests_per_class(strong_signal):
    from nkstate.markers import STATE_PANELS

    importance = {
        state: [(g, 1.0) for g in STATE_PANELS[state].genes]
        for state in ("resting", "cytotoxic")
    }
    report = model_gene_enrichment(importance, strong_signal.gene_names, top_n=20)
    assert report["all_classes_enriched"]
    assert set(report["per_class_enrichment"]) == {"resting", "cytotoxic"}


def test_model_gene_enrichment_marks_unsigned_importances_untestable(strong_signal):
    """Tree importances are shared across classes and cannot be attributed."""
    report = model_gene_enrichment(
        {"all_classes": [(g, 1.0) for g in strong_signal.gene_names[:10]]},
        strong_signal.gene_names,
    )
    assert report["untestable_importance_keys"] == ["all_classes"]
    assert report["note"] is not None
