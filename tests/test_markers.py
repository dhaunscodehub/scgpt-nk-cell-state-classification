"""Marker panel definitions and resolution."""

from __future__ import annotations

import pytest

from nkstate import markers


def test_all_nk_states_have_panels():
    assert set(markers.STATE_PANELS) == set(markers.NK_STATES)


def test_every_panel_has_genes_and_a_reference():
    for name, panel in markers.ALL_PANELS.items():
        assert panel.genes, f"{name} has no genes"
        assert panel.reference, f"{name} has no literature reference"


def test_panels_reject_duplicate_genes():
    with pytest.raises(ValueError, match="repeats gene"):
        markers.MarkerPanel("bad", ("CD69", "CD69"), "d", "r")


def test_panels_reject_emptiness():
    with pytest.raises(ValueError, match="no genes"):
        markers.MarkerPanel("bad", (), "d", "r")


def test_panel_direction_is_validated():
    with pytest.raises(ValueError, match="direction"):
        markers.MarkerPanel("bad", ("CD69",), "d", "r", direction="sideways")


def test_state_panels_are_not_identical():
    """Panels must be distinguishable, or the states cannot be separated."""
    genes = {name: set(p.genes) for name, p in markers.STATE_PANELS.items()}
    for left in genes:
        for right in genes:
            if left < right:
                assert genes[left] != genes[right], f"{left} and {right} are identical"


def test_panel_overlap_is_symmetric_and_reported():
    overlap = markers.panel_overlap()
    assert overlap, "expected the overlap report to be non-empty"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("CD4 T cells", "T_cell"),
        ("CD8 T cells", "T_cell"),
        ("NK cells", "NK_cell"),
        ("CD14+ Monocytes", "monocyte"),
        ("FCGR3A+ Monocytes", "monocyte"),
        ("B cells", "B_cell"),
        ("Dendritic cells", "dendritic"),
        ("Megakaryocytes", "megakaryocyte"),
        ("cytotoxic", "cytotoxic"),
        ("NK_cell", "NK_cell"),
    ],
)
def test_resolve_panel_handles_annotation_vocabularies(label, expected):
    assert markers.resolve_panel(label) == expected


def test_resolve_panel_returns_none_for_unknown_label():
    """An unmapped cell type is untested, not an error."""
    assert markers.resolve_panel("erythrocyte") is None


def test_resolve_panel_is_case_insensitive():
    assert markers.resolve_panel("nk cells") == markers.resolve_panel("NK CELLS")


def test_all_state_genes_is_the_union_of_state_panels():
    union = set()
    for panel in markers.STATE_PANELS.values():
        union |= set(panel.genes)
    assert set(markers.all_state_genes()) == union


def test_panel_coverage_reports_present_and_missing_genes():
    report = markers.panel_coverage({"GNLY", "NKG7"})
    cytotoxic = report["cytotoxic"]
    assert 0.0 < cytotoxic["coverage"] < 1.0
    assert set(cytotoxic["present"]) == {"GNLY", "NKG7"}
    # The missing genes are listed, not just counted: knowing *which* marker
    # is absent is what tells you whether a panel is still usable.
    assert "PRF1" in cytotoxic["missing"]
    assert cytotoxic["n_present"] + cytotoxic["n_missing"] == cytotoxic["n_panel_genes"]


def test_panel_coverage_is_zero_on_an_empty_dataset():
    report = markers.panel_coverage(set())
    assert {entry["coverage"] for entry in report.values()} == {0.0}
    assert all(entry["n_present"] == 0 for entry in report.values())
