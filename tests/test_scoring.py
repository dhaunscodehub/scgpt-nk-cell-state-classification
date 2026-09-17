"""Marker scoring and pseudo-label assignment."""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.markers import NK_STATES
from nkstate.scoring import (
    MIN_PANEL_COVERAGE, ScoringError, assign_states, score_dataframe,
    score_panel, score_states, state_separability,
)


def test_scores_have_one_column_per_state(strong_signal):
    scores = score_states(strong_signal.adata, states=NK_STATES)
    assert scores.scores.shape == (strong_signal.adata.n_obs, len(NK_STATES))
    assert scores.states == NK_STATES


def test_scores_are_tagged_as_pseudo_labels(strong_signal):
    scores = score_states(strong_signal.adata, states=NK_STATES)
    assert scores.label_type == "marker_pseudo_label"


def test_unknown_state_is_rejected(strong_signal):
    with pytest.raises(ScoringError, match="unknown state"):
        score_states(strong_signal.adata, states=("resting", "nonsense"))


def test_scoring_refuses_a_dataset_missing_the_panel(strong_signal):
    """Below the coverage floor a score is not interpretable, so it is refused."""
    from nkstate.markers import STATE_PANELS

    # Keep the background genes but drop most of the cytotoxic panel, so the
    # failure is coverage and not the control pool.
    cytotoxic = set(STATE_PANELS["cytotoxic"].genes)
    keep = [g for g in strong_signal.adata.var_names if str(g) not in cytotoxic]
    trimmed = strong_signal.adata[:, keep].copy()
    with pytest.raises(ScoringError, match="coverage|present"):
        score_states(trimmed, states=("cytotoxic",))


def test_scoring_refuses_too_small_a_background_pool(strong_signal):
    """Control-corrected scoring needs a background pool, and says so clearly."""
    from nkstate.markers import STATE_PANELS

    panel = [
        g for g in strong_signal.adata.var_names
        if str(g) in set(STATE_PANELS["cytotoxic"].genes)
    ]
    only_panel = strong_signal.adata[:, panel].copy()
    with pytest.raises(ScoringError, match="control set"):
        score_states(only_panel, states=("cytotoxic",))


def test_coverage_floor_is_documented_and_used():
    assert 0.0 < MIN_PANEL_COVERAGE < 1.0


def test_assignment_recovers_the_generating_state(strong_signal):
    from sklearn.metrics import adjusted_rand_score

    assignment = assign_states(score_states(strong_signal.adata, states=NK_STATES))
    confident = assignment.confident_mask()
    ari = adjusted_rand_score(
        strong_signal.true_labels[confident], assignment.labels[confident]
    )
    # Expected by construction: the synthetic signal is planted on the panel
    # genes, so this tests the scoring code and not the biology.
    assert ari > 0.9


def test_assignment_labels_near_ties_as_ambiguous(strong_signal):
    """A large margin must push cells into 'ambiguous' rather than force a call."""
    scores = score_states(strong_signal.adata, states=NK_STATES)
    strict = assign_states(scores, margin=10.0)
    assert strict.n_ambiguous == strict.labels.size
    assert set(strict.labels.tolist()) == {"ambiguous"}


def test_zero_margin_assigns_every_cell(strong_signal):
    scores = score_states(strong_signal.adata, states=NK_STATES)
    assignment = assign_states(scores, margin=0.0)
    assert assignment.n_ambiguous == 0
    assert "ambiguous" not in set(assignment.labels.tolist())


def test_ambiguous_fraction_grows_with_the_margin(strong_signal):
    scores = score_states(strong_signal.adata, states=NK_STATES)
    fractions = [
        assign_states(scores, margin=m).fraction_ambiguous
        for m in (0.0, 0.25, 1.0, 2.0)
    ]
    assert fractions == sorted(fractions)


def test_negative_margin_is_rejected(strong_signal):
    scores = score_states(strong_signal.adata, states=NK_STATES)
    with pytest.raises(ScoringError):
        assign_states(scores, margin=-1.0)


def test_confidence_is_non_negative(strong_signal):
    assignment = assign_states(score_states(strong_signal.adata, states=NK_STATES))
    assert float(assignment.confidence.min()) >= 0.0


def test_state_scores_are_not_all_identical(strong_signal):
    """Identical columns would make assignment arbitrary."""
    scores = score_states(strong_signal.adata, states=NK_STATES)
    columns = scores.scores.T
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            assert not np.allclose(left, right)


def test_separability_reports_pairwise_correlations(strong_signal):
    scores = score_states(strong_signal.adata, states=NK_STATES)
    report = state_separability(scores)
    assert report, "expected pairwise separability entries"


def test_score_dataframe_carries_the_caveat(strong_signal):
    frame, assignment = score_dataframe(strong_signal.adata, states=NK_STATES)
    assert len(frame) == strong_signal.adata.n_obs
    # A table on disk must not be readable as a curated annotation.
    assert set(frame["label_type"]) == {"marker_pseudo_label"}
    assert "state_pseudo_label" in frame.columns
    assert assignment.labels.size == strong_signal.adata.n_obs


def test_scoring_on_no_signal_gives_no_real_structure(no_signal):
    """With no planted signal, assignment must not recover the labels."""
    from sklearn.metrics import adjusted_rand_score

    assignment = assign_states(score_states(no_signal.adata, states=NK_STATES))
    confident = assignment.confident_mask()
    if confident.sum() < 10:
        pytest.skip("too few confident cells to compare")
    ari = adjusted_rand_score(
        no_signal.true_labels[confident], assignment.labels[confident]
    )
    assert ari < 0.1, f"recovered structure from noise (ARI {ari:.3f})"
