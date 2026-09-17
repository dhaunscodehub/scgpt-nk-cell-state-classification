"""Marker-based scoring of NK functional states.

Scoring uses the control-gene-corrected approach of Tirosh et al., *Science*
352:189 (2016), as implemented in ``scanpy.tl.score_genes``: the mean
expression of the panel minus the mean expression of a randomly sampled
reference set matched for expression bin. The correction matters because a
panel of highly expressed genes (the cytotoxic granule transcripts, say) would
otherwise score high in every cell simply because those genes are abundant.

Everything here produces **pseudo-labels**. A state assigned from marker scores
is a hypothesis about a cell, and the functions below refuse to present it
otherwise:

* :func:`assign_states` returns an explicit ``confidence`` and marks
  low-margin cells ``"ambiguous"`` rather than forcing a call.
* Every returned record carries ``label_type="marker_pseudo_label"``.
* :func:`state_separability` reports how distinguishable the scores actually
  are, so a classifier trained on them can be judged against that ceiling
  rather than against 100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .markers import NK_STATES, STATE_PANELS, MarkerPanel, genes_present

# Minimum fraction of a panel's genes that must be present to score it. Below
# this the score measures a different gene set from the one the panel defines.
MIN_PANEL_COVERAGE = 0.4

# Minimum number of non-panel genes needed for control-corrected scoring.
# ``scanpy.tl.score_genes`` bins genes by mean expression and samples controls
# from the panel genes' bins; with too small a pool some bin has no candidate
# and it fails with a message that does not identify the cause.
MIN_BACKGROUND_GENES = 50
# Minimum margin between the best and second-best state score for a confident
# call, in units of the score's own standard deviation across cells.
DEFAULT_MARGIN = 0.25


class ScoringError(ValueError):
    """Raised when a panel cannot be scored against a dataset."""


def score_panel(
    adata,
    panel: MarkerPanel,
    n_control_genes: int = 50,
    n_bins: int = 25,
    min_coverage: float = MIN_PANEL_COVERAGE,
    seed: int = 0,
) -> np.ndarray:
    """Control-corrected mean expression of a panel, per cell.

    Raises when panel coverage is below ``min_coverage``: a score computed from
    3 of 12 panel genes is not the panel's score, and returning it would put an
    incomparable number alongside the others.
    """
    import scanpy as sc

    available = set(map(str, adata.var_names))
    present, missing = genes_present(panel, available)
    coverage = len(present) / len(panel.genes)
    if coverage < min_coverage:
        raise ScoringError(
            f"panel {panel.name!r} has only {len(present)}/{len(panel.genes)} genes "
            f"present ({coverage:.0%}, below the {min_coverage:.0%} minimum). "
            f"Missing: {missing[:8]}. A score from this subset is not comparable "
            "with the other panels."
        )

    # score_genes subtracts the mean of an expression-matched control set, so it
    # needs a background pool substantially larger than the panel itself.
    # Without this guard scanpy raises "No control genes found in any cut",
    # which does not say what the caller actually did wrong.
    n_background = adata.n_vars - len(present)
    if n_background < MIN_BACKGROUND_GENES:
        raise ScoringError(
            f"scoring panel {panel.name!r} needs at least {MIN_BACKGROUND_GENES} "
            f"non-panel genes to build an expression-matched control set, but the "
            f"dataset has {adata.n_vars} genes of which {len(present)} are in the "
            "panel. Score against the full gene set rather than a pre-filtered "
            "subset, or raise n_top_genes."
        )

    working = adata.copy()
    key = f"_score_{panel.name}"
    sc.tl.score_genes(
        working, gene_list=present, score_name=key,
        ctrl_size=min(n_control_genes, max(len(present), 1) * 10),
        n_bins=min(n_bins, max(2, working.n_vars // 20)),
        random_state=seed,
    )
    return np.asarray(working.obs[key], dtype=float)


@dataclass
class StateScores:
    """Per-cell marker scores for every functional state."""

    scores: np.ndarray                  # (n_cells, n_states)
    states: tuple[str, ...]
    coverage: dict[str, float]
    n_cells: int
    label_type: str = "marker_pseudo_label"

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.scores, columns=list(self.states))

    def to_dict(self) -> dict:
        return {
            "n_cells": self.n_cells, "states": list(self.states),
            "panel_coverage": self.coverage, "label_type": self.label_type,
            "score_means": {
                s: float(self.scores[:, i].mean()) for i, s in enumerate(self.states)
            },
            "score_sds": {
                s: float(self.scores[:, i].std()) for i, s in enumerate(self.states)
            },
        }


def score_states(
    adata,
    states: tuple[str, ...] = NK_STATES,
    seed: int = 0,
    min_coverage: float = MIN_PANEL_COVERAGE,
) -> StateScores:
    """Score every functional-state panel against a dataset."""
    unknown = [s for s in states if s not in STATE_PANELS]
    if unknown:
        raise ScoringError(f"unknown state(s) {unknown}; available: {list(STATE_PANELS)}")

    available = set(map(str, adata.var_names))
    columns, coverage = [], {}
    for state in states:
        panel = STATE_PANELS[state]
        present, _ = genes_present(panel, available)
        coverage[state] = len(present) / len(panel.genes)
        columns.append(score_panel(adata, panel, min_coverage=min_coverage, seed=seed))
    return StateScores(
        scores=np.column_stack(columns), states=tuple(states),
        coverage=coverage, n_cells=int(adata.n_obs),
    )


@dataclass
class StateAssignment:
    """Pseudo-labels derived from marker scores, with their confidence."""

    labels: np.ndarray            # str, may include "ambiguous"
    confidence: np.ndarray        # margin between best and second-best, in SD units
    scores: StateScores
    margin_threshold: float
    n_ambiguous: int
    label_type: str = "marker_pseudo_label"

    @property
    def fraction_ambiguous(self) -> float:
        return self.n_ambiguous / len(self.labels) if len(self.labels) else 0.0

    def confident_mask(self) -> np.ndarray:
        return self.labels != "ambiguous"

    def to_dict(self) -> dict:
        from collections import Counter

        return {
            "n_cells": int(len(self.labels)),
            "label_type": self.label_type,
            "margin_threshold": self.margin_threshold,
            "n_ambiguous": self.n_ambiguous,
            "fraction_ambiguous": self.fraction_ambiguous,
            "label_counts": dict(sorted(Counter(self.labels.tolist()).items())),
            "panel_coverage": self.scores.coverage,
            "caveat": (
                "These are marker-derived pseudo-labels, not measured ground "
                "truth. A classifier's agreement with them is agreement with a "
                "marker heuristic, and must not be reported as accuracy."
            ),
        }


def assign_states(
    scores: StateScores, margin: float = DEFAULT_MARGIN
) -> StateAssignment:
    """Assign each cell its highest-scoring state, or ``"ambiguous"``.

    Scores are standardised per state before comparison, because the panels
    differ in size and in baseline expression so their raw scores are not on a
    common scale. A cell whose best and second-best standardised scores differ
    by less than ``margin`` is left unassigned: the state panels genuinely
    overlap (see :func:`~nkstate.markers.panel_overlap`), so forcing a call on
    a near-tie manufactures a label the data does not support.
    """
    if scores.n_cells == 0:
        raise ScoringError("no cells to assign")
    if margin < 0:
        raise ScoringError("margin must be non-negative")

    matrix = scores.scores
    sd = matrix.std(axis=0)
    # A state with zero variance carries no information; standardising it would
    # divide by zero, so it is left at its mean and cannot win a comparison.
    safe_sd = np.where(sd > 0, sd, 1.0)
    standardised = (matrix - matrix.mean(axis=0)) / safe_sd
    standardised[:, sd == 0] = -np.inf

    order = np.argsort(-standardised, axis=1)
    best = order[:, 0]
    margins = (
        standardised[np.arange(len(best)), best]
        - standardised[np.arange(len(best)), order[:, 1]]
        if standardised.shape[1] > 1
        else np.full(len(best), np.inf)
    )
    labels = np.asarray([scores.states[i] for i in best], dtype=object)
    ambiguous = margins < margin
    labels[ambiguous] = "ambiguous"

    return StateAssignment(
        labels=labels.astype(str), confidence=margins, scores=scores,
        margin_threshold=margin, n_ambiguous=int(ambiguous.sum()),
    )


def state_separability(scores: StateScores) -> dict:
    """How distinguishable the marker scores are from each other.

    Reports the pairwise correlation between state scores and the silhouette
    of the argmax assignment. This is the **ceiling** on what a classifier
    trained against these pseudo-labels can meaningfully achieve: if the
    cytotoxic and activated scores correlate at 0.9, a classifier separating
    them is separating noise.
    """
    from itertools import combinations

    from sklearn.metrics import silhouette_score

    matrix = scores.scores
    correlations = {}
    for i, j in combinations(range(len(scores.states)), 2):
        a, b = matrix[:, i], matrix[:, j]
        if a.std() == 0 or b.std() == 0:
            correlations[f"{scores.states[i]}-{scores.states[j]}"] = None
            continue
        correlations[f"{scores.states[i]}-{scores.states[j]}"] = float(
            np.corrcoef(a, b)[0, 1]
        )

    labels = matrix.argmax(axis=1)
    silhouette = None
    if len(set(labels.tolist())) > 1 and matrix.shape[0] > len(set(labels.tolist())):
        silhouette = float(silhouette_score(matrix, labels))

    finite = [v for v in correlations.values() if v is not None]
    return {
        "pairwise_score_correlations": correlations,
        "max_absolute_correlation": float(max(abs(v) for v in finite)) if finite else None,
        "argmax_silhouette": silhouette,
        "interpretation": (
            "Marker scores that correlate strongly cannot be separated reliably; "
            "the silhouette bounds how well any classifier can reproduce the "
            "argmax assignment."
        ),
    }


def score_dataframe(
    adata,
    states: tuple[str, ...] = NK_STATES,
    seed: int = 0,
    margin: float = DEFAULT_MARGIN,
):
    """Per-cell scores, assignment and confidence as one DataFrame.

    Every row carries ``label_type`` so that a table written to disk cannot be
    read later as though the state column were a curated annotation.
    """
    scores = score_states(adata, states=states, seed=seed)
    assignment = assign_states(scores, margin=margin)
    frame = scores.to_frame()
    frame.index = adata.obs_names
    frame["state_pseudo_label"] = assignment.labels
    frame["confidence_margin"] = assignment.confidence
    frame["label_type"] = assignment.label_type
    for key in ("donor_id", "cell_type"):
        if key in adata.obs:
            frame[key] = adata.obs[key].to_numpy()
    return frame, assignment
