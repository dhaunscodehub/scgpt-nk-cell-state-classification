"""Interpretation: differential expression and marker-panel enrichment.

A cell-state classifier that reports only macro-F1 is unfalsifiable as biology.
The question this module answers is: **are the genes the model relies on the
genes that should define these states?** A classifier that separates NK states
using ribosomal and mitochondrial genes has learned library-size structure, and
its F1 will not tell you that.

Two complementary views:

:func:`differential_expression`
    Per-state one-vs-rest Wilcoxon rank-sum tests with Benjamini-Hochberg
    correction. This describes the *data*, independent of any model.

:func:`marker_enrichment`
    Whether a ranked gene list (from DE, or from a model's own coefficients)
    is enriched for the state's curated marker panel, with a hypergeometric
    p-value. This is the falsifiable check: if the model's top genes for
    "cytotoxic" are not enriched for the cytotoxic panel, the model is
    separating something other than cytotoxicity.

The enrichment test is one-sided (over-representation) and uses the measured
gene universe, not the whole transcriptome — testing against ~20000 genes when
the model only saw 2000 HVGs inflates significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Genes whose enrichment usually signals a technical artefact rather than
# biology. Reported as a warning, never filtered silently.
TECHNICAL_PREFIXES = ("MT-", "RPS", "RPL", "MRPS", "MRPL", "HB")


class InterpretationError(ValueError):
    """Raised when an interpretation cannot be computed as requested."""


@dataclass
class DEGene:
    """One gene's one-vs-rest differential expression result."""

    gene: str
    log2_fold_change: float
    statistic: float
    p_value: float
    q_value: float
    mean_in_group: float
    mean_out_group: float
    fraction_expressing_in: float

    @property
    def is_technical(self) -> bool:
        return self.gene.upper().startswith(TECHNICAL_PREFIXES)

    def to_dict(self) -> dict:
        return {
            "gene": self.gene, "log2_fold_change": self.log2_fold_change,
            "statistic": self.statistic, "p_value": self.p_value,
            "q_value": self.q_value, "mean_in_group": self.mean_in_group,
            "mean_out_group": self.mean_out_group,
            "fraction_expressing_in": self.fraction_expressing_in,
            "is_technical": self.is_technical,
        }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values.

    Implemented directly rather than pulled from statsmodels: it is six lines,
    and this way the enforced monotonicity — an adjusted p-value can never be
    smaller than that of a more significant gene — is visible and testable.
    """
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    if np.any((p_values < 0) | (p_values > 1)):
        raise InterpretationError("p-values must lie in [0, 1]")

    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p-value downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def differential_expression(
    expression: np.ndarray,
    gene_names: Sequence[str],
    labels: Sequence[str],
    group: str,
    min_cells: int = 10,
    top_n: int | None = None,
) -> list[DEGene]:
    """One-vs-rest Wilcoxon rank-sum DE for one group.

    Wilcoxon rather than a t-test: log-normalised single-cell expression is
    zero-inflated and far from normal, and the rank-sum test is what scanpy's
    ``rank_genes_groups`` defaults to, which keeps this comparable to the
    standard workflow.

    Fold changes are computed on the expm1 of the log-normalised values, so a
    "log2 fold change" is a ratio of mean normalised expression rather than a
    difference of log means — the latter is not a fold change of anything.
    """
    from scipy.stats import mannwhitneyu

    labels = np.asarray([str(l) for l in labels])
    if expression.shape[0] != labels.size:
        raise InterpretationError(
            f"expression has {expression.shape[0]} cells but {labels.size} labels"
        )
    if expression.shape[1] != len(gene_names):
        raise InterpretationError(
            f"expression has {expression.shape[1]} genes but {len(gene_names)} names"
        )

    in_group = labels == str(group)
    if in_group.sum() < min_cells:
        raise InterpretationError(
            f"group {group!r} has {int(in_group.sum())} cells, fewer than the "
            f"minimum of {min_cells}; DE on this few cells is not interpretable"
        )
    if (~in_group).sum() < min_cells:
        raise InterpretationError(
            f"only {int((~in_group).sum())} cells outside group {group!r}; "
            "one-vs-rest DE needs a comparison group"
        )

    inside = expression[in_group]
    outside = expression[~in_group]

    # Ratio of mean normalised expression, with a pseudocount so that a gene
    # absent from the out-group gives a large finite value rather than inf.
    linear_in = np.expm1(inside).mean(axis=0)
    linear_out = np.expm1(outside).mean(axis=0)
    log2_fc = np.log2((linear_in + 1e-9) / (linear_out + 1e-9))

    statistics = np.zeros(len(gene_names))
    p_values = np.ones(len(gene_names))
    for index in range(len(gene_names)):
        left, right = inside[:, index], outside[:, index]
        # A gene constant across every cell has no rank information; the test
        # would return p=1 with a warning, so short-circuit it.
        if left.max() == left.min() == right.max() == right.min():
            continue
        statistic, p_value = mannwhitneyu(left, right, alternative="two-sided")
        statistics[index] = float(statistic)
        p_values[index] = float(p_value)

    q_values = benjamini_hochberg(p_values)
    results = [
        DEGene(
            gene=str(gene_names[i]), log2_fold_change=float(log2_fc[i]),
            statistic=float(statistics[i]), p_value=float(p_values[i]),
            q_value=float(q_values[i]), mean_in_group=float(inside[:, i].mean()),
            mean_out_group=float(outside[:, i].mean()),
            fraction_expressing_in=float((inside[:, i] > 0).mean()),
        )
        for i in range(len(gene_names))
    ]
    # Rank by significance, then by effect size, so ties in a saturated
    # p-value (common with thousands of cells) are broken by fold change.
    results.sort(key=lambda g: (g.q_value, -g.log2_fold_change))
    return results[:top_n] if top_n else results


@dataclass
class EnrichmentResult:
    """Marker-panel enrichment for one ranked gene list."""

    state: str
    panel_size_in_universe: int
    n_top_genes: int
    n_overlap: int
    overlap_genes: list[str]
    fold_enrichment: float
    p_value: float
    universe_size: int
    technical_genes_in_top: list[str] = field(default_factory=list)

    @property
    def is_enriched(self) -> bool:
        return self.p_value < 0.05 and self.fold_enrichment > 1.0

    def to_dict(self) -> dict:
        return {
            "state": self.state, "panel_size_in_universe": self.panel_size_in_universe,
            "n_top_genes": self.n_top_genes, "n_overlap": self.n_overlap,
            "overlap_genes": self.overlap_genes,
            "fold_enrichment": self.fold_enrichment, "p_value": self.p_value,
            "universe_size": self.universe_size, "is_enriched": self.is_enriched,
            "technical_genes_in_top": self.technical_genes_in_top,
        }


def marker_enrichment(
    ranked_genes: Sequence[str],
    state: str,
    universe: Sequence[str],
    top_n: int = 50,
) -> EnrichmentResult:
    """Hypergeometric over-representation of a state's marker panel.

    The universe is the set of genes that were actually measured and available
    to the model. Using the full transcriptome instead would make every
    enrichment look far more significant than the experiment supports, because
    HVG selection has already enriched for informative genes.
    """
    from scipy.stats import hypergeom

    from .markers import ALL_PANELS, resolve_panel

    panel_name = resolve_panel(state)
    if panel_name is None:
        raise InterpretationError(
            f"no marker panel for {state!r}; panels available: "
            f"{sorted(ALL_PANELS)}"
        )
    universe_set = {str(g) for g in universe}
    if not universe_set:
        raise InterpretationError("the gene universe is empty")

    panel = {g for g in ALL_PANELS[panel_name].genes if g in universe_set}
    if not panel:
        raise InterpretationError(
            f"no gene from the {panel_name!r} panel is in the measured universe; "
            "enrichment cannot be tested (check that HVG selection did not "
            "remove the marker genes)"
        )

    top = [str(g) for g in ranked_genes if str(g) in universe_set][:top_n]
    if not top:
        raise InterpretationError("no ranked gene is in the universe")

    overlap = [g for g in top if g in panel]
    n_universe, n_panel, n_top, n_overlap = (
        len(universe_set), len(panel), len(top), len(overlap)
    )
    expected = n_top * n_panel / n_universe
    # Survival function at n_overlap-1 is P(X >= n_overlap): one-sided
    # over-representation, which is the only direction of interest here.
    p_value = float(hypergeom.sf(n_overlap - 1, n_universe, n_panel, n_top))

    return EnrichmentResult(
        state=state, panel_size_in_universe=n_panel, n_top_genes=n_top,
        n_overlap=n_overlap, overlap_genes=overlap,
        fold_enrichment=float(n_overlap / expected) if expected > 0 else 0.0,
        p_value=p_value, universe_size=n_universe,
        technical_genes_in_top=[
            g for g in top if g.upper().startswith(TECHNICAL_PREFIXES)
        ],
    )


def interpret_states(
    expression: np.ndarray,
    gene_names: Sequence[str],
    labels: Sequence[str],
    states: Sequence[str] | None = None,
    top_n: int = 50,
    min_cells: int = 10,
) -> dict:
    """DE plus marker enrichment for every state present in the labels.

    Returns a record whose top-level ``all_states_enriched`` field is the
    single falsifiable claim: every state's own DE genes are enriched for its
    own marker panel. If that is false for a state, the record says which.
    """
    from .markers import resolve_panel

    labels = np.asarray([str(l) for l in labels])
    present = sorted(set(labels.tolist()))
    no_panel = [s for s in present if resolve_panel(s) is None]
    if states is None:
        states = [s for s in present if resolve_panel(s) is not None]
    if not states:
        raise InterpretationError(
            f"no label in {present} has a marker panel, so enrichment cannot "
            "be tested for any class. Set interpret.enabled to false, or supply "
            "labels that correspond to a known panel."
        )

    de_results: dict[str, list[dict]] = {}
    enrichments: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for state in states:
        try:
            genes = differential_expression(
                expression, gene_names, labels, state,
                min_cells=min_cells, top_n=top_n,
            )
        except InterpretationError as error:
            skipped[state] = str(error)
            continue
        de_results[state] = [g.to_dict() for g in genes]
        enrichments[state] = marker_enrichment(
            [g.gene for g in genes], state, gene_names, top_n=top_n,
        ).to_dict()

    return {
        "states_tested": sorted(enrichments),
        "states_skipped": skipped,
        # Classes with no marker panel are listed, not silently omitted: a
        # reader needs to know that e.g. megakaryocytes were never tested
        # rather than inferring they passed.
        "classes_without_marker_panel": no_panel,
        "top_n": top_n,
        "differential_expression": de_results,
        "marker_enrichment": enrichments,
        "all_states_enriched": (
            bool(enrichments)
            and all(e["is_enriched"] for e in enrichments.values())
        ),
        "states_not_enriched": [
            s for s, e in enrichments.items() if not e["is_enriched"]
        ],
    }


def model_gene_enrichment(
    importance: dict[str, list[tuple[str, float]]],
    universe: Sequence[str],
    top_n: int = 50,
) -> dict:
    """Marker enrichment of a *model's own* top genes, per class.

    This is the stronger version of the check: it tests the genes the fitted
    model actually weights, not the genes that happen to be differentially
    expressed. A model can score well while relying on genes unrelated to the
    state, and only this test would show it.

    ``"all_classes"`` importances (tree ensembles, which give one unsigned
    vector for every class) cannot be attributed to a state, so they are
    reported as untestable rather than tested against an arbitrary panel.
    """
    from .markers import resolve_panel

    results: dict[str, dict] = {}
    untestable: list[str] = []
    for label, genes in importance.items():
        if resolve_panel(label) is None:
            untestable.append(label)
            continue
        results[label] = marker_enrichment(
            [g for g, _ in genes], label, universe, top_n=top_n,
        ).to_dict()

    return {
        "per_class_enrichment": results,
        "untestable_importance_keys": untestable,
        "note": (
            "unsigned importances shared across classes (tree ensembles) cannot "
            "be attributed to a single state and are listed as untestable"
        ) if untestable else None,
        "all_classes_enriched": (
            bool(results) and all(r["is_enriched"] for r in results.values())
        ),
    }
