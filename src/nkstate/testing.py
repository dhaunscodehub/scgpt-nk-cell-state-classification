"""Synthetic single-cell data with known state structure and donor effects.

The reference PBMC dataset is a **single donor**, so it cannot support a
donor-grouped split. These fixtures exist to supply what it cannot: multiple
donors, known state labels, known differentially expressed genes, and a
controllable donor confound.

Three knobs, each with a predicted consequence:

``state_effect``
    How strongly a cell's state shifts its marker-gene expression. At 0 the
    task is unlearnable and any reported accuracy above chance is a bug.

``donor_effect``
    A per-donor multiplicative shift applied to all genes — a batch effect. On
    its own it is a nuisance the classifier must ignore.

``donor_confound``
    Makes a donor's state composition depend on the donor. Combined with a
    donor effect this creates the failure mode donor-grouped splitting exists
    to catch: a cell-level split scores well by recognising the donor, and a
    donor-held-out split does not.

Counts are drawn from a negative binomial, which is the standard model for
scRNA-seq overdispersion. Everything here is synthetic and no biological
conclusion may be drawn from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .markers import NK_STATES, PROLIFERATION, STATE_PANELS, all_state_genes

# Background genes added so the dataset has a realistic ratio of informative to
# uninformative genes; a dataset of only marker genes would be trivially easy.
N_BACKGROUND_GENES = 400
# Negative-binomial dispersion. 0.5 is in the range reported for 10x data.
NB_DISPERSION = 0.5


@dataclass
class SyntheticSpec:
    """Shape and difficulty of a synthetic single-cell dataset."""

    n_donors: int = 6
    cells_per_donor: int = 200
    n_background_genes: int = N_BACKGROUND_GENES
    states: tuple[str, ...] = NK_STATES
    # 0 = unlearnable, 1 = strongly separable.
    state_effect: float = 0.8
    # Per-donor multiplicative expression shift (batch effect).
    donor_effect: float = 0.0
    # Make state composition depend on donor, creating a confound.
    donor_confound: bool = False
    mean_counts_per_cell: float = 3000.0
    dispersion: float = NB_DISPERSION
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_donors < 1:
            raise ValueError("n_donors must be >= 1")
        if self.cells_per_donor < 10:
            raise ValueError("cells_per_donor must be >= 10")
        for name in ("state_effect", "donor_effect"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        unknown = [s for s in self.states if s not in STATE_PANELS]
        if unknown:
            raise ValueError(f"unknown state(s) {unknown}")


@dataclass
class SyntheticDataset:
    """A synthetic dataset with the ground truth that generated it."""

    adata: object
    true_labels: np.ndarray
    donors: np.ndarray
    differential_genes: dict[str, list[str]]
    spec: SyntheticSpec
    marker_genes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def features(self) -> np.ndarray:
        """Dense expression matrix, in whatever state ``adata.X`` is in.

        On a dataset from :func:`build_preprocessed` this is log-normalised; on
        one from :func:`build_synthetic` it is raw counts. The caller chose
        which builder to use, so the distinction is theirs to track.
        """
        matrix = self.adata.X
        return (
            np.asarray(matrix.toarray(), dtype=float)
            if hasattr(matrix, "toarray")
            else np.asarray(matrix, dtype=float)
        )

    @property
    def gene_names(self) -> list[str]:
        return [str(g) for g in self.adata.var_names]

    def summary(self) -> dict:
        from collections import Counter

        return {
            "n_cells": int(self.adata.n_obs), "n_genes": int(self.adata.n_vars),
            "n_donors": len(set(self.donors.tolist())),
            "states": list(self.spec.states),
            "state_counts": dict(sorted(Counter(self.true_labels.tolist()).items())),
            "state_effect": self.spec.state_effect,
            "donor_effect": self.spec.donor_effect,
            "donor_confound": self.spec.donor_confound,
            "n_marker_genes": len(self.marker_genes),
            "n_differential_genes_per_state": {
                k: len(v) for k, v in self.differential_genes.items()
            },
            "note": "SYNTHETIC -- a fixture with known ground truth, not real cells",
        }


def build_synthetic(spec: SyntheticSpec | None = None) -> SyntheticDataset:
    """Generate a synthetic scRNA-seq dataset with known labels and DE genes."""
    import anndata
    import pandas as pd

    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)

    marker_genes = all_state_genes()
    background = tuple(f"BG{i:04d}" for i in range(spec.n_background_genes))
    # Proliferation genes included so the orthogonal-signal test has inputs.
    genes = tuple(dict.fromkeys(marker_genes + PROLIFERATION.genes + background))
    gene_index = {g: i for i, g in enumerate(genes)}

    # Baseline expression: log-normal across genes, which reproduces the
    # heavy-tailed mean-expression distribution of real scRNA-seq.
    baseline = rng.lognormal(mean=-1.0, sigma=1.2, size=len(genes))

    labels: list[str] = []
    donor_ids: list[str] = []
    rates = np.zeros((spec.n_donors * spec.cells_per_donor, len(genes)))

    row = 0
    for donor in range(spec.n_donors):
        donor_id = f"DONOR{donor:02d}"
        # A per-donor multiplicative shift, the same for every gene in that
        # donor: a batch effect.
        shift = (
            np.exp(rng.normal(0.0, 0.6 * spec.donor_effect, size=len(genes)))
            if spec.donor_effect > 0 else np.ones(len(genes))
        )
        if spec.donor_confound:
            # Each donor is dominated by one state, so donor predicts state.
            dominant = spec.states[donor % len(spec.states)]
            weights = np.full(len(spec.states), 0.05)
            weights[spec.states.index(dominant)] = 1.0 - 0.05 * (len(spec.states) - 1)
        else:
            weights = np.full(len(spec.states), 1.0 / len(spec.states))

        for _ in range(spec.cells_per_donor):
            state = spec.states[int(rng.choice(len(spec.states), p=weights))]
            multiplier = np.ones(len(genes))
            for gene in STATE_PANELS[state].genes:
                if gene in gene_index:
                    # Up to 8-fold upregulation of the state's own panel.
                    multiplier[gene_index[gene]] = 1.0 + 7.0 * spec.state_effect
            rates[row] = baseline * multiplier * shift
            labels.append(state)
            donor_ids.append(donor_id)
            row += 1

    # Scale each cell to the target library size, then draw negative-binomial
    # counts: Poisson mixed with a Gamma, the standard scRNA-seq count model.
    rates = rates / rates.sum(axis=1, keepdims=True) * spec.mean_counts_per_cell
    shape = 1.0 / spec.dispersion
    gamma = rng.gamma(shape=shape, scale=1.0 / shape, size=rates.shape)
    counts = rng.poisson(rates * gamma).astype(np.float32)

    adata = anndata.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "donor_id": donor_ids,
                "true_state": labels,
                "cell_type": ["NK cells"] * len(labels),
            },
            index=[f"cell{i:06d}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=list(genes)),
    )
    adata.uns["nkstate_provenance"] = {
        "dataset": "synthetic", "spec": spec.__dict__,
        "note": "SYNTHETIC fixture with known ground truth; not real cells",
    }

    return SyntheticDataset(
        adata=adata,
        true_labels=np.asarray(labels),
        donors=np.asarray(donor_ids),
        differential_genes={
            state: [g for g in STATE_PANELS[state].genes if g in gene_index]
            for state in spec.states
        },
        spec=spec,
        marker_genes=marker_genes,
    )


def build_preprocessed(spec: SyntheticSpec | None = None) -> SyntheticDataset:
    """Generate, quality-control and normalise a synthetic dataset in one step."""
    from .data.loading import normalise, quality_control

    dataset = build_synthetic(spec)
    filtered, _ = quality_control(
        dataset.adata, min_genes=10, min_counts=100, max_mito_fraction=1.0,
        min_cells_per_gene=1,
    )
    dataset.adata = normalise(filtered)
    # Re-read labels and donors from the filtered object: quality control can
    # drop cells, and arrays carried over from before filtering would be
    # misaligned with the matrix by exactly the number of cells removed.
    dataset.true_labels = np.asarray(
        [str(v) for v in dataset.adata.obs["true_state"]]
    )
    dataset.donors = np.asarray([str(v) for v in dataset.adata.obs["donor_id"]])
    return dataset
