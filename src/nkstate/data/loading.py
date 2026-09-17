"""AnnData loading, quality control, normalisation and gene selection.

The preprocessing order is fixed and matters:

1. **QC filter** on raw counts. Removing low-quality cells *after* normalising
   would let their counts influence the size factors of the cells that stay.
2. **Normalise to a common library size, then log1p.** Total-count
   normalisation removes the dominant technical axis (sequencing depth per
   cell); log1p stabilises the variance of counts that span four orders of
   magnitude.
3. **Highly-variable genes** selected on the log-normalised data, and — when
   donors are known — **within each donor separately**, then intersected.
   Selecting HVGs across the pooled dataset picks up donor-specific variation,
   which is exactly the signal a donor-grouped split is trying to exclude.

Raw counts are always retained in ``adata.layers["counts"]`` so a downstream
method that needs them (scGPT's expression binning, for instance) is not forced
to work backwards from log-normalised values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Default QC thresholds. Deliberately permissive: aggressive filtering on an NK
# subset removes the small, low-RNA resting cells preferentially, which biases
# the state distribution before any analysis begins.
DEFAULT_MIN_GENES = 200
DEFAULT_MIN_COUNTS = 500
DEFAULT_MAX_MITO_FRACTION = 0.20
DEFAULT_TARGET_SUM = 1e4


class ScanpyUnavailable(RuntimeError):
    """Raised when scanpy/anndata are needed but not installed."""


class DataError(ValueError):
    """Raised for an unusable dataset or an inconsistent request."""


def _scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ScanpyUnavailable(
            "scanpy is required for single-cell data handling: pip install scanpy"
        ) from exc
    return sc


@dataclass
class QCReport:
    """What quality control removed, and why."""

    n_cells_before: int
    n_cells_after: int
    n_genes_before: int
    n_genes_after: int
    removed_low_genes: int
    removed_low_counts: int
    removed_high_mito: int
    median_genes_per_cell: float
    median_counts_per_cell: float
    median_mito_fraction: float
    thresholds: dict = field(default_factory=dict)

    @property
    def fraction_retained(self) -> float:
        return self.n_cells_after / self.n_cells_before if self.n_cells_before else 0.0

    def to_dict(self) -> dict:
        return {
            "n_cells_before": self.n_cells_before, "n_cells_after": self.n_cells_after,
            "fraction_retained": self.fraction_retained,
            "n_genes_before": self.n_genes_before, "n_genes_after": self.n_genes_after,
            "removed_low_genes": self.removed_low_genes,
            "removed_low_counts": self.removed_low_counts,
            "removed_high_mito": self.removed_high_mito,
            "median_genes_per_cell": self.median_genes_per_cell,
            "median_counts_per_cell": self.median_counts_per_cell,
            "median_mito_fraction": self.median_mito_fraction,
            "thresholds": self.thresholds,
        }


def load_anndata(path: str | Path):
    """Read an ``.h5ad`` file, or a 10x directory/``.h5``."""
    sc = _scanpy()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.is_dir():
        return sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    if path.suffix == ".h5ad":
        return sc.read_h5ad(path)
    if path.suffix == ".h5":
        return sc.read_10x_h5(path)
    raise DataError(
        f"unsupported format {path.suffix!r}; expected .h5ad, .h5 or a 10x directory"
    )


def load_reference_pbmc(cache_dir: str | Path = "data/scanpy"):
    """Download the annotated 3k-PBMC reference dataset.

    Zheng et al., *Nat Commun* 8:14049 (2017), as distributed and annotated by
    scanpy. 2638 cells x 1838 genes with eight curated cell-type labels
    including 154 NK cells.

    Used as the real-data validation set for the **cell-type** classifier,
    where the labels are curated and independent of this package. It is a
    **single donor**, so it cannot support a donor-grouped split — that is what
    the synthetic multi-donor fixtures are for, and the distinction is enforced
    rather than assumed.
    """
    import os

    sc = _scanpy()
    try:
        import certifi

        # scanpy downloads over urllib, which on python.org macOS builds has no
        # CA bundle unless Install Certificates.command has been run.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    sc.settings.datasetdir = str(Path(cache_dir))
    processed = sc.datasets.pbmc3k_processed()

    # ``pbmc3k_processed.X`` is z-scored to unit variance and clipped at 10, so
    # it contains negative values over only 1838 highly variable genes. Using
    # it directly would be wrong in two ways that produce plausible-looking
    # numbers rather than errors: log fold changes computed on z-scores are not
    # fold changes of anything, and marker scoring against a scaled matrix
    # compares genes whose scales have already been equalised, destroying the
    # expression-level information the panels rely on.
    #
    # ``.raw`` holds the log1p-of-CPM-normalised matrix over all 13714 detected
    # genes, which is what every downstream step here actually wants.
    if processed.raw is None:
        raise DataError(
            "pbmc3k_processed has no .raw matrix; this build of scanpy "
            "distributes a different version of the dataset than expected and "
            "the log-normalised values cannot be recovered from the scaled .X"
        )
    adata = processed.raw.to_adata()
    adata.obs = processed.obs.copy()
    for key in ("X_pca", "X_umap", "X_tsne"):
        if key in processed.obsm:
            adata.obsm[key] = processed.obsm[key]

    adata.obs["cell_type"] = processed.obs["louvain"].astype(str)
    # One donor: recorded explicitly so a donor-grouped split refuses rather
    # than silently degenerating into a cell-level split.
    adata.obs["donor_id"] = "pbmc3k_donor1"
    # Already normalised by the distributor. Recorded so ``normalise`` refuses
    # to log1p a second time, which would compress all expression towards zero
    # while leaving the data superficially intact.
    adata.uns["nkstate_normalisation"] = {
        "target_sum": 1e4, "log1p": True, "counts_layer": False,
        "applied_by": "scanpy.datasets.pbmc3k_processed (pre-normalised)",
    }
    adata.uns["nkstate_provenance"] = {
        "dataset": "pbmc3k_processed",
        "source": "scanpy.datasets, from 10x Genomics",
        "reference": "Zheng et al., Nat Commun 8:14049 (2017)",
        "n_donors": 1,
        "matrix": "log1p of CPM-normalised counts, taken from .raw",
        "n_genes_in_raw": int(adata.n_vars),
        "labels": "curated cell types (louvain clusters annotated by scanpy)",
        "scaled_X_discarded": (
            "the distributed .X is z-scored over 1838 HVGs and is not used"
        ),
    }
    return adata


def quality_control(
    adata,
    min_genes: int = DEFAULT_MIN_GENES,
    min_counts: int = DEFAULT_MIN_COUNTS,
    max_mito_fraction: float = DEFAULT_MAX_MITO_FRACTION,
    min_cells_per_gene: int = 3,
    mito_prefix: str = "MT-",
) -> tuple[object, QCReport]:
    """Filter cells and genes on raw counts, reporting what each rule removed.

    Attributing removals to individual rules matters: if the mitochondrial
    filter removes 40% of cells, the tissue was stressed and no amount of
    downstream modelling fixes it. A single combined count hides that.
    """
    sc = _scanpy()
    n_cells_before, n_genes_before = adata.shape

    counts = _as_dense(adata.X)
    genes_per_cell = (counts > 0).sum(axis=1)
    counts_per_cell = counts.sum(axis=1)
    mito_mask = np.asarray(
        [str(g).upper().startswith(mito_prefix) for g in adata.var_names]
    )
    mito_fraction = (
        counts[:, mito_mask].sum(axis=1) / np.maximum(counts_per_cell, 1)
        if mito_mask.any() else np.zeros(n_cells_before)
    )

    fail_genes = genes_per_cell < min_genes
    fail_counts = counts_per_cell < min_counts
    fail_mito = mito_fraction > max_mito_fraction
    keep = ~(fail_genes | fail_counts | fail_mito)

    if not keep.any():
        raise DataError(
            f"quality control removed every cell "
            f"({int(fail_genes.sum())} below {min_genes} genes, "
            f"{int(fail_counts.sum())} below {min_counts} counts, "
            f"{int(fail_mito.sum())} above {max_mito_fraction:.0%} mitochondrial). "
            "Check that adata.X holds raw counts rather than normalised values."
        )

    filtered = adata[keep].copy()
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(filtered, min_cells=min_cells_per_gene)

    report = QCReport(
        n_cells_before=n_cells_before, n_cells_after=int(filtered.n_obs),
        n_genes_before=n_genes_before, n_genes_after=int(filtered.n_vars),
        removed_low_genes=int(fail_genes.sum()),
        removed_low_counts=int(fail_counts.sum()),
        removed_high_mito=int(fail_mito.sum()),
        median_genes_per_cell=float(np.median(genes_per_cell)),
        median_counts_per_cell=float(np.median(counts_per_cell)),
        median_mito_fraction=float(np.median(mito_fraction)),
        thresholds={
            "min_genes": min_genes, "min_counts": min_counts,
            "max_mito_fraction": max_mito_fraction,
            "min_cells_per_gene": min_cells_per_gene,
        },
    )
    return filtered, report


def normalise(adata, target_sum: float = DEFAULT_TARGET_SUM, keep_counts: bool = True):
    """Total-count normalise then log1p, retaining raw counts in a layer.

    Raw counts are kept because scGPT's expression binning and any
    count-distribution model need them, and reconstructing counts from
    log-normalised values is not possible.
    """
    sc = _scanpy()
    result = adata.copy()

    existing = result.uns.get("nkstate_normalisation")
    if existing and existing.get("log1p"):
        # Applying log1p twice leaves a matrix that still looks like plausible
        # log-normalised expression but has had every fold change flattened, so
        # this is a refusal rather than a warning.
        result.uns["nkstate_normalisation"] = {
            **existing,
            "skipped_second_pass": (
                "already log1p-normalised by "
                f"{existing.get('applied_by', 'an earlier call')}; not repeated"
            ),
        }
        return result

    if keep_counts and "counts" not in result.layers:
        result.layers["counts"] = result.X.copy()
    sc.pp.normalize_total(result, target_sum=target_sum)
    sc.pp.log1p(result)
    result.uns["nkstate_normalisation"] = {
        "target_sum": target_sum, "log1p": True, "counts_layer": keep_counts,
    }
    return result


def select_hvgs(
    adata,
    n_top_genes: int = 2000,
    donor_key: str | None = None,
    keep_genes: tuple[str, ...] = (),
):
    """Select highly variable genes, optionally within each donor.

    With ``donor_key`` set, HVGs are selected per donor and the union of
    per-donor selections is used. Selecting across the pooled dataset would
    rank donor-specific variation highly — that variation is real, but it is
    the confound a donor-grouped split exists to exclude, so training on it
    defeats the split.

    ``keep_genes`` are retained regardless of variance. Marker genes must
    survive selection or the marker scores and the interpretation step lose
    their inputs.
    """
    sc = _scanpy()
    if n_top_genes < 10:
        raise DataError("n_top_genes below 10 cannot support a classifier")
    result = adata.copy()

    if donor_key and donor_key in result.obs:
        donors = [d for d in result.obs[donor_key].unique()]
        if len(donors) < 2:
            # One donor: per-donor selection is identical to pooled selection,
            # so say so rather than pretending the guard did something.
            result.uns["nkstate_hvg"] = {
                "strategy": "pooled (only one donor present)",
                "donor_key": donor_key, "n_donors": 1,
            }
            selected = _pooled_hvgs(sc, result, n_top_genes)
        else:
            per_donor: list[set[str]] = []
            for donor in donors:
                subset = result[result.obs[donor_key] == donor]
                if subset.n_obs < 20:
                    continue
                per_donor.append(set(_pooled_hvgs(sc, subset.copy(), n_top_genes)))
            if not per_donor:
                raise DataError(
                    f"no donor has at least 20 cells, so per-donor HVG selection is "
                    f"impossible; donors have sizes "
                    f"{result.obs[donor_key].value_counts().to_dict()}"
                )
            union: list[str] = []
            for genes in per_donor:
                for gene in sorted(genes):
                    if gene not in union:
                        union.append(gene)
            selected = union
            result.uns["nkstate_hvg"] = {
                "strategy": "per-donor union",
                "donor_key": donor_key, "n_donors": len(per_donor),
                "n_selected": len(selected),
            }
    else:
        selected = _pooled_hvgs(sc, result, n_top_genes)
        result.uns["nkstate_hvg"] = {"strategy": "pooled", "donor_key": None}

    forced = [g for g in keep_genes if g in result.var_names and g not in set(selected)]
    final = list(selected) + forced
    result.uns["nkstate_hvg"]["n_forced_marker_genes"] = len(forced)
    result.uns["nkstate_hvg"]["n_genes"] = len(final)
    return result[:, final].copy()


def _pooled_hvgs(sc, adata, n_top_genes: int) -> list[str]:
    n = min(n_top_genes, adata.n_vars)
    working = adata.copy()
    sc.pp.highly_variable_genes(working, n_top_genes=n)
    return working.var_names[working.var["highly_variable"]].tolist()


def _as_dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def expression_matrix(adata, layer: str | None = None) -> np.ndarray:
    """Dense expression matrix, from ``.X`` or a named layer."""
    source = adata.layers[layer] if layer else adata.X
    return _as_dense(source)


def subset_nk_cells(
    adata,
    cell_type_key: str | None = "cell_type",
    nk_labels: tuple[str, ...] = ("NK cells", "NK", "NK_cell", "nk"),
    identity_threshold: float | None = None,
):
    """Isolate NK cells, by curated label when available or by marker score.

    A curated label is preferred and used whenever the column exists. Marker
    gating is the fallback and is explicitly worse: it requires a threshold on
    a continuous score, and CD3+ cytotoxic T and NKT cells share most of the NK
    effector programme. When gating is used, T-cell genes are scored too and
    cells positive for them are excluded, which is the minimum needed to make
    the gate mean anything.
    """
    from ..markers import NK_IDENTITY, T_CELL_EXCLUSION
    from ..scoring import score_panel

    if cell_type_key and cell_type_key in adata.obs:
        labels = adata.obs[cell_type_key].astype(str)
        mask = labels.isin(nk_labels).to_numpy()
        if mask.any():
            result = adata[mask].copy()
            result.uns["nkstate_nk_gate"] = {
                "method": "curated label", "key": cell_type_key,
                "labels_used": [l for l in nk_labels if (labels == l).any()],
                "n_cells": int(result.n_obs),
            }
            return result

    if identity_threshold is None:
        raise DataError(
            f"no curated NK label found in obs[{cell_type_key!r}] and no "
            "identity_threshold given. Marker gating needs an explicit threshold "
            "because the NK identity score is continuous; pick one from the score "
            "distribution rather than letting a default decide."
        )

    identity = score_panel(adata, NK_IDENTITY)
    t_cell = score_panel(adata, T_CELL_EXCLUSION)
    mask = (identity >= identity_threshold) & (t_cell < identity_threshold)
    if not mask.any():
        raise DataError(
            f"marker gating at threshold {identity_threshold} selected no cells "
            f"(identity score range {identity.min():.3f} to {identity.max():.3f})"
        )
    result = adata[mask].copy()
    result.obs["nk_identity_score"] = identity[mask]
    result.obs["t_cell_score"] = t_cell[mask]
    result.uns["nkstate_nk_gate"] = {
        "method": "marker score",
        "identity_threshold": identity_threshold,
        "n_cells": int(result.n_obs),
        "n_excluded_t_cell_positive": int(((identity >= identity_threshold) & (t_cell >= identity_threshold)).sum()),
        "caveat": "CD3+ cytotoxic T and NKT cells share the NK effector programme; "
                  "a marker gate is weaker than a curated label",
    }
    return result
