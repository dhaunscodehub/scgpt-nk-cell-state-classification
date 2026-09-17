"""End-to-end pipeline: load, label, split, fit, evaluate, interpret.

The order is fixed and the reason is leakage. Specifically:

1. **Load and QC.** Cell and gene filters depend only on the cells themselves.
2. **Normalise.** Per-cell library-size normalisation is a within-cell
   operation and cannot move information between cells.
3. **Label.** Marker scoring is per cell.
4. **Split by donor.** Everything after this point sees folds, not the dataset.
5. **Select HVGs on the training fold only.** This is the step most often done
   wrong. Choosing highly variable genes on the full dataset uses the test
   cells' expression to decide what the model is allowed to see — a real leak,
   and one that does not announce itself.
6. **Standardise on training statistics, fit, predict.**
7. **Evaluate at cell and donor level; interpret.**

Step 5 is the reason HVG selection is not part of preprocessing here even
though scanpy tutorials place it there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import RunConfig
from .evaluate import EvaluationResult, compare_models, evaluate_predictions
from .io_utils import provenance, write_json
from .markers import STATE_PANELS, all_state_genes


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot proceed."""


# Exit-code-worthy: a leak was detected in a split that promised none.
class LeakageDetected(PipelineError):
    """Raised when a donor-grouped split turns out to share donors."""


@dataclass
class PipelineResult:
    """Everything one pipeline run produced."""

    name: str
    n_cells: int
    n_genes: int
    n_donors: int
    split_kind: str
    label_type: str
    class_balance: dict
    evaluations: list[EvaluationResult] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    interpretation: dict = field(default_factory=dict)
    scgpt: dict = field(default_factory=dict)
    qc: dict = field(default_factory=dict)
    hvg: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dataset": {
                "n_cells": self.n_cells, "n_genes": self.n_genes,
                "n_donors": self.n_donors, "class_balance": self.class_balance,
            },
            "split_kind": self.split_kind, "label_type": self.label_type,
            "qc": self.qc, "hvg_selection": self.hvg,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "comparison": self.comparison,
            "interpretation": self.interpretation,
            "scgpt": self.scgpt,
            "warnings": self.warnings,
            "provenance": provenance(),
        }


def _dense(matrix) -> np.ndarray:
    """Return a dense float array from a sparse or dense AnnData matrix."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def prepare_dataset(config: RunConfig):
    """Load, QC and normalise, returning an AnnData with labels attached."""
    from .data.loading import (
        load_anndata, load_reference_pbmc, normalise, quality_control,
        subset_nk_cells,
    )

    if config.data.source == "reference_pbmc":
        adata = load_reference_pbmc()
    elif config.data.source == "h5ad":
        adata = load_anndata(config.data.path, donor_key=config.data.donor_key)
    else:
        from .testing import build_synthetic

        adata = build_synthetic(config.synthetic.to_spec()).adata

    adata, qc = quality_control(
        adata,
        min_genes=config.data.min_genes,
        min_counts=config.data.min_counts,
        max_mito_fraction=config.data.max_mito_fraction,
        min_cells_per_gene=config.data.min_cells_per_gene,
    )
    adata = normalise(adata, target_sum=config.data.target_sum)

    if config.data.subset_nk:
        kwargs = {}
        if config.data.nk_labels:
            kwargs["nk_labels"] = tuple(config.data.nk_labels)
        adata = subset_nk_cells(
            adata,
            cell_type_key=config.data.cell_type_key,
            identity_threshold=config.data.nk_identity_threshold,
            **kwargs,
        )
    return adata, qc.to_dict()


def assign_labels(adata, config: RunConfig) -> tuple[np.ndarray, str, list[str]]:
    """Attach state labels, returning (labels, label_type, warnings)."""
    warnings: list[str] = []

    if config.labels.source == "annotation":
        key = config.data.label_key
        if not key or key not in adata.obs:
            raise PipelineError(
                f"labels.source is 'annotation' but data.label_key "
                f"({key!r}) is not a column in obs; available columns: "
                f"{sorted(adata.obs.columns)}"
            )
        return (
            np.asarray([str(v) for v in adata.obs[key]]), "curated_annotation", warnings
        )

    from .scoring import assign_states, score_states

    scores = score_states(adata, states=tuple(config.labels.states))
    assignment = assign_states(scores, margin=config.labels.margin)
    labels = np.asarray(assignment.labels)
    n_ambiguous = int((labels == "ambiguous").sum())
    if n_ambiguous:
        warnings.append(
            f"{n_ambiguous}/{labels.size} cells ({100 * n_ambiguous / labels.size:.1f}%) "
            f"scored within the {config.labels.margin} margin between their top two "
            "states and are labelled 'ambiguous'"
            + (
                "; they are excluded from training and evaluation"
                if config.labels.drop_ambiguous else
                "; they are retained as their own class"
            )
        )
    warnings.append(
        "state labels are marker-panel pseudo-labels derived from expression, "
        "not independent ground truth: a classifier scored against them measures "
        "agreement with the marker rule, and cannot exceed it"
    )
    return labels, "marker_pseudo_label", warnings


def build_split(adata, labels: np.ndarray, config: RunConfig):
    """Construct the train/val/test split and re-assert it is clean."""
    from .data.splits import assert_no_donor_leakage, cell_split, donor_split

    donors = np.asarray([str(d) for d in adata.obs[config.data.donor_key]])
    if config.split.kind == "donor":
        split = donor_split(
            donors,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            seed=config.split.seed,
        )
        # Asserted again here, not only inside the constructor: this is the
        # invariant the whole result rests on, and it is cheap to check twice.
        assert_no_donor_leakage(split, donors)
    else:
        split = cell_split(
            donors,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            seed=config.split.seed,
        )
    return split, donors


def select_training_hvgs(
    adata, train_index: np.ndarray, config: RunConfig
) -> tuple[list[str], dict]:
    """Choose highly variable genes using the training fold only.

    Marker-panel genes are added back afterwards regardless of their variance
    rank. That is a deliberate, disclosed inclusion of prior knowledge — not a
    leak, because the panels come from published literature and not from this
    dataset's test cells. Without it, HVG selection sometimes drops the very
    genes the states are defined by, and the classifier is then asked to
    recover cytotoxicity without seeing GNLY.
    """
    from .data.loading import select_hvgs

    train_view = adata[train_index].copy()
    present = set(map(str, adata.var_names))
    selected = select_hvgs(
        train_view,
        n_top_genes=min(config.data.n_top_genes, train_view.n_vars),
        donor_key=config.data.donor_key,
        keep_genes=tuple(g for g in all_state_genes() if g in present),
    )
    # select_hvgs returns the subset AnnData; the pipeline needs the gene names
    # so it can index the *full* matrix with training-derived gene choices.
    return [str(g) for g in selected.var_names], dict(selected.uns["nkstate_hvg"])


def run_pipeline(config: RunConfig, output_dir: str | Path | None = None) -> PipelineResult:
    """Execute the full pipeline for one configuration."""
    from .data.splits import class_balance
    from .interpret import interpret_states, model_gene_enrichment
    from .models.baselines import ModelError, fit_baseline

    adata, qc = prepare_dataset(config)
    labels, label_type, warnings = assign_labels(adata, config)

    if config.labels.drop_ambiguous:
        keep = labels != "ambiguous"
        if keep.sum() < labels.size:
            adata = adata[keep].copy()
            labels = labels[keep]
    if labels.size == 0:
        raise PipelineError(
            "no cell has a confident state label; lower labels.margin or check "
            "that the marker panels are present in this dataset"
        )

    present = sorted(set(labels.tolist()))
    if len(present) < 2:
        raise PipelineError(
            f"only one state ({present}) survives labelling; a classifier needs "
            "at least two. This usually means the dataset does not contain the "
            "states being asked for."
        )

    split, donors = build_split(adata, labels, config)
    hvg_genes, hvg_info = select_training_hvgs(adata, split.train, config)
    gene_index = [adata.var_names.get_loc(g) for g in hvg_genes]
    features = _dense(adata.X)[:, gene_index]

    train_x, train_y = features[split.train], labels[split.train]
    test_x, test_y = features[split.test], labels[split.test]
    test_donors = donors[split.test]

    evaluations: list[EvaluationResult] = []
    importances: dict[str, dict] = {}
    for name in config.models.baselines:
        try:
            fitted = fit_baseline(
                name, train_x, train_y, hvg_genes,
                seed=config.models.seed, balanced=config.models.balanced,
            )
        except ModelError as error:
            warnings.append(f"model {name} skipped: {error}")
            continue

        predicted = fitted.predict(test_x)
        probabilities = None
        try:
            probabilities = fitted.predict_proba(test_x)
        except ModelError:
            pass

        evaluations.append(
            evaluate_predictions(
                model=name, true_labels=test_y, predicted_labels=predicted,
                donor_ids=test_donors, probabilities=probabilities,
                model_classes=fitted.classes, split_kind=config.split.kind,
                label_type=label_type,
            )
        )
        importance = fitted.gene_importance()
        if importance:
            importances[name] = model_gene_enrichment(
                importance, hvg_genes, top_n=config.interpret.top_n_genes,
            )

    if not evaluations:
        raise PipelineError(
            "no model could be fitted; see warnings for the reason for each"
        )

    balance = class_balance(labels, split)
    unseen = balance["classes_in_test_but_not_train"]
    if unseen:
        warnings.append(
            f"state(s) {unseen} appear in the test fold but not in training; "
            "a classifier cannot predict a class it never saw, so their recall "
            "is 0 by construction and the macro-F1 is bounded below 1"
        )

    interpretation: dict = {}
    if config.interpret.enabled:
        interpretation = interpret_states(
            train_x, hvg_genes, train_y,
            top_n=config.interpret.top_n_genes,
            min_cells=config.interpret.min_cells_per_state,
        )
        interpretation["model_gene_enrichment"] = importances

        # A class whose own DE genes are not enriched for its own panel is the
        # most informative single line of output here, so it is surfaced as a
        # warning rather than left for a reader to find in the JSON.
        unenriched = interpretation.get("states_not_enriched") or []
        if unenriched:
            warnings.append(
                f"marker enrichment failed for {unenriched}: the top "
                "differentially expressed genes for these classes are not "
                "enriched for their own marker panel. This is expected when a "
                "class shares its panel with a sibling subtype — a one-vs-rest "
                "comparison cannot recover markers the comparison group also "
                "expresses — and is a statement about panel resolution, not "
                "about the classifier's accuracy"
            )
        technical = {
            state: record["technical_genes_in_top"]
            for state, record in interpretation.get("marker_enrichment", {}).items()
            if record.get("technical_genes_in_top")
        }
        if technical:
            warnings.append(
                "mitochondrial or ribosomal genes appear among the top "
                f"differentially expressed genes for {sorted(technical)}; these "
                "usually track library size and cell stress rather than the "
                "biology being classified"
            )

    result = PipelineResult(
        name=config.name, n_cells=int(adata.n_obs), n_genes=len(hvg_genes),
        n_donors=len(set(donors.tolist())), split_kind=config.split.kind,
        label_type=label_type, class_balance=balance,
        evaluations=evaluations, comparison=compare_models(evaluations),
        interpretation=interpretation,
        scgpt=scgpt_stage(config, hvg_genes, len(present)),
        qc=qc, warnings=warnings, hvg=hvg_info,
    )

    target = Path(output_dir or config.output_dir)
    write_json(target / f"{config.name}.json", result.to_dict())
    return result


def scgpt_stage(config: RunConfig, gene_names: Sequence[str], n_classes: int) -> dict:
    """Report the scGPT stage: what it would run, and what it needs.

    This never fabricates a transformer result. With no checkpoint configured
    it returns the constructed command, the checkpoint report, and — because
    these *can* be validated without weights — the marker-gene coverage the
    tokenisation would achieve if a vocabulary is available.
    """
    from .models.scgpt import (
        ScgptUnavailable, build_finetune_command, checkpoint_report,
        load_vocabulary, map_to_vocabulary,
    )

    report = checkpoint_report(config.scgpt.checkpoint_dir)
    command = build_finetune_command(
        checkpoint_dir=config.scgpt.checkpoint_dir,
        data_path=Path(config.output_dir) / "scgpt_input.h5ad",
        output_dir=Path(config.output_dir) / "scgpt",
        n_classes=n_classes,
        epochs=config.scgpt.epochs,
        batch_size=config.scgpt.batch_size,
        learning_rate=config.scgpt.learning_rate,
        max_seq_len=config.scgpt.max_seq_len,
        n_bins=config.scgpt.n_bins,
        freeze_encoder=config.scgpt.freeze_encoder,
        dry_run=not report["available"],
    )

    stage: dict = {
        "status": (
            "IMPLEMENTED_NOT_FULLY_EXECUTED"
            if not report["available"] else "READY"
        ),
        "checkpoint": report,
        "finetune_command": command.to_dict(),
        "executed": False,
        "reason": None if report["available"] else (
            "no scGPT checkpoint is configured. The pretrained weights are "
            "distributed by the scGPT authors under their own terms and are not "
            "redistributed by this repository, and fine-tuning additionally "
            "requires a GPU. No transformer metric is reported."
        ),
    }

    if config.scgpt.vocabulary_path:
        try:
            vocabulary = load_vocabulary(config.scgpt.vocabulary_path)
        except ScgptUnavailable as error:
            stage["vocabulary_error"] = str(error)
        else:
            panel = [g for g in all_state_genes()]
            stage["vocabulary_coverage"] = map_to_vocabulary(
                gene_names, vocabulary, panel
            ).to_dict()
    return stage
