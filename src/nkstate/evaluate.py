"""Classification metrics at cell level and at donor level.

Two levels, reported separately and never averaged together:

**Cell level** is the conventional number — accuracy and macro-F1 over cells.
It is also the optimistic one, because cells from one donor are not independent
observations. A model that has learned a donor's batch signature scores well on
cell-level metrics computed over a random split.

**Donor level** treats each held-out donor as one observation: the model is
evaluated per donor and the spread across donors is reported. This is the
number that predicts behaviour on the next donor, and it is usually
substantially worse. When it is much worse, that gap is the finding, not a
nuisance — so :func:`evaluate_predictions` always reports both and the
difference between them.

Macro-averaged F1 is the headline rather than accuracy: NK functional states
are imbalanced, and accuracy on a sample that is 70% one state rewards a model
that only ever predicts that state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Below this many cells a per-donor metric is too noisy to report as a point
# estimate; the donor is counted and excluded rather than silently averaged in.
MIN_CELLS_PER_DONOR = 10


class EvaluationError(ValueError):
    """Raised when predictions and labels cannot be compared."""


@dataclass
class ClassMetrics:
    """Per-class precision, recall, F1 and support."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict:
        return {
            "label": self.label, "precision": self.precision,
            "recall": self.recall, "f1": self.f1, "support": self.support,
        }


@dataclass
class CellMetrics:
    """Cell-level metrics over a set of predictions."""

    n_cells: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float
    per_class: list[ClassMetrics]
    confusion: list[list[int]]
    labels: list[str]
    macro_auroc: float | None = None
    chance_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "level": "cell", "n_cells": self.n_cells,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1, "weighted_f1": self.weighted_f1,
            "cohen_kappa": self.cohen_kappa, "macro_auroc": self.macro_auroc,
            "chance_accuracy": self.chance_accuracy,
            "labels": self.labels, "confusion": self.confusion,
            "per_class": [c.to_dict() for c in self.per_class],
        }


@dataclass
class DonorMetrics:
    """Donor-level metrics: one score per donor, then the spread."""

    n_donors: int
    per_donor: dict[str, dict]
    mean_macro_f1: float
    std_macro_f1: float
    min_macro_f1: float
    max_macro_f1: float
    mean_accuracy: float
    excluded_donors: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": "donor", "n_donors": self.n_donors,
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "min_macro_f1": self.min_macro_f1,
            "max_macro_f1": self.max_macro_f1,
            "mean_accuracy": self.mean_accuracy,
            "per_donor": self.per_donor,
            "excluded_donors_below_min_cells": self.excluded_donors,
            "min_cells_per_donor": MIN_CELLS_PER_DONOR,
        }


@dataclass
class EvaluationResult:
    """Cell- and donor-level metrics plus the gap between them."""

    model: str
    split_kind: str
    cell: CellMetrics
    donor: DonorMetrics | None = None
    label_type: str = "unspecified"
    caveats: list[str] = field(default_factory=list)

    @property
    def optimism(self) -> float | None:
        """Cell-level macro-F1 minus mean donor-level macro-F1.

        Positive means the cell-level number flatters the model relative to how
        it does on an average held-out donor. This is the quantity that a
        random cell-level split is designed not to reveal.
        """
        if self.donor is None:
            return None
        return self.cell.macro_f1 - self.donor.mean_macro_f1

    def to_dict(self) -> dict:
        return {
            "model": self.model, "split_kind": self.split_kind,
            "label_type": self.label_type,
            "cell_level": self.cell.to_dict(),
            "donor_level": self.donor.to_dict() if self.donor else None,
            "cell_minus_donor_macro_f1": self.optimism,
            "caveats": self.caveats,
        }


def _confusion(true: np.ndarray, predicted: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, predicted):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix


def cell_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    probabilities: np.ndarray | None = None,
    model_classes: Sequence[str] | None = None,
) -> CellMetrics:
    """Cell-level metrics.

    ``probabilities`` enables macro one-vs-rest AUROC, which needs the column
    order to match ``model_classes``. AUROC is reported as ``None`` rather than
    a made-up value when a class has no positive example in the evaluation set,
    since one-vs-rest AUROC is undefined for that class.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, cohen_kappa_score,
        f1_score, precision_recall_fscore_support, roc_auc_score,
    )

    true = np.asarray([str(l) for l in true_labels])
    predicted = np.asarray([str(l) for l in predicted_labels])
    if true.size != predicted.size:
        raise EvaluationError(
            f"{true.size} true labels but {predicted.size} predictions"
        )
    if true.size == 0:
        raise EvaluationError("no cells to evaluate")

    labels = sorted(set(true.tolist()) | set(predicted.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0,
    )
    counts = np.array([int((true == label).sum()) for label in labels])

    macro_auroc: float | None = None
    if probabilities is not None and model_classes is not None:
        macro_auroc = _macro_auroc(true, probabilities, model_classes)

    return CellMetrics(
        n_cells=int(true.size),
        accuracy=float(accuracy_score(true, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(true, predicted)),
        macro_f1=float(f1_score(true, predicted, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(true, predicted, average="weighted", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(true, predicted)),
        per_class=[
            ClassMetrics(labels[i], float(precision[i]), float(recall[i]),
                         float(f1[i]), int(support[i]))
            for i in range(len(labels))
        ],
        confusion=_confusion(true, predicted, labels).tolist(),
        labels=labels, macro_auroc=macro_auroc,
        # Chance for a classifier that always predicts the largest class.
        chance_accuracy=float(counts.max() / counts.sum()),
    )


def _macro_auroc(
    true: np.ndarray, probabilities: np.ndarray, model_classes: Sequence[str]
) -> float | None:
    """Macro one-vs-rest AUROC, or ``None`` if it is undefined here."""
    from sklearn.metrics import roc_auc_score

    classes = [str(c) for c in model_classes]
    if probabilities.shape != (true.size, len(classes)):
        raise EvaluationError(
            f"probabilities have shape {probabilities.shape}, expected "
            f"({true.size}, {len(classes)}) to match the model's classes"
        )
    scores = []
    for index, label in enumerate(classes):
        positive = (true == label).astype(int)
        # One-vs-rest AUROC needs both a positive and a negative example.
        if positive.sum() == 0 or positive.sum() == positive.size:
            return None
        scores.append(roc_auc_score(positive, probabilities[:, index]))
    return float(np.mean(scores)) if scores else None


def donor_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    donor_ids: Sequence[str],
) -> DonorMetrics:
    """Per-donor metrics and their spread across donors.

    Donors with fewer than :data:`MIN_CELLS_PER_DONOR` cells are excluded and
    listed: a macro-F1 over four cells is noise, and averaging it in would let
    the smallest donors dominate the variance estimate.
    """
    from sklearn.metrics import accuracy_score, f1_score

    true = np.asarray([str(l) for l in true_labels])
    predicted = np.asarray([str(l) for l in predicted_labels])
    donors = np.asarray([str(d) for d in donor_ids])
    if not (true.size == predicted.size == donors.size):
        raise EvaluationError(
            f"lengths differ: {true.size} labels, {predicted.size} predictions, "
            f"{donors.size} donor ids"
        )

    per_donor: dict[str, dict] = {}
    excluded: dict[str, int] = {}
    for donor in sorted(set(donors.tolist())):
        mask = donors == donor
        if int(mask.sum()) < MIN_CELLS_PER_DONOR:
            excluded[donor] = int(mask.sum())
            continue
        per_donor[donor] = {
            "n_cells": int(mask.sum()),
            "accuracy": float(accuracy_score(true[mask], predicted[mask])),
            "macro_f1": float(
                f1_score(true[mask], predicted[mask], average="macro", zero_division=0)
            ),
            "n_states_present": int(len(set(true[mask].tolist()))),
        }

    if not per_donor:
        raise EvaluationError(
            f"no donor has at least {MIN_CELLS_PER_DONOR} cells "
            f"(donor cell counts: {excluded}); donor-level metrics are not "
            "meaningful on this split"
        )

    f1_values = np.array([v["macro_f1"] for v in per_donor.values()])
    accuracies = np.array([v["accuracy"] for v in per_donor.values()])
    return DonorMetrics(
        n_donors=len(per_donor), per_donor=per_donor,
        mean_macro_f1=float(f1_values.mean()),
        # Population std: this describes the donors observed, and with 2-5
        # held-out donors the sample correction is a large, arbitrary inflation.
        std_macro_f1=float(f1_values.std()),
        min_macro_f1=float(f1_values.min()), max_macro_f1=float(f1_values.max()),
        mean_accuracy=float(accuracies.mean()),
        excluded_donors=excluded,
    )


def evaluate_predictions(
    model: str,
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    donor_ids: Sequence[str] | None = None,
    probabilities: np.ndarray | None = None,
    model_classes: Sequence[str] | None = None,
    split_kind: str = "unspecified",
    label_type: str = "unspecified",
) -> EvaluationResult:
    """Evaluate one model's predictions at both levels."""
    cell = cell_metrics(true_labels, predicted_labels, probabilities, model_classes)

    donor: DonorMetrics | None = None
    caveats: list[str] = []
    if donor_ids is not None:
        unique = sorted(set(str(d) for d in donor_ids))
        if len(unique) < 2:
            caveats.append(
                f"donor-level metrics need at least 2 held-out donors, found "
                f"{len(unique)} ({unique}); only cell-level metrics are reported "
                "and they cannot speak to generalisation across donors"
            )
        else:
            donor = donor_metrics(true_labels, predicted_labels, donor_ids)
    else:
        caveats.append("no donor ids supplied; donor-level metrics not computed")

    if label_type == "marker_pseudo_label":
        caveats.append(
            "labels are marker-panel pseudo-labels, not independent ground "
            "truth; a model scored against them is measured on agreement with "
            "the marker rule, which is an upper bound on its agreement with "
            "true cell state"
        )
    if split_kind == "cell_random":
        caveats.append(
            "cell-random split: cells from a donor appear in both train and "
            "test, so these numbers are optimistic by construction and are "
            "reported only for comparison against the donor-grouped split"
        )

    return EvaluationResult(
        model=model, split_kind=split_kind, cell=cell, donor=donor,
        label_type=label_type, caveats=caveats,
    )


def compare_models(results: Sequence[EvaluationResult]) -> dict:
    """Rank models and state whether any beat the marker-score floor.

    The comparison that matters is not which model wins but whether the learned
    models beat ``marker_score``, which does no fitting. If they do not, the
    learning added nothing over reading the markers directly, and saying so is
    the result.
    """
    if not results:
        raise EvaluationError("no results to compare")

    rows = [
        {
            "model": r.model, "split_kind": r.split_kind,
            "cell_macro_f1": r.cell.macro_f1,
            "cell_balanced_accuracy": r.cell.balanced_accuracy,
            "donor_mean_macro_f1": r.donor.mean_macro_f1 if r.donor else None,
            "cell_minus_donor": r.optimism,
        }
        for r in results
    ]
    rows.sort(key=lambda row: -row["cell_macro_f1"])

    floor = next((r for r in results if r.model == "marker_score"), None)
    beats_floor = None
    if floor is not None:
        beats_floor = {
            r.model: bool(r.cell.macro_f1 > floor.cell.macro_f1)
            for r in results if r.model != "marker_score"
        }

    return {
        "ranking": rows, "best_model": rows[0]["model"],
        "marker_score_floor_macro_f1": floor.cell.macro_f1 if floor else None,
        "beats_marker_score_floor": beats_floor,
        "note": (
            "marker_score does no fitting; a learned model that does not exceed "
            "it has not earned its complexity"
        ),
    }
