"""Donor-grouped splits for single-cell classification.

A single-cell dataset has one obvious and one subtle unit of replication. The
obvious one is the cell; the correct one is the **donor**. Cells from one donor
share that donor's genotype, its ambient RNA background, its dissociation
stress response and its sequencing batch. A cell-level random split therefore
puts thousands of cells from the same donor on both sides of the evaluation
boundary, and a classifier scores well by recognising the donor.

Every split here is built over donors and asserted:
:func:`assert_no_donor_leakage` verifies that no donor — and therefore no cell
— appears in two folds, and it runs inside the split constructors so a caller
cannot forget it.

A single-donor dataset cannot support this. :func:`donor_split` refuses rather
than silently degenerating into a cell-level split, which is the failure mode
that makes a published single-donor result look like a validated one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class SplitError(ValueError):
    """Raised when a split cannot be constructed as requested."""


class LeakageError(AssertionError):
    """Raised when a split shares a donor or a cell between folds."""


@dataclass
class Split:
    """Cell indices per fold, plus the donors behind them."""

    name: str
    train: list[int]
    val: list[int]
    test: list[int]
    strategy: str
    train_donors: list[str] = field(default_factory=list)
    val_donors: list[str] = field(default_factory=list)
    test_donors: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "name": self.name, "strategy": self.strategy,
            "n_train_cells": len(self.train), "n_val_cells": len(self.val),
            "n_test_cells": len(self.test),
            "n_train_donors": len(self.train_donors),
            "n_val_donors": len(self.val_donors),
            "n_test_donors": len(self.test_donors),
            "train_donors": self.train_donors, "val_donors": self.val_donors,
            "test_donors": self.test_donors, **self.extra,
        }


def assert_no_donor_leakage(split: Split, donors: Sequence[str] | None = None) -> None:
    """Verify that no donor, and no cell, appears in two folds."""
    cells = {"train": set(split.train), "val": set(split.val), "test": set(split.test)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = cells[a] & cells[b]
        if shared:
            raise LeakageError(
                f"{split.name}: {len(shared)} cell(s) appear in both {a} and {b}"
            )

    donor_sets = {
        "train": set(split.train_donors), "val": set(split.val_donors),
        "test": set(split.test_donors),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = donor_sets[a] & donor_sets[b]
        if shared:
            raise LeakageError(
                f"{split.name}: donor(s) {sorted(shared)[:3]} appear in both {a} and "
                f"{b}. Cells from one donor share genotype, ambient RNA and "
                "sequencing batch, so this split measures donor recognition rather "
                "than biology."
            )

    if donors is None:
        return
    # Every cell's donor must be in its fold's donor list, which catches a
    # hand-assembled split whose index and donor lists disagree.
    donors = np.asarray(donors)
    for fold, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
        for index in indices:
            if donors[index] not in donor_sets[fold]:
                raise LeakageError(
                    f"{split.name}: cell {index} is in the {fold} fold but its donor "
                    f"{donors[index]!r} is not in the {fold} donor list"
                )


def donor_split(
    donors: Sequence[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 0,
    name: str = "donor_holdout",
) -> Split:
    """Hold out whole donors.

    Requires at least three donors. With fewer, the split cannot separate
    donors into three folds and the function refuses rather than falling back
    to a cell-level split.
    """
    donors = np.asarray([str(d) for d in donors])
    unique = sorted(set(donors.tolist()))
    if len(unique) < 3:
        raise SplitError(
            f"a donor-grouped split needs at least 3 donors, found {len(unique)} "
            f"({unique}). A single- or two-donor dataset cannot support one: "
            "splitting its cells at random would measure donor recognition, which "
            "is exactly what this split exists to prevent. Use synthetic "
            "multi-donor data to validate the pipeline, and report a single-donor "
            "result as single-donor."
        )
    if not 0 < val_fraction + test_fraction < 1:
        raise SplitError("val_fraction + test_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    shuffled = [str(d) for d in rng.permutation(unique)]
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    if n_test + n_val >= len(shuffled):
        n_test, n_val = 1, 1
    test_donors = sorted(shuffled[:n_test])
    val_donors = sorted(shuffled[n_test : n_test + n_val])
    train_donors = sorted(shuffled[n_test + n_val :])
    if not train_donors:
        raise SplitError("the requested fractions leave no training donors")

    def indices(selected: list[str]) -> list[int]:
        wanted = set(selected)
        return [int(i) for i, d in enumerate(donors) if d in wanted]

    split = Split(
        name=name, train=indices(train_donors), val=indices(val_donors),
        test=indices(test_donors), strategy="donor",
        train_donors=train_donors, val_donors=val_donors, test_donors=test_donors,
        extra={"n_donors_total": len(unique)},
    )
    assert_no_donor_leakage(split, donors)
    return split


def cell_split(
    donors: Sequence[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 0,
    name: str = "cell_random",
) -> Split:
    """Uniform random split over cells.

    Provided **only** for the comparison that demonstrates why it is wrong. It
    shares donors between folds by construction, so
    :func:`assert_no_donor_leakage` is not applied to it and its reported
    performance is optimistic. Never use it to report a result.
    """
    donors = np.asarray([str(d) for d in donors])
    n = donors.size
    if n < 10:
        raise SplitError(f"{n} cells is too few to split")
    if not 0 < val_fraction + test_fraction < 1:
        raise SplitError("val_fraction + test_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test = sorted(int(i) for i in order[:n_test])
    val = sorted(int(i) for i in order[n_test : n_test + n_val])
    train = sorted(int(i) for i in order[n_test + n_val :])

    shared = len(set(donors[train].tolist()) & set(donors[test].tolist()))
    return Split(
        name=name, train=train, val=val, test=test, strategy="cell_random",
        train_donors=sorted(set(donors[train].tolist())),
        val_donors=sorted(set(donors[val].tolist())),
        test_donors=sorted(set(donors[test].tolist())),
        extra={
            "seed": seed, "n_donors_total": len(set(donors.tolist())),
            "shared_donors_train_test": shared,
            "warning": "donors are shared between folds; this split is optimistic",
        },
    )


def donor_kfold(
    donors: Sequence[str], n_folds: int = 5, val_fraction: float = 0.15, seed: int = 0
) -> list[Split]:
    """Donor-grouped k-fold cross-validation; every donor tested exactly once."""
    donors = np.asarray([str(d) for d in donors])
    unique = sorted(set(donors.tolist()))
    if n_folds < 2:
        raise SplitError("n_folds must be >= 2")
    if len(unique) < n_folds:
        raise SplitError(
            f"{len(unique)} donors cannot be divided into {n_folds} folds"
        )

    rng = np.random.default_rng(seed)
    shuffled = [str(d) for d in rng.permutation(unique)]
    groups = [shuffled[i::n_folds] for i in range(n_folds)]

    def indices(selected: list[str]) -> list[int]:
        wanted = set(selected)
        return [int(i) for i, d in enumerate(donors) if d in wanted]

    splits: list[Split] = []
    for index, test_donors in enumerate(groups):
        remaining = [d for d in shuffled if d not in set(test_donors)]
        n_val = max(1, int(round(len(remaining) * val_fraction)))
        val_donors, train_donors = remaining[:n_val], remaining[n_val:]
        if not train_donors:
            raise SplitError(f"fold {index + 1} has no training donors")
        split = Split(
            name=f"fold{index + 1}", train=indices(train_donors),
            val=indices(val_donors), test=indices(test_donors),
            strategy=f"donor_kfold({n_folds})",
            train_donors=sorted(train_donors), val_donors=sorted(val_donors),
            test_donors=sorted(test_donors),
            extra={"n_donors_total": len(unique)},
        )
        assert_no_donor_leakage(split, donors)
        splits.append(split)

    tested = Counter(d for s in splits for d in s.test_donors)
    if set(tested.values()) != {1} or set(tested) != set(unique):
        raise LeakageError(
            "k-fold construction is not a partition over donors: each donor must "
            "be tested exactly once"
        )
    return splits


def split_from_adata(
    adata, strategy: str = "donor", donor_key: str = "donor_id", **kwargs
) -> Split:
    """Build a split from an AnnData object's donor column."""
    if donor_key not in adata.obs:
        raise SplitError(
            f"obs[{donor_key!r}] not found; a donor-grouped split needs a donor "
            f"column. Available columns: {list(adata.obs.columns)}"
        )
    donors = adata.obs[donor_key].astype(str).to_numpy()
    if strategy == "donor":
        return donor_split(donors, **kwargs)
    if strategy == "cell_random":
        return cell_split(donors, **kwargs)
    raise SplitError(
        f"unknown strategy {strategy!r}; use 'donor' or, only for the "
        "demonstration comparison, 'cell_random'"
    )


def class_balance(labels: Sequence[str], split: Split) -> dict:
    """Class counts per fold, so an absent class is visible before training."""
    labels = np.asarray([str(l) for l in labels])
    out: dict[str, object] = {}
    for fold, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
        counts = Counter(labels[indices].tolist()) if indices else Counter()
        out[f"{fold}_class_counts"] = dict(sorted(counts.items()))
        out[f"{fold}_n_classes"] = len(counts)
    train_classes = set(labels[split.train].tolist()) if split.train else set()
    test_classes = set(labels[split.test].tolist()) if split.test else set()
    out["classes_in_test_but_not_train"] = sorted(test_classes - train_classes)
    out["classes_in_train_but_not_test"] = sorted(train_classes - test_classes)
    return out
