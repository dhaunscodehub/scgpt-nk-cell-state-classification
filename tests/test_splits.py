"""Splitting and the leakage guards.

These are the tests that matter most in this repository: every reported metric
is conditional on the split being what it claims to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.data.splits import (
    LeakageError, Split, SplitError, assert_no_donor_leakage, cell_split,
    class_balance, donor_kfold, donor_split,
)


def _donors(n_donors: int = 6, per_donor: int = 50) -> np.ndarray:
    return np.repeat([f"D{i}" for i in range(n_donors)], per_donor)


def test_donor_split_shares_no_donor():
    donors = _donors()
    split = donor_split(donors, test_fraction=0.25, val_fraction=0.25, seed=0)
    train = set(donors[np.asarray(split.train)].tolist())
    val = set(donors[np.asarray(split.val)].tolist())
    test = set(donors[np.asarray(split.test)].tolist())
    assert not train & test
    assert not train & val
    assert not val & test


def test_donor_split_covers_every_cell_exactly_once():
    donors = _donors()
    split = donor_split(donors, test_fraction=0.25, val_fraction=0.25, seed=0)
    allocated = sorted(list(split.train) + list(split.val) + list(split.test))
    assert allocated == list(range(donors.size))


def test_donor_split_is_deterministic_for_a_seed():
    donors = _donors()
    first = donor_split(donors, test_fraction=0.25, seed=7)
    second = donor_split(donors, test_fraction=0.25, seed=7)
    assert list(first.test) == list(second.test)


def test_donor_split_changes_with_the_seed():
    donors = _donors(n_donors=12)
    a = donor_split(donors, test_fraction=0.25, seed=0)
    b = donor_split(donors, test_fraction=0.25, seed=1)
    assert set(a.test_donors) != set(b.test_donors)


@pytest.mark.parametrize("n_donors", [1, 2])
def test_donor_split_refuses_too_few_donors(n_donors):
    """Fewer than three donors cannot make train/val/test groups of donors."""
    with pytest.raises(SplitError, match="at least 3 donors"):
        donor_split(_donors(n_donors=n_donors), test_fraction=0.25, seed=0)


def test_donor_split_refusal_names_the_donors():
    """The message has to be actionable, not just a refusal."""
    with pytest.raises(SplitError) as info:
        donor_split(np.array(["only"] * 50), test_fraction=0.25, seed=0)
    assert "only" in str(info.value)


def test_cell_split_shares_donors_and_records_it():
    """The leaking split must document its own leakage."""
    donors = _donors()
    split = cell_split(donors, test_fraction=0.25, seed=0)
    shared = set(donors[np.asarray(split.train)].tolist()) & set(
        donors[np.asarray(split.test)].tolist()
    )
    assert shared, "a cell-random split over 6 donors should share donors"
    assert split.extra["shared_donors_train_test"] == len(shared)


def test_cell_split_is_not_labelled_as_donor_grouped():
    split = cell_split(_donors(), test_fraction=0.25, seed=0)
    assert split.strategy != "donor"


def test_leakage_guard_accepts_a_clean_split():
    donors = _donors()
    split = donor_split(donors, test_fraction=0.25, seed=0)
    assert_no_donor_leakage(split, donors)  # must not raise


def test_leakage_guard_rejects_a_corrupted_split():
    """A guard that never fires is not evidence of safety."""
    donors = _donors()
    split = donor_split(donors, test_fraction=0.34, val_fraction=0.0, seed=0)
    corrupted = Split(
        name=split.name,
        train=list(split.train[1:]),
        val=list(split.val),
        test=list(split.test) + list(split.train[:1]),
        strategy=split.strategy,
        train_donors=list(split.train_donors),
        val_donors=list(split.val_donors),
        test_donors=list(split.test_donors),
    )
    with pytest.raises(LeakageError):
        assert_no_donor_leakage(corrupted, donors)


def test_leakage_guard_names_the_offending_donor():
    donors = _donors()
    split = donor_split(donors, test_fraction=0.34, val_fraction=0.0, seed=0)
    offender = donors[split.train[0]]
    corrupted = Split(
        name=split.name, train=list(split.train[1:]), val=list(split.val),
        test=list(split.test) + list(split.train[:1]), strategy=split.strategy,
        train_donors=list(split.train_donors), val_donors=list(split.val_donors),
        test_donors=list(split.test_donors),
    )
    with pytest.raises(LeakageError) as info:
        assert_no_donor_leakage(corrupted, donors)
    assert str(offender) in str(info.value)


def test_donor_kfold_holds_out_disjoint_donor_groups():
    donors = _donors(n_donors=6)
    folds = donor_kfold(donors, n_folds=3, seed=0)
    assert len(folds) == 3
    held_out = [set(f.test_donors) for f in folds]
    for index, donors_out in enumerate(held_out):
        assert donors_out, "every fold must hold out at least one donor"
        for other in held_out[index + 1 :]:
            assert not donors_out & other, "folds must hold out disjoint donors"


def test_donor_kfold_covers_every_donor():
    donors = _donors(n_donors=6)
    folds = donor_kfold(donors, n_folds=3, seed=0)
    covered = set()
    for fold in folds:
        covered |= set(fold.test_donors)
    assert covered == set(donors.tolist())


def test_donor_kfold_refuses_more_folds_than_donors():
    with pytest.raises(SplitError):
        donor_kfold(_donors(n_donors=3), n_folds=5, seed=0)


def test_class_balance_flags_a_class_absent_from_training():
    donors = _donors(n_donors=6, per_donor=10)
    labels = np.array(["a"] * 50 + ["rare"] * 10)
    split = Split(
        name="manual", train=list(range(50)), val=[], test=list(range(50, 60)),
        strategy="donor",
    )
    balance = class_balance(labels, split)
    # A class the model never saw gets zero recall for a reason that has
    # nothing to do with the data, so it has to be surfaced.
    assert balance["classes_in_test_but_not_train"] == ["rare"]


def test_class_balance_counts_each_fold():
    labels = np.array(["a", "a", "b", "b"])
    split = Split(name="m", train=[0, 2], val=[], test=[1, 3], strategy="donor")
    balance = class_balance(labels, split)
    assert balance["train_class_counts"] == {"a": 1, "b": 1}
    assert balance["test_class_counts"] == {"a": 1, "b": 1}
