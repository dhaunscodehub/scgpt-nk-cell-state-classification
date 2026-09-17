"""Control experiments that establish what the metrics mean.

A macro-F1 on its own is uninterpretable: it does not say whether the model
learned biology, learned the donor, or was handed a task that could not be
failed. These controls pin the scale at both ends and demonstrate the specific
failure mode this repository is built around.

The controls:

1. **Negative control.** State labels carry no expression signal
   (``state_effect = 0``). A correct pipeline scores at chance. Anything above
   chance here means information is leaking from labels into features, and
   every other number in the repository would be suspect.

2. **Positive control.** Strong state signal. A correct pipeline scores near
   perfectly. Failure here means the pipeline cannot learn a signal that is
   plainly present, so a poor result elsewhere would be uninformative.

3. **Donor confound, cell-random split.** No biological state signal, but
   state composition depends on donor and donors differ by a batch effect.
   A cell-random split scores far above chance — by recognising the donor, not
   the state. This is the failure mode, reproduced on demand.

4. **Donor confound, donor-held-out split.** The same data, split by donor.
   The score collapses to chance, because the only usable signal was donor
   identity and it is no longer shared between folds. The contrast between
   experiments 3 and 4 is the quantitative argument for donor-grouped
   splitting.

5. **Single-donor refusal.** A donor-grouped split on one donor is refused
   rather than silently degenerating into a cell-level split.

6. **Marker pseudo-label recovery.** With a strong state effect, the marker
   panels recover the generating state. This checks the scoring implementation,
   not biology: the synthetic signal is planted on the panel genes, so recovery
   is expected by construction and is a test of the code.

7. **Leakage assertion.** The donor-split leakage check detects a deliberately
   corrupted split. A guard that never fires is not evidence of safety.
"""

from __future__ import annotations

import numpy as np

from .data.splits import LeakageError, Split, assert_no_donor_leakage, cell_split, donor_split
from .evaluate import cell_metrics
from .models.baselines import fit_baseline
from .testing import SyntheticSpec, build_preprocessed

# A control that should sit at chance is allowed this much above it, applied to
# the mean over replicate seeds. Wide enough to absorb sampling noise, narrow
# enough that the confounded cell-random split (~0.48 above chance) cannot hide.
CHANCE_TOLERANCE = 0.05
# A positive control must reach at least this balanced accuracy.
POSITIVE_CONTROL_FLOOR = 0.85
# The confounded experiments are replicated across seeds. A single draw with
# only a few held-out donors has large composition variance — measured at
# 0.185 to 0.368 balanced accuracy across seeds around a mean of 0.254 — so a
# single-seed assertion would be testing the draw rather than the claim.
CONFOUND_SEEDS = (0, 1, 2, 3, 4)
# The contrast the whole donor-grouped design rests on: a cell-random split
# must overstate balanced accuracy by at least this much on data whose only
# signal is donor identity.
MIN_CONFOUND_OPTIMISM = 0.25


def _fit_and_score(
    features: np.ndarray,
    labels: np.ndarray,
    gene_names: list[str],
    split: Split,
    model: str = "logistic",
    seed: int = 0,
) -> dict:
    """Fit on the training fold and score on the test fold."""
    train_index = np.asarray(split.train, dtype=int)
    test_index = np.asarray(split.test, dtype=int)
    fitted = fit_baseline(
        model, features[train_index], labels[train_index], gene_names, seed=seed,
    )
    predicted = fitted.predict(features[test_index])
    metrics = cell_metrics(labels[test_index], predicted)
    n_classes = len(set(labels.tolist()))
    return {
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "accuracy": metrics.accuracy,
        # Chance for balanced accuracy is 1/n_classes regardless of prevalence,
        # which is why balanced accuracy is the control's metric.
        "chance": 1.0 / n_classes,
        "n_train": len(split.train), "n_test": len(split.test),
        "n_classes": n_classes,
    }


def negative_control(seed: int = 0) -> dict:
    """State labels with no expression signal must score at chance."""
    spec = SyntheticSpec(
        n_donors=8, cells_per_donor=150, state_effect=0.0, donor_effect=0.0,
        seed=seed,
    )
    data = build_preprocessed(spec)
    split = donor_split(data.donors, test_fraction=0.25, val_fraction=0.0, seed=seed)
    scores = _fit_and_score(
        data.features, data.true_labels, data.gene_names, split, seed=seed,
    )
    excess = scores["balanced_accuracy"] - scores["chance"]
    return {
        "name": "negative control: no state signal, donor-grouped split",
        "expectation": f"balanced accuracy within {CHANCE_TOLERANCE} of chance",
        "passed": bool(abs(excess) <= CHANCE_TOLERANCE),
        "excess_over_chance": float(excess),
        "state_effect": spec.state_effect,
        "donor_confound": spec.donor_confound,
        "split_kind": "donor",
        **scores,
    }


def positive_control(seed: int = 0) -> dict:
    """A strong state signal must be learnable."""
    spec = SyntheticSpec(
        n_donors=8, cells_per_donor=150, state_effect=0.8, donor_effect=0.0,
        seed=seed,
    )
    data = build_preprocessed(spec)
    split = donor_split(data.donors, test_fraction=0.25, val_fraction=0.0, seed=seed)
    scores = _fit_and_score(
        data.features, data.true_labels, data.gene_names, split, seed=seed,
    )
    return {
        "name": "positive control: strong state signal, donor-grouped split",
        "expectation": f"balanced accuracy >= {POSITIVE_CONTROL_FLOOR}",
        "passed": bool(scores["balanced_accuracy"] >= POSITIVE_CONTROL_FLOOR),
        "excess_over_chance": float(scores["balanced_accuracy"] - scores["chance"]),
        "state_effect": spec.state_effect,
        "split_kind": "donor",
        **scores,
    }


def _confounded_spec(seed: int) -> SyntheticSpec:
    """A cohort whose only usable signal is donor identity.

    ``state_effect = 0`` removes all genuine biological signal, so a model can
    only do better than chance by identifying the donor and exploiting the fact
    that state composition depends on it. Twelve donors rather than eight so a
    0.25 test fraction holds out three, which keeps the per-draw composition
    variance manageable.
    """
    return SyntheticSpec(
        n_donors=12, cells_per_donor=150, state_effect=0.0, donor_effect=0.8,
        donor_confound=True, seed=seed,
    )


def _confound_replicate(seed: int, split_kind: str) -> dict:
    """One replicate of the confound experiment under one splitting strategy."""
    data = build_preprocessed(_confounded_spec(seed))
    if split_kind == "donor":
        split = donor_split(
            data.donors, test_fraction=0.25, val_fraction=0.0, seed=seed
        )
        # The claim of this arm is that no donor is shared; assert it rather
        # than assume the constructor got it right.
        assert_no_donor_leakage(split, data.donors)
        shared = 0
    else:
        split = cell_split(
            data.donors, test_fraction=0.25, val_fraction=0.0, seed=seed
        )
        shared = len(
            set(data.donors[np.asarray(split.train)].tolist())
            & set(data.donors[np.asarray(split.test)].tolist())
        )
    scores = _fit_and_score(
        data.features, data.true_labels, data.gene_names, split, seed=seed,
    )
    return {"seed": seed, "shared_donors_train_test": shared, **scores}


def _replicate_summary(replicates: list[dict]) -> dict:
    """Mean and range of balanced accuracy across replicates."""
    values = np.array([r["balanced_accuracy"] for r in replicates])
    chance = float(replicates[0]["chance"])
    return {
        "balanced_accuracy": float(values.mean()),
        "balanced_accuracy_sd": float(values.std()),
        "balanced_accuracy_min": float(values.min()),
        "balanced_accuracy_max": float(values.max()),
        "chance": chance,
        "excess_over_chance": float(values.mean() - chance),
        "n_replicates": len(replicates),
        "seeds": [r["seed"] for r in replicates],
        "per_seed_balanced_accuracy": [
            round(float(r["balanced_accuracy"]), 4) for r in replicates
        ],
    }


def donor_confound_cell_split(seeds: tuple[int, ...] = CONFOUND_SEEDS) -> dict:
    """The failure mode: a cell-random split exploits the donor confound."""
    replicates = [_confound_replicate(s, "cell_random") for s in seeds]
    summary = _replicate_summary(replicates)
    return {
        "name": "donor confound, cell-random split: scores above chance on no signal",
        "expectation": (
            "mean balanced accuracy far above chance despite state_effect = 0, "
            "because donor identity is shared across folds"
        ),
        "passed": bool(summary["excess_over_chance"] > 4 * CHANCE_TOLERANCE),
        "shared_donors_train_test": replicates[0]["shared_donors_train_test"],
        "state_effect": 0.0, "donor_confound": True,
        "split_kind": "cell_random",
        **summary,
    }


def donor_confound_donor_split(seeds: tuple[int, ...] = CONFOUND_SEEDS) -> dict:
    """The control: a donor-grouped split on the same data falls to chance.

    Asserted on the mean over replicates. A single draw holding out three of
    twelve donors has substantial composition variance, so an individual seed
    can land well above chance while the expectation is exactly chance;
    asserting per-seed would test the draw instead of the design.
    """
    replicates = [_confound_replicate(s, "donor") for s in seeds]
    summary = _replicate_summary(replicates)
    return {
        "name": "donor confound, donor-held-out split: falls back to chance",
        "expectation": (
            f"mean balanced accuracy within {CHANCE_TOLERANCE} of chance on the "
            "same data, because donor identity no longer transfers between folds"
        ),
        "passed": bool(abs(summary["excess_over_chance"]) <= CHANCE_TOLERANCE),
        "shared_donors_train_test": 0,
        "state_effect": 0.0, "donor_confound": True,
        "split_kind": "donor",
        **summary,
    }


def single_donor_refusal() -> dict:
    """A donor-grouped split on one donor must be refused, not approximated."""
    donors = np.array(["only_donor"] * 200)
    try:
        donor_split(donors, test_fraction=0.25, val_fraction=0.0, seed=0)
    except Exception as error:  # noqa: BLE001 - the type is the thing under test
        return {
            "name": "single-donor donor-grouped split is refused",
            "expectation": "raises rather than degenerating to a cell-level split",
            "passed": True,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    return {
        "name": "single-donor donor-grouped split is refused",
        "expectation": "raises rather than degenerating to a cell-level split",
        "passed": False,
        "message": (
            "a donor-grouped split on one donor returned a split instead of "
            "refusing; every donor-level metric downstream would be invalid"
        ),
    }


def marker_label_recovery(seed: int = 0) -> dict:
    """Marker pseudo-labels must recover the generating state.

    A test of the scoring code, not of biology: the synthetic state signal is
    planted on the marker-panel genes, so an implementation that works has to
    recover them. Reported as a code check to avoid it being read as evidence
    that marker scoring identifies real NK states.
    """
    from sklearn.metrics import adjusted_rand_score, f1_score

    from .scoring import assign_states, score_states

    spec = SyntheticSpec(
        n_donors=6, cells_per_donor=150, state_effect=0.8, donor_effect=0.0,
        seed=seed,
    )
    data = build_preprocessed(spec)
    assignment = assign_states(score_states(data.adata, states=spec.states))
    confident = assignment.confident_mask()
    predicted = assignment.labels[confident]
    truth = data.true_labels[confident]

    ari = float(adjusted_rand_score(truth, predicted))
    macro = float(f1_score(truth, predicted, average="macro", zero_division=0))
    return {
        "name": "marker pseudo-labels recover the generating state (code check)",
        "expectation": "ARI >= 0.9 on confidently-labelled cells",
        "passed": bool(ari >= 0.9),
        "adjusted_rand_index": ari, "macro_f1": macro,
        "n_confident": int(confident.sum()),
        "n_cells": int(assignment.labels.size),
        "fraction_ambiguous": float(assignment.fraction_ambiguous),
        "caveat": (
            "the synthetic signal is planted on the marker-panel genes, so "
            "recovery is expected by construction; this validates the scoring "
            "implementation and says nothing about real NK states"
        ),
    }


def leakage_guard_fires() -> dict:
    """The leakage assertion must reject a deliberately corrupted split."""
    donors = np.repeat([f"D{i}" for i in range(6)], 50)
    split = donor_split(donors, test_fraction=0.34, val_fraction=0.0, seed=0)

    # Move one training cell into the test fold. Its donor is now on both
    # sides, which is exactly the corruption the guard exists to catch.
    corrupted = Split(
        name=split.name,
        train=list(split.train[1:]),
        val=list(split.val),
        test=list(split.test) + list(split.train[:1]),
        strategy=split.strategy,
        train_donors=list(split.train_donors),
        val_donors=list(split.val_donors),
        test_donors=list(split.test_donors),
        extra=dict(split.extra),
    )
    try:
        assert_no_donor_leakage(corrupted, donors)
    except LeakageError as error:
        return {
            "name": "leakage guard rejects a corrupted donor split",
            "expectation": "raises LeakageError when a donor spans train and test",
            "passed": True,
            "message": str(error),
        }
    return {
        "name": "leakage guard rejects a corrupted donor split",
        "expectation": "raises LeakageError when a donor spans train and test",
        "passed": False,
        "message": (
            "the guard accepted a split in which a donor appears in both train "
            "and test; the leakage checks elsewhere cannot be relied on"
        ),
    }


def run_controls(seed: int = 0) -> dict:
    """Run every control experiment and summarise the outcome."""
    checks = [
        negative_control(seed),
        positive_control(seed),
        donor_confound_cell_split(),
        donor_confound_donor_split(),
        confound_optimism_is_large(),
        single_donor_refusal(),
        marker_label_recovery(seed),
        leakage_guard_fires(),
    ]

    cell_leak = next(c for c in checks if c.get("split_kind") == "cell_random")
    donor_clean = next(
        c for c in checks
        if c.get("split_kind") == "donor" and c.get("donor_confound")
    )
    return {
        "seed": seed,
        "confound_seeds": list(CONFOUND_SEEDS),
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        # The headline quantity: how much a cell-random split overstates
        # performance on data whose only signal is the donor.
        "confound_optimism_balanced_accuracy": float(
            cell_leak["balanced_accuracy"] - donor_clean["balanced_accuracy"]
        ),
        "interpretation": (
            "On a cohort with no biological state signal, averaged over "
            f"{len(CONFOUND_SEEDS)} seeds, a cell-random split reaches "
            f"{cell_leak['balanced_accuracy']:.3f} balanced accuracy while a "
            f"donor-held-out split reaches {donor_clean['balanced_accuracy']:.3f} "
            f"against a chance level of {donor_clean['chance']:.3f}. The "
            "difference is donor identity, not biology."
        ),
    }


def confound_optimism_is_large(
    seeds: tuple[int, ...] = CONFOUND_SEEDS,
) -> dict:
    """The contrast between the two splits must be large, not merely present.

    This is the assertion that justifies donor-grouped splitting: it is not
    enough that the donor-held-out score is lower, it has to be lower by a
    margin that would change a conclusion.
    """
    leaking = donor_confound_cell_split(seeds)
    clean = donor_confound_donor_split(seeds)
    optimism = leaking["balanced_accuracy"] - clean["balanced_accuracy"]
    return {
        "name": "cell-random splitting overstates accuracy by a large margin",
        "expectation": (
            f"cell-random minus donor-held-out balanced accuracy >= "
            f"{MIN_CONFOUND_OPTIMISM}"
        ),
        "passed": bool(optimism >= MIN_CONFOUND_OPTIMISM),
        "optimism_balanced_accuracy": float(optimism),
        "cell_random_balanced_accuracy": leaking["balanced_accuracy"],
        "donor_held_out_balanced_accuracy": clean["balanced_accuracy"],
        "chance": clean["chance"],
        "n_replicates": len(seeds),
    }
