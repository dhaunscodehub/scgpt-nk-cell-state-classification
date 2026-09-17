"""The control experiments themselves.

These tests assert that the controls behave as controls: the negative control
sits at chance, the positive control is learnable, and the donor confound is
exploited by a cell-random split but not by a donor-grouped one.
"""

from __future__ import annotations

import pytest

from nkstate.validation import (
    CHANCE_TOLERANCE, CONFOUND_SEEDS, MIN_CONFOUND_OPTIMISM,
    POSITIVE_CONTROL_FLOOR, confound_optimism_is_large,
    donor_confound_cell_split, donor_confound_donor_split, leakage_guard_fires,
    marker_label_recovery, negative_control, positive_control, run_controls,
    single_donor_refusal,
)


@pytest.fixture(scope="module")
def controls():
    return run_controls(seed=0)


def test_all_controls_pass(controls):
    failed = [c["name"] for c in controls["checks"] if not c["passed"]]
    assert not failed, f"failing controls: {failed}"


def test_every_control_states_its_expectation(controls):
    for check in controls["checks"]:
        assert check["expectation"], f"{check['name']} has no stated expectation"


def test_negative_control_sits_at_chance():
    """Above chance here would mean labels leak into features."""
    check = negative_control(seed=0)
    assert abs(check["excess_over_chance"]) <= CHANCE_TOLERANCE
    assert check["state_effect"] == 0.0


def test_positive_control_is_learnable():
    check = positive_control(seed=0)
    assert check["balanced_accuracy"] >= POSITIVE_CONTROL_FLOOR


def test_positive_and_negative_controls_bracket_the_scale():
    low = negative_control(seed=0)["balanced_accuracy"]
    high = positive_control(seed=0)["balanced_accuracy"]
    assert low < high


def test_cell_random_split_exploits_the_donor_confound():
    """The failure mode: accuracy above chance with no biological signal."""
    check = donor_confound_cell_split()
    assert check["state_effect"] == 0.0
    assert check["excess_over_chance"] > 4 * CHANCE_TOLERANCE
    assert check["shared_donors_train_test"] > 0


def test_donor_split_does_not_exploit_the_donor_confound():
    check = donor_confound_donor_split()
    assert abs(check["excess_over_chance"]) <= CHANCE_TOLERANCE
    assert check["shared_donors_train_test"] == 0


def test_confound_experiments_are_replicated():
    """A single draw with few held-out donors is too noisy to assert on."""
    check = donor_confound_donor_split()
    assert check["n_replicates"] == len(CONFOUND_SEEDS)
    assert len(check["per_seed_balanced_accuracy"]) == len(CONFOUND_SEEDS)


def test_confound_experiments_report_their_spread():
    check = donor_confound_donor_split()
    assert check["balanced_accuracy_min"] <= check["balanced_accuracy"]
    assert check["balanced_accuracy_max"] >= check["balanced_accuracy"]
    assert check["balanced_accuracy_sd"] >= 0.0


def test_optimism_from_the_confound_is_large():
    check = confound_optimism_is_large()
    assert check["optimism_balanced_accuracy"] >= MIN_CONFOUND_OPTIMISM
    assert (
        check["cell_random_balanced_accuracy"]
        > check["donor_held_out_balanced_accuracy"]
    )


def test_single_donor_split_is_refused():
    check = single_donor_refusal()
    assert check["passed"]
    assert check["error_type"] in {"SplitError", "ValueError"}


def test_leakage_guard_fires_on_a_corrupted_split():
    assert leakage_guard_fires()["passed"]


def test_marker_recovery_carries_its_own_caveat():
    """Recovery is expected by construction and must be labelled as such."""
    check = marker_label_recovery(seed=0)
    assert check["adjusted_rand_index"] >= 0.9
    assert "by construction" in check["caveat"]


def test_controls_summarise_the_headline_contrast(controls):
    assert controls["confound_optimism_balanced_accuracy"] > MIN_CONFOUND_OPTIMISM
    assert "donor identity" in controls["interpretation"]


def test_controls_are_reproducible():
    first = run_controls(seed=3)
    second = run_controls(seed=3)
    assert (
        first["confound_optimism_balanced_accuracy"]
        == second["confound_optimism_balanced_accuracy"]
    )
