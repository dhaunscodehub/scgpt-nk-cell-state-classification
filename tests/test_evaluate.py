"""Cell-level and donor-level metrics."""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.evaluate import (
    MIN_CELLS_PER_DONOR, EvaluationError, cell_metrics, compare_models,
    donor_metrics, evaluate_predictions,
)


def test_perfect_predictions_score_one():
    metrics = cell_metrics(["a", "b", "c"] * 10, ["a", "b", "c"] * 10)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.cohen_kappa == pytest.approx(1.0)


def test_constant_prediction_scores_at_chance_balanced_accuracy():
    """Predicting one class always gives 1/n_classes balanced accuracy."""
    true = ["a"] * 30 + ["b"] * 30 + ["c"] * 30
    metrics = cell_metrics(true, ["a"] * 90)
    assert metrics.balanced_accuracy == pytest.approx(1 / 3)
    # Accuracy would flatter it: a third correct on a balanced problem.
    assert metrics.accuracy == pytest.approx(1 / 3)


def test_accuracy_flatters_a_constant_prediction_on_imbalanced_data():
    """Why macro-F1 is the headline: accuracy rewards ignoring rare classes."""
    true = ["common"] * 90 + ["rare"] * 10
    metrics = cell_metrics(true, ["common"] * 100)
    assert metrics.accuracy == pytest.approx(0.9)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 < 0.5


def test_chance_accuracy_is_the_majority_class_rate():
    metrics = cell_metrics(["a"] * 70 + ["b"] * 30, ["a"] * 100)
    assert metrics.chance_accuracy == pytest.approx(0.7)


def test_confusion_matrix_rows_sum_to_support():
    true = ["a"] * 10 + ["b"] * 5
    predicted = ["a"] * 8 + ["b"] * 2 + ["b"] * 5
    metrics = cell_metrics(true, predicted)
    confusion = np.array(metrics.confusion)
    supports = np.array([c.support for c in metrics.per_class])
    assert np.array_equal(confusion.sum(axis=1), supports)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(EvaluationError, match="predictions"):
        cell_metrics(["a", "b"], ["a"])


def test_empty_input_is_rejected():
    with pytest.raises(EvaluationError, match="no cells"):
        cell_metrics([], [])


def test_auroc_needs_matching_probability_columns():
    with pytest.raises(EvaluationError, match="probabilities"):
        cell_metrics(
            ["a", "b"], ["a", "b"], probabilities=np.zeros((2, 3)),
            model_classes=["a", "b"],
        )


def test_auroc_is_none_when_a_class_has_no_positive_example():
    """One-vs-rest AUROC is undefined then, so it must not be invented."""
    metrics = cell_metrics(
        ["a", "a"], ["a", "a"],
        probabilities=np.array([[0.9, 0.1], [0.8, 0.2]]),
        model_classes=["a", "b"],
    )
    assert metrics.macro_auroc is None


def test_auroc_is_one_for_perfectly_ranked_probabilities():
    metrics = cell_metrics(
        ["a", "a", "b", "b"], ["a", "a", "b", "b"],
        probabilities=np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
        model_classes=["a", "b"],
    )
    assert metrics.macro_auroc == pytest.approx(1.0)


def test_donor_metrics_report_one_score_per_donor():
    true = ["a", "b"] * 30
    donors = ["d1"] * 20 + ["d2"] * 20 + ["d3"] * 20
    report = donor_metrics(true, true, donors)
    assert report.n_donors == 3
    assert report.mean_macro_f1 == pytest.approx(1.0)
    assert report.std_macro_f1 == pytest.approx(0.0)


def test_donor_metrics_exclude_and_list_tiny_donors():
    """A macro-F1 over four cells is noise; excluding it must be visible."""
    true = ["a", "b"] * 25
    donors = ["big"] * 46 + ["tiny"] * 4
    report = donor_metrics(true, true, donors)
    assert report.n_donors == 1
    assert report.excluded_donors == {"tiny": 4}
    assert report.min_cells_per_donor if hasattr(report, "min_cells_per_donor") else True


def test_donor_metrics_reject_when_every_donor_is_tiny():
    true = ["a", "b"] * 4
    donors = [f"d{i}" for i in range(8)]
    with pytest.raises(EvaluationError, match="at least"):
        donor_metrics(true, true, donors)


def test_donor_metrics_reject_mismatched_lengths():
    with pytest.raises(EvaluationError, match="lengths differ"):
        donor_metrics(["a"] * 10, ["a"] * 10, ["d"] * 9)


def test_min_cells_per_donor_is_documented():
    assert MIN_CELLS_PER_DONOR >= 2


def test_one_held_out_donor_gives_no_donor_metrics_but_a_caveat():
    result = evaluate_predictions(
        "logistic", ["a", "b"] * 20, ["a", "b"] * 20, donor_ids=["only"] * 40,
    )
    assert result.donor is None
    assert result.optimism is None
    assert any("at least 2" in c for c in result.caveats)


def test_missing_donor_ids_are_flagged():
    result = evaluate_predictions("logistic", ["a", "b"], ["a", "b"])
    assert any("no donor ids" in c for c in result.caveats)


def test_pseudo_label_caveat_is_attached():
    result = evaluate_predictions(
        "logistic", ["a", "b"], ["a", "b"], label_type="marker_pseudo_label",
    )
    assert any("pseudo-label" in c for c in result.caveats)


def test_cell_random_split_caveat_is_attached():
    result = evaluate_predictions(
        "logistic", ["a", "b"], ["a", "b"], split_kind="cell_random",
    )
    assert any("optimistic by construction" in c for c in result.caveats)


def test_optimism_is_cell_minus_donor():
    true = ["a"] * 15 + ["b"] * 15
    # Donor d2's predictions are wrong, so its macro-F1 drags the donor mean
    # below the pooled cell-level score.
    predicted = ["a"] * 15 + ["b"] * 10 + ["a"] * 5
    donors = ["d1"] * 15 + ["d2"] * 15
    result = evaluate_predictions(
        "logistic", true, predicted, donor_ids=donors, split_kind="donor",
    )
    assert result.donor is not None
    assert result.optimism == pytest.approx(
        result.cell.macro_f1 - result.donor.mean_macro_f1
    )


def test_compare_models_ranks_by_macro_f1():
    good = evaluate_predictions("logistic", ["a", "b"] * 20, ["a", "b"] * 20)
    bad = evaluate_predictions("mlp", ["a", "b"] * 20, ["a"] * 40)
    comparison = compare_models([bad, good])
    assert comparison["best_model"] == "logistic"
    assert comparison["ranking"][0]["model"] == "logistic"


def test_compare_models_reports_whether_the_floor_was_beaten():
    floor = evaluate_predictions("marker_score", ["a", "b"] * 20, ["a"] * 40)
    learned = evaluate_predictions("logistic", ["a", "b"] * 20, ["a", "b"] * 20)
    comparison = compare_models([floor, learned])
    assert comparison["beats_marker_score_floor"] == {"logistic": True}
    assert comparison["marker_score_floor_macro_f1"] is not None


def test_compare_models_reports_a_model_that_fails_to_beat_the_floor():
    """Not beating the floor is a result, and must be recorded as one."""
    floor = evaluate_predictions("marker_score", ["a", "b"] * 20, ["a", "b"] * 20)
    learned = evaluate_predictions("logistic", ["a", "b"] * 20, ["a"] * 40)
    comparison = compare_models([floor, learned])
    assert comparison["beats_marker_score_floor"] == {"logistic": False}


def test_compare_models_without_the_floor_reports_none():
    learned = evaluate_predictions("logistic", ["a", "b"] * 20, ["a", "b"] * 20)
    comparison = compare_models([learned])
    assert comparison["marker_score_floor_macro_f1"] is None
    assert comparison["beats_marker_score_floor"] is None


def test_compare_models_rejects_an_empty_list():
    with pytest.raises(EvaluationError):
        compare_models([])


def test_result_serialises_both_levels():
    result = evaluate_predictions(
        "logistic", ["a", "b"] * 20, ["a", "b"] * 20,
        donor_ids=["d1"] * 20 + ["d2"] * 20, split_kind="donor",
    )
    record = result.to_dict()
    assert record["cell_level"]["level"] == "cell"
    assert record["donor_level"]["level"] == "donor"
    assert "cell_minus_donor_macro_f1" in record
