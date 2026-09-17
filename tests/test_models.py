"""Baseline classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.data.splits import donor_split
from nkstate.models.baselines import (
    BASELINE_MODELS, MarkerScoreClassifier, ModelError, build_estimator,
    fit_baseline,
)


@pytest.fixture(scope="module")
def fold(strong_signal):
    split = donor_split(strong_signal.donors, test_fraction=0.25, val_fraction=0.0, seed=0)
    return (
        np.asarray(split.train, dtype=int), np.asarray(split.test, dtype=int),
    )


@pytest.mark.parametrize("name", BASELINE_MODELS)
def test_every_baseline_fits_and_predicts(name, strong_signal, fold):
    train, test = fold
    model = fit_baseline(
        name, strong_signal.features[train], strong_signal.true_labels[train],
        strong_signal.gene_names, seed=0,
    )
    predicted = model.predict(strong_signal.features[test])
    assert predicted.size == test.size
    assert set(predicted.tolist()) <= set(model.classes)


@pytest.mark.parametrize("name", BASELINE_MODELS)
def test_every_baseline_learns_the_positive_control(name, strong_signal, fold):
    from sklearn.metrics import balanced_accuracy_score

    train, test = fold
    model = fit_baseline(
        name, strong_signal.features[train], strong_signal.true_labels[train],
        strong_signal.gene_names, seed=0,
    )
    score = balanced_accuracy_score(
        strong_signal.true_labels[test], model.predict(strong_signal.features[test])
    )
    assert score > 0.8, f"{name} scored {score:.3f} on a separable task"


def test_unknown_model_is_rejected(strong_signal):
    with pytest.raises(ModelError, match="unknown model"):
        build_estimator("magic", strong_signal.gene_names, ["a", "b"])


def test_mismatched_label_count_is_rejected(strong_signal):
    with pytest.raises(ModelError, match="labels"):
        fit_baseline(
            "logistic", strong_signal.features, strong_signal.true_labels[:10],
            strong_signal.gene_names,
        )


def test_mismatched_gene_count_is_rejected(strong_signal):
    with pytest.raises(ModelError, match="genes"):
        fit_baseline(
            "logistic", strong_signal.features, strong_signal.true_labels,
            strong_signal.gene_names[:10],
        )


def test_single_class_is_rejected(strong_signal):
    """A one-class problem is not a classification problem."""
    labels = np.array(["only"] * strong_signal.features.shape[0])
    with pytest.raises(ModelError, match="one class"):
        fit_baseline(
            "logistic", strong_signal.features, labels, strong_signal.gene_names
        )


def test_too_few_cells_is_rejected(strong_signal):
    with pytest.raises(ModelError, match="too few"):
        fit_baseline(
            "logistic", strong_signal.features[:8], strong_signal.true_labels[:8],
            strong_signal.gene_names,
        )


def test_logistic_gene_importance_is_per_class(strong_signal, fold):
    train, _ = fold
    model = fit_baseline(
        "logistic", strong_signal.features[train], strong_signal.true_labels[train],
        strong_signal.gene_names, seed=0,
    )
    importance = model.gene_importance(top_n=10)
    assert set(importance) == set(model.classes)
    for genes in importance.values():
        assert len(genes) == 10
        # Descending by signed coefficient.
        assert [w for _, w in genes] == sorted([w for _, w in genes], reverse=True)


def test_random_forest_importance_is_not_attributed_per_class(strong_signal, fold):
    """Tree importances are unsigned and shared, so they must not be split up."""
    train, _ = fold
    model = fit_baseline(
        "random_forest", strong_signal.features[train],
        strong_signal.true_labels[train], strong_signal.gene_names, seed=0,
    )
    importance = model.gene_importance(top_n=5)
    assert set(importance) == {"all_classes"}


def test_marker_baseline_ignores_training_expression(strong_signal, fold):
    """The floor baseline must not learn; its predictions cannot depend on fitting."""
    train, test = fold
    genes = strong_signal.gene_names
    labels = strong_signal.true_labels

    real = fit_baseline(
        "marker_score", strong_signal.features[train], labels[train], genes,
    )
    shuffled = strong_signal.features[train].copy()
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled)
    scrambled = fit_baseline("marker_score", shuffled, labels[train], genes)

    assert np.array_equal(
        real.predict(strong_signal.features[test]),
        scrambled.predict(strong_signal.features[test]),
    )


def test_marker_baseline_refuses_a_class_with_no_panel(strong_signal, fold):
    train, _ = fold
    labels = strong_signal.true_labels[train].copy()
    labels[:50] = "erythrocyte"
    with pytest.raises(ModelError, match="no marker panel genes"):
        fit_baseline(
            "marker_score", strong_signal.features[train], labels,
            strong_signal.gene_names,
        )


def test_marker_baseline_probabilities_sum_to_one(strong_signal, fold):
    train, test = fold
    model = fit_baseline(
        "marker_score", strong_signal.features[train],
        strong_signal.true_labels[train], strong_signal.gene_names,
    )
    probabilities = model.predict_proba(strong_signal.features[test])
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_mlp_has_no_probability_free_lunch(strong_signal, fold):
    """Every model that claims probabilities must produce a valid simplex."""
    train, test = fold
    for name in ("logistic", "random_forest", "mlp"):
        model = fit_baseline(
            name, strong_signal.features[train], strong_signal.true_labels[train],
            strong_signal.gene_names, seed=0,
        )
        probabilities = model.predict_proba(strong_signal.features[test])
        assert probabilities.shape == (test.size, len(model.classes))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert float(probabilities.min()) >= 0.0


def test_fits_are_deterministic_for_a_seed(strong_signal, fold):
    train, test = fold
    predictions = [
        fit_baseline(
            "random_forest", strong_signal.features[train],
            strong_signal.true_labels[train], strong_signal.gene_names, seed=3,
        ).predict(strong_signal.features[test])
        for _ in range(2)
    ]
    assert np.array_equal(predictions[0], predictions[1])


def test_to_dict_records_the_training_size(strong_signal, fold):
    train, _ = fold
    model = fit_baseline(
        "logistic", strong_signal.features[train], strong_signal.true_labels[train],
        strong_signal.gene_names, seed=0,
    )
    record = model.to_dict()
    assert record["n_train_cells"] == train.size
    assert record["n_features"] == len(strong_signal.gene_names)


def test_marker_classifier_requires_known_classes(strong_signal):
    with pytest.raises(ModelError):
        MarkerScoreClassifier(strong_signal.gene_names, ["nonsense_class"])
