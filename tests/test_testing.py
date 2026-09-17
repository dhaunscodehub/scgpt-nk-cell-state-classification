"""The synthetic generator: the knobs must actually do what they claim."""

from __future__ import annotations

import numpy as np
import pytest

from nkstate.testing import SyntheticSpec, build_preprocessed, build_synthetic


def test_shape_matches_the_spec():
    data = build_synthetic(SyntheticSpec(n_donors=5, cells_per_donor=40, seed=0))
    assert data.adata.n_obs == 200
    assert len(set(data.donors.tolist())) == 5


def test_counts_are_non_negative_integers():
    data = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=0))
    matrix = data.adata.X
    counts = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    assert float(counts.min()) >= 0
    assert np.allclose(counts, np.round(counts))


def test_generation_is_deterministic_for_a_seed():
    a = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=11))
    b = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=11))
    assert np.array_equal(
        np.asarray(a.adata.X, dtype=float), np.asarray(b.adata.X, dtype=float)
    )


def test_generation_changes_with_the_seed():
    a = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=1))
    b = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=2))
    assert not np.array_equal(
        np.asarray(a.adata.X, dtype=float), np.asarray(b.adata.X, dtype=float)
    )


def test_state_effect_controls_separability():
    """The knob must change the thing it names, or the controls prove nothing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    scores = {}
    for effect in (0.0, 0.8):
        data = build_preprocessed(
            SyntheticSpec(
                n_donors=4, cells_per_donor=100, state_effect=effect, seed=0
            )
        )
        train_x, test_x, train_y, test_y = train_test_split(
            data.features, data.true_labels, test_size=0.3, random_state=0
        )
        model = LogisticRegression(max_iter=1000).fit(train_x, train_y)
        scores[effect] = balanced_accuracy_score(test_y, model.predict(test_x))

    assert scores[0.0] < 0.4, f"no-signal cohort was separable ({scores[0.0]:.3f})"
    assert scores[0.8] > 0.8, f"strong-signal cohort was not ({scores[0.8]:.3f})"


def test_donor_effect_creates_donor_structure():
    """A donor effect must make donors distinguishable from expression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    scores = {}
    for effect in (0.0, 0.8):
        data = build_preprocessed(
            SyntheticSpec(
                n_donors=4, cells_per_donor=100, state_effect=0.0,
                donor_effect=effect, seed=0,
            )
        )
        train_x, test_x, train_y, test_y = train_test_split(
            data.features, data.donors, test_size=0.3, random_state=0
        )
        model = LogisticRegression(max_iter=1000).fit(train_x, train_y)
        scores[effect] = balanced_accuracy_score(test_y, model.predict(test_x))

    assert scores[0.8] > scores[0.0]
    assert scores[0.8] > 0.5, "donor_effect did not make donors identifiable"


def test_donor_confound_ties_composition_to_donor():
    data = build_synthetic(
        SyntheticSpec(
            n_donors=6, cells_per_donor=100, state_effect=0.0,
            donor_confound=True, seed=0,
        )
    )
    import pandas as pd

    table = pd.crosstab(data.donors, data.true_labels)
    fractions = table.div(table.sum(axis=1), axis=0)
    # Composition must differ between donors, otherwise there is no confound.
    assert float(fractions.max(axis=0).max() - fractions.min(axis=0).min()) > 0.2


def test_without_the_confound_composition_is_even():
    data = build_synthetic(
        SyntheticSpec(
            n_donors=6, cells_per_donor=100, state_effect=0.0,
            donor_confound=False, seed=0,
        )
    )
    import pandas as pd

    table = pd.crosstab(data.donors, data.true_labels)
    fractions = table.div(table.sum(axis=1), axis=0)
    assert float(fractions.std().max()) < 0.15


def test_marker_genes_are_present_in_the_generated_dataset():
    data = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=0))
    assert data.marker_genes
    assert set(data.marker_genes) <= set(map(str, data.adata.var_names))


def test_differential_genes_are_recorded_per_state():
    data = build_synthetic(SyntheticSpec(n_donors=3, cells_per_donor=30, seed=0))
    assert set(data.differential_genes) == set(data.spec.states)


def test_preprocessed_labels_stay_aligned_after_filtering():
    """Quality control drops cells; carried-over arrays would misalign."""
    data = build_preprocessed(
        SyntheticSpec(n_donors=4, cells_per_donor=60, seed=0)
    )
    assert data.true_labels.size == data.adata.n_obs
    assert data.donors.size == data.adata.n_obs
    assert np.array_equal(
        data.true_labels, np.asarray([str(v) for v in data.adata.obs["true_state"]])
    )


def test_preprocessed_features_are_log_normalised():
    data = build_preprocessed(SyntheticSpec(n_donors=3, cells_per_donor=40, seed=0))
    assert float(data.features.min()) >= 0.0
    # log1p of CPM values stays well under raw count magnitudes.
    assert float(data.features.max()) < 20.0


def test_summary_records_the_knobs():
    data = build_synthetic(
        SyntheticSpec(n_donors=3, cells_per_donor=30, state_effect=0.5, seed=0)
    )
    summary = data.summary()
    assert summary["state_effect"] == 0.5
    assert summary["n_donors"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_donors": 0},
        {"cells_per_donor": 2},
        {"state_effect": -0.1},
        {"state_effect": 1.1},
        {"donor_effect": 2.0},
        {"states": ("nonsense",)},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SyntheticSpec(**kwargs)
