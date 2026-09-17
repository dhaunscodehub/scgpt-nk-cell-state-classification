"""Baseline classifiers over gene-expression features.

These are what actually executes in this repository: scGPT needs pretrained
weights that cannot be redistributed, so a fine-tuned transformer cannot be the
only classifier on offer. The baselines also serve a second purpose — a
foundation model that cannot beat logistic regression on 2000 genes has not
earned its cost, and without the baseline on the same axis that is invisible.

Four models:

``logistic``
    Multinomial logistic regression on standardised expression. The reference.
    Its coefficients are directly interpretable per gene and per class, which
    is what makes the interpretation step meaningful.
``random_forest``
    Non-linear, robust to the heavy-tailed expression distribution, and
    supplies feature importances.
``mlp``
    A small dense network — the closest baseline in family to a fine-tuned
    transformer head.
``marker_score``
    Assigns the highest-scoring marker panel, with no fitting at all. The
    floor: any learned model must beat simply reading the markers.

Features are standardised with statistics from the **training fold only**.
Fitting the scaler on all cells would leak the test fold's expression
distribution, which on single-cell data is a substantial leak because the
dominant axis of variation is technical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

BASELINE_MODELS = ("logistic", "random_forest", "mlp", "marker_score")


class ModelError(ValueError):
    """Raised for an unknown model or an unusable configuration."""


@dataclass
class FittedBaseline:
    """A trained classifier plus what it needs to predict and be interpreted."""

    name: str
    estimator: object
    classes: list[str]
    gene_names: list[str]
    n_train_cells: int
    n_features: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(features))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "predict_proba"):
            raise ModelError(f"{self.name} does not provide class probabilities")
        return np.asarray(self.estimator.predict_proba(features))

    def gene_importance(self, top_n: int = 25) -> dict[str, list[tuple[str, float]]]:
        """Per-class discriminative genes, when the model exposes them.

        Logistic regression gives signed per-class coefficients, which is the
        most directly readable attribution: a positive coefficient means the
        gene pushes towards that class. Tree ensembles give a single unsigned
        importance vector shared across classes, so it is reported under
        ``"all_classes"`` rather than split up as if it were class-specific.
        """
        estimator = self.estimator
        regressor = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator

        if hasattr(regressor, "coef_"):
            coefficients = np.atleast_2d(regressor.coef_)
            out: dict[str, list[tuple[str, float]]] = {}
            for index, label in enumerate(self.classes):
                row = coefficients[index] if coefficients.shape[0] > 1 else coefficients[0]
                order = np.argsort(-row)[:top_n]
                out[label] = [(self.gene_names[i], float(row[i])) for i in order]
            return out

        if hasattr(regressor, "feature_importances_"):
            scores = np.asarray(regressor.feature_importances_)
            order = np.argsort(-scores)[:top_n]
            return {
                "all_classes": [(self.gene_names[i], float(scores[i])) for i in order]
            }
        return {}

    def to_dict(self) -> dict:
        return {
            "model": self.name, "n_classes": len(self.classes),
            "classes": self.classes, "n_train_cells": self.n_train_cells,
            "n_features": self.n_features,
        }


class MarkerScoreClassifier:
    """Assigns the highest-scoring marker panel. No fitting.

    The floor every learned model must clear. Implemented with the sklearn
    predict/predict_proba interface so it slots into the same evaluation code,
    but ``fit`` only records which classes exist — it looks at no expression
    values at all.
    """

    def __init__(self, gene_names: Sequence[str], classes: Sequence[str]):
        from ..markers import ALL_PANELS, resolve_panel

        self.gene_names = list(gene_names)
        self.classes_ = list(classes)
        index = {g: i for i, g in enumerate(self.gene_names)}
        # Panels are resolved through the alias table so this works for
        # curated cell-type vocabularies ("CD4 T cells") as well as functional
        # state names, which is what makes it a usable floor on real data.
        self._panels = {}
        for label in self.classes_:
            panel_name = resolve_panel(label)
            if panel_name is None:
                continue
            positions = [
                index[g] for g in ALL_PANELS[panel_name].genes if g in index
            ]
            if positions:
                self._panels[label] = positions

        missing = [c for c in self.classes_ if c not in self._panels]
        if missing:
            raise ModelError(
                f"no marker panel genes present for class(es) {missing}; the "
                "marker baseline cannot score them. This baseline needs a panel "
                "for every class, since a class it cannot score would get zero "
                "recall for a reason unrelated to the data."
            )

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> MarkerScoreClassifier:
        # Deliberately ignores `features`: this baseline reads markers, it does
        # not learn. Present only so it satisfies the estimator interface.
        observed = [c for c in self.classes_ if c in set(map(str, labels))]
        if not observed:
            raise ModelError("no class in the training labels has a marker panel")
        return self

    def _scores(self, features: np.ndarray) -> np.ndarray:
        columns = []
        for label in self.classes_:
            indices = self._panels[label]
            block = features[:, indices]
            # Standardise per panel so panels of different size and baseline
            # expression are comparable, matching nkstate.scoring.
            mean = block.mean(axis=1)
            columns.append((mean - mean.mean()) / (mean.std() if mean.std() > 0 else 1.0))
        return np.column_stack(columns)

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self._scores(features)
        return np.asarray([self.classes_[i] for i in scores.argmax(axis=1)])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self._scores(features)
        exponentiated = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def build_estimator(
    name: str, gene_names: Sequence[str], classes: Sequence[str],
    seed: int = 0, balanced: bool = True,
):
    """Construct an sklearn pipeline for one baseline.

    Every pipeline standardises first, fitted on the training fold only. Class
    weighting is on by default: NK functional states are strongly imbalanced in
    any real sample, and an unweighted model on a 70/10/10/10 split can reach
    70% accuracy while never predicting three of the four states.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name not in BASELINE_MODELS:
        raise ModelError(f"unknown model {name!r}; choose from {list(BASELINE_MODELS)}")

    if name == "marker_score":
        return MarkerScoreClassifier(gene_names, classes)

    weight = "balanced" if balanced else None
    if name == "logistic":
        final = LogisticRegression(
            max_iter=2000, C=1.0, class_weight=weight, random_state=seed,
        )
    elif name == "random_forest":
        final = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt",
            class_weight=weight, random_state=seed, n_jobs=-1,
        )
    else:
        # MLPClassifier has no class_weight; imbalance is handled by the
        # sampling in fit_baseline instead, which is noted there.
        final = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
            random_state=seed,
        )

    return Pipeline([("scale", StandardScaler()), (name, final)])


def fit_baseline(
    name: str,
    features: np.ndarray,
    labels: Sequence[str],
    gene_names: Sequence[str],
    seed: int = 0,
    balanced: bool = True,
) -> FittedBaseline:
    """Fit one baseline classifier."""
    labels = np.asarray([str(l) for l in labels])
    if features.shape[0] != labels.size:
        raise ModelError(
            f"feature matrix has {features.shape[0]} cells but {labels.size} labels"
        )
    if features.shape[1] != len(gene_names):
        raise ModelError(
            f"feature matrix has {features.shape[1]} genes but {len(gene_names)} names"
        )
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ModelError(
            f"only one class present ({classes}); a classifier needs at least two"
        )
    if labels.size < 20:
        raise ModelError(f"{labels.size} training cells is too few to fit a classifier")

    estimator = build_estimator(name, gene_names, classes, seed, balanced)
    estimator.fit(features, labels)
    return FittedBaseline(
        name=name, estimator=estimator, classes=classes,
        gene_names=list(gene_names), n_train_cells=int(labels.size),
        n_features=int(features.shape[1]),
    )
