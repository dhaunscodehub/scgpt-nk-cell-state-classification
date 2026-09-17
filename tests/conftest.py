"""Shared fixtures.

The synthetic builders are session-scoped because negative-binomial sampling
over a few hundred genes dominates the runtime of the whole suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def strong_signal():
    """A separable multi-donor cohort: the positive-control fixture."""
    from nkstate.testing import SyntheticSpec, build_preprocessed

    return build_preprocessed(
        SyntheticSpec(n_donors=6, cells_per_donor=120, state_effect=0.8, seed=0)
    )


@pytest.fixture(scope="session")
def no_signal():
    """A cohort whose labels carry no expression signal."""
    from nkstate.testing import SyntheticSpec, build_preprocessed

    return build_preprocessed(
        SyntheticSpec(n_donors=6, cells_per_donor=120, state_effect=0.0, seed=1)
    )


@pytest.fixture(scope="session")
def confounded():
    """A cohort whose only usable signal is donor identity."""
    from nkstate.testing import SyntheticSpec, build_preprocessed

    return build_preprocessed(
        SyntheticSpec(
            n_donors=10, cells_per_donor=120, state_effect=0.0,
            donor_effect=0.8, donor_confound=True, seed=2,
        )
    )
