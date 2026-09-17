"""Serialisation and provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from nkstate.io_utils import package_versions, provenance, sanitise, write_json


def test_numpy_scalars_serialise(tmp_path):
    path = write_json(
        tmp_path / "out.json",
        {"b": np.bool_(True), "i": np.int64(4), "f": np.float64(1.25)},
    )
    loaded = json.loads(path.read_text())
    assert loaded == {"b": True, "i": 4, "f": 1.25}


def test_numpy_bool_is_not_coerced_to_an_integer():
    """np.bool_ is not an np.integer subclass; order in the encoder matters."""
    assert sanitise({"x": np.bool_(False)})["x"] is False


def test_arrays_become_lists(tmp_path):
    path = write_json(tmp_path / "out.json", {"a": np.arange(3)})
    assert json.loads(path.read_text())["a"] == [0, 1, 2]


@pytest.mark.parametrize("value", [float("nan"), np.float64("nan")])
def test_nan_becomes_null(value, tmp_path):
    """NaN has no JSON representation; emitting null keeps the file portable.

    Both spellings are covered deliberately: np.float64 is a subclass of float
    and is serialised natively by the C encoder, so it never reaches
    JSONEncoder.default and can only be handled before the dump.
    """
    path = write_json(tmp_path / "out.json", {"x": value})
    assert json.loads(path.read_text())["x"] is None


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), np.float64("inf")])
def test_infinity_becomes_null(value, tmp_path):
    path = write_json(tmp_path / "out.json", {"x": value})
    assert json.loads(path.read_text())["x"] is None


def test_nan_nested_in_lists_and_dicts_becomes_null(tmp_path):
    payload = {"outer": [{"inner": np.float64("nan")}, [float("nan")]]}
    path = write_json(tmp_path / "out.json", payload)
    loaded = json.loads(path.read_text())
    assert loaded["outer"][0]["inner"] is None
    assert loaded["outer"][1][0] is None


def test_output_is_parseable_by_a_strict_parser(tmp_path):
    """A file containing bare NaN parses in Python but nowhere else."""
    path = write_json(tmp_path / "out.json", {"x": np.float64("nan")})
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=_reject)


def _reject(value):  # pragma: no cover - only called on invalid input
    raise AssertionError(f"non-standard JSON constant in output: {value}")


def test_finite_floats_are_untouched():
    assert sanitise({"x": 1.5})["x"] == 1.5
    assert sanitise({"x": np.float64(-2.25)})["x"] == -2.25


def test_dataclasses_serialise(tmp_path):
    @dataclass
    class Point:
        x: int
        y: float

    path = write_json(tmp_path / "out.json", {"p": Point(1, 2.0)})
    assert json.loads(path.read_text())["p"] == {"x": 1, "y": 2.0}


def test_paths_serialise_as_strings(tmp_path):
    path = write_json(tmp_path / "out.json", {"p": Path("/a/b")})
    assert json.loads(path.read_text())["p"] == "/a/b"


def test_undefined_cohen_kappa_survives_serialisation(tmp_path):
    """sklearn returns NaN for an undefined kappa; the run must not crash."""
    from nkstate.evaluate import cell_metrics

    metrics = cell_metrics(["a", "a"], ["a", "a"])
    path = write_json(tmp_path / "out.json", metrics.to_dict())
    loaded = json.loads(path.read_text())
    assert "cohen_kappa" in loaded


def test_write_json_creates_parent_directories(tmp_path):
    path = write_json(tmp_path / "deep" / "nested" / "out.json", {"x": 1})
    assert path.is_file()


def test_provenance_records_the_environment():
    record = provenance()
    for key in ("timestamp_utc", "python", "platform", "packages"):
        assert key in record


def test_provenance_accepts_extra_fields():
    assert provenance({"run": "abc"})["run"] == "abc"


def test_package_versions_records_absent_packages_as_none():
    versions = package_versions(("numpy", "definitely-not-installed-xyz"))
    assert versions["numpy"] is not None
    assert versions["definitely-not-installed-xyz"] is None
