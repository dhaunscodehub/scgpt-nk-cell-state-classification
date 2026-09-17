"""Typed configuration loaded from YAML.

Unknown keys and unknown sections are **errors**, not warnings. A misspelled
``n_top_genes`` that is silently ignored produces a run with the default value
and a config file that claims otherwise, which is the kind of discrepancy that
makes a result impossible to reproduce later.

Paths resolve relative to the config file's own directory, so a config is
portable: moving the repository, or running from a different working
directory, does not change what it refers to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .markers import NK_STATES
from .models.baselines import BASELINE_MODELS


class ConfigError(ValueError):
    """Raised for an invalid or unrecognised configuration."""


def _check_keys(section: str, payload: dict, cls: type) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ConfigError(
            f"[{section}] unknown key(s) {unknown}; valid keys are "
            f"{sorted(known)}. Unknown keys are rejected rather than ignored so "
            "a typo cannot silently change what a run does."
        )


@dataclass
class DataConfig:
    """Where the data comes from and how it is preprocessed."""

    source: str = "reference_pbmc"
    path: str | None = None
    donor_key: str = "donor_id"
    label_key: str | None = None
    cell_type_key: str | None = "cell_type"
    min_genes: int = 200
    min_counts: int = 500
    max_mito_fraction: float = 0.20
    min_cells_per_gene: int = 3
    target_sum: float = 1e4
    n_top_genes: int = 2000
    subset_nk: bool = False
    nk_labels: list[str] = field(default_factory=list)
    nk_identity_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.source not in ("reference_pbmc", "h5ad", "synthetic"):
            raise ConfigError(
                f"[data] source must be one of reference_pbmc, h5ad, synthetic; "
                f"got {self.source!r}"
            )
        if self.source == "h5ad" and not self.path:
            raise ConfigError("[data] source 'h5ad' requires a path")
        if not 0 < self.max_mito_fraction <= 1:
            raise ConfigError(
                f"[data] max_mito_fraction is a fraction in (0, 1], not a "
                f"percentage; got {self.max_mito_fraction}"
            )
        if self.n_top_genes < 50:
            raise ConfigError(
                f"[data] n_top_genes={self.n_top_genes} is too few for a "
                "transcriptome-wide classifier; use at least 50"
            )
        if min(self.min_genes, self.min_counts, self.min_cells_per_gene) < 0:
            raise ConfigError(
                "[data] min_genes, min_counts and min_cells_per_gene must be "
                "non-negative"
            )
        if self.target_sum <= 0:
            raise ConfigError(f"[data] target_sum must be positive; got {self.target_sum}")

    def resolve(self, root: Path) -> None:
        if self.path:
            self.path = str((root / self.path).resolve())


@dataclass
class SyntheticConfig:
    """Shape and difficulty of the synthetic cohort, when data.source is synthetic.

    These are the knobs that make a control experiment a control: ``state_effect``
    sets how recoverable the biology is (0 makes it unlearnable, which is the
    negative control), ``donor_effect`` adds a per-donor batch shift, and
    ``donor_confound`` ties state composition to donor identity so a
    cell-random split can be shown to exploit it.
    """

    n_donors: int = 12
    cells_per_donor: int = 150
    state_effect: float = 0.35
    donor_effect: float = 0.3
    donor_confound: bool = False
    n_background_genes: int = 400
    mean_counts_per_cell: float = 3000.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_donors < 1:
            raise ConfigError(f"[synthetic] n_donors must be >= 1; got {self.n_donors}")
        if self.cells_per_donor < 10:
            raise ConfigError(
                f"[synthetic] cells_per_donor must be >= 10; got {self.cells_per_donor}"
            )
        for name in ("state_effect", "donor_effect"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"[synthetic] {name} must be in [0, 1]; got {value}")

    def to_spec(self):
        from .testing import SyntheticSpec

        return SyntheticSpec(
            n_donors=self.n_donors, cells_per_donor=self.cells_per_donor,
            n_background_genes=self.n_background_genes,
            state_effect=self.state_effect, donor_effect=self.donor_effect,
            donor_confound=self.donor_confound,
            mean_counts_per_cell=self.mean_counts_per_cell, seed=self.seed,
        )


@dataclass
class SplitConfig:
    """How cells are divided into train, validation and test."""

    kind: str = "donor"
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    seed: int = 0
    n_folds: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("donor", "cell_random"):
            raise ConfigError(
                f"[split] kind must be 'donor' or 'cell_random'; got {self.kind!r}. "
                "'cell_random' leaks donors between folds and exists only as a "
                "comparison against 'donor'."
            )
        for name in ("test_fraction", "val_fraction"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ConfigError(f"[split] {name} must be in [0, 1); got {value}")
        if self.test_fraction + self.val_fraction >= 1:
            raise ConfigError(
                f"[split] test_fraction + val_fraction = "
                f"{self.test_fraction + self.val_fraction:.2f} leaves no training data"
            )
        if self.n_folds and self.n_folds < 2:
            raise ConfigError(f"[split] n_folds must be 0 or >= 2; got {self.n_folds}")


@dataclass
class LabelConfig:
    """Where the state labels come from."""

    source: str = "marker_score"
    states: list[str] = field(default_factory=lambda: list(NK_STATES))
    margin: float = 0.25
    drop_ambiguous: bool = True

    def __post_init__(self) -> None:
        if self.source not in ("marker_score", "annotation"):
            raise ConfigError(
                f"[labels] source must be 'marker_score' or 'annotation'; "
                f"got {self.source!r}"
            )
        if len(self.states) < 2:
            raise ConfigError(
                f"[labels] need at least 2 states to classify; got {self.states}"
            )
        # Marker-based scoring is only meaningful for a class that has a
        # panel; lineage panels are accepted as well as functional states, so
        # the same machinery covers the cell-type task.
        if self.source == "marker_score":
            from .markers import ALL_PANELS, resolve_panel

            unknown = [s for s in self.states if resolve_panel(s) is None]
            if unknown:
                raise ConfigError(
                    f"[labels] no marker panel for {unknown}; available panels "
                    f"are {sorted(ALL_PANELS)}"
                )
        if self.margin < 0:
            raise ConfigError(f"[labels] margin must be non-negative; got {self.margin}")


@dataclass
class ModelConfig:
    """Which classifiers to fit."""

    baselines: list[str] = field(default_factory=lambda: list(BASELINE_MODELS))
    seed: int = 0
    balanced: bool = True

    def __post_init__(self) -> None:
        unknown = [m for m in self.baselines if m not in BASELINE_MODELS]
        if unknown:
            raise ConfigError(
                f"[models] unknown baseline(s) {unknown}; available: "
                f"{list(BASELINE_MODELS)}"
            )
        if not self.baselines:
            raise ConfigError("[models] at least one baseline is required")


@dataclass
class ScgptConfig:
    """scGPT checkpoint location and fine-tuning settings.

    ``checkpoint_dir`` is ``None`` by default because the weights are not
    redistributable; with it unset, the scGPT stage reports what it would run
    and why it cannot, rather than failing the whole pipeline.
    """

    checkpoint_dir: str | None = None
    vocabulary_path: str | None = None
    max_seq_len: int = 1200
    n_bins: int = 51
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    freeze_encoder: bool = False

    def __post_init__(self) -> None:
        if self.max_seq_len < 64:
            raise ConfigError(
                f"[scgpt] max_seq_len={self.max_seq_len} is too short to carry "
                "the marker panels plus context; use at least 64"
            )
        if self.n_bins < 2:
            raise ConfigError(f"[scgpt] n_bins must be at least 2; got {self.n_bins}")
        if self.epochs < 1:
            raise ConfigError(f"[scgpt] epochs must be at least 1; got {self.epochs}")
        if not 0 < self.learning_rate < 1:
            raise ConfigError(
                f"[scgpt] learning_rate must be in (0, 1); got {self.learning_rate}"
            )

    def resolve(self, root: Path) -> None:
        for name in ("checkpoint_dir", "vocabulary_path"):
            value = getattr(self, name)
            if value:
                setattr(self, name, str((root / value).resolve()))


@dataclass
class InterpretConfig:
    """Differential expression and enrichment settings."""

    top_n_genes: int = 50
    min_cells_per_state: int = 10
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.top_n_genes < 5:
            raise ConfigError(
                f"[interpret] top_n_genes={self.top_n_genes} is too few for a "
                "meaningful enrichment test; use at least 5"
            )


@dataclass
class RunConfig:
    """A complete pipeline configuration."""

    name: str = "nk-state-run"
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    scgpt: ScgptConfig = field(default_factory=ScgptConfig)
    interpret: InterpretConfig = field(default_factory=InterpretConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def resolve(self, root: Path) -> None:
        self.output_dir = str((root / self.output_dir).resolve())
        self.data.resolve(root)
        self.scgpt.resolve(root)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


_SECTIONS = {
    "data": DataConfig, "split": SplitConfig, "labels": LabelConfig,
    "models": ModelConfig, "scgpt": ScgptConfig, "interpret": InterpretConfig,
    "synthetic": SyntheticConfig,
}


def config_from_dict(payload: dict, root: Path | None = None) -> RunConfig:
    """Build a :class:`RunConfig` from a plain mapping."""
    if not isinstance(payload, dict):
        raise ConfigError(f"expected a mapping at the top level, got {type(payload).__name__}")

    top_level = {"name", "output_dir", *_SECTIONS}
    unknown = sorted(set(payload) - top_level)
    if unknown:
        raise ConfigError(
            f"unknown top-level section(s) {unknown}; valid sections are "
            f"{sorted(top_level)}"
        )

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        block = payload.get(name, {}) or {}
        if not isinstance(block, dict):
            raise ConfigError(f"[{name}] must be a mapping, got {type(block).__name__}")
        _check_keys(name, block, cls)
        sections[name] = cls(**block)

    config = RunConfig(
        name=str(payload.get("name", "nk-state-run")),
        output_dir=str(payload.get("output_dir", "results")),
        **sections,
    )
    if root is not None:
        config.resolve(root)
    return config


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config, resolving paths relative to the file itself."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    return config_from_dict(payload, root=path.resolve().parent)
