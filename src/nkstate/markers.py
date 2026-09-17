"""Marker gene panels for NK-cell identity and functional state.

The four states in the resume claim — resting, activated, cytotoxic and
dysfunctional/exhausted — are not discrete populations with a consensus
definition. They are regions of a continuum, and the marker panels below are
assembled from the NK biology literature cited per panel. Two consequences are
enforced throughout this package:

1. A state assignment derived from these markers is a **pseudo-label**, not
   ground truth. Every score and every label carries that provenance, and
   nothing reports a classifier's agreement with a marker score as if it were
   accuracy against a measured label.
2. Panels overlap on purpose. Cytotoxic and activated NK cells both upregulate
   effector genes; exhausted cells retain cytotoxic machinery while gaining
   inhibitory receptors. A panel design that forced disjoint gene sets would
   misrepresent the biology, so the overlap is quantified by
   :func:`panel_overlap` and reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NK_STATES = ("resting", "activated", "cytotoxic", "exhausted")


@dataclass(frozen=True)
class MarkerPanel:
    """A named gene panel with its provenance."""

    name: str
    genes: tuple[str, ...]
    description: str
    reference: str
    direction: str = "up"

    def __post_init__(self) -> None:
        if not self.genes:
            raise ValueError(f"panel {self.name!r} has no genes")
        if len(set(self.genes)) != len(self.genes):
            duplicates = sorted({g for g in self.genes if self.genes.count(g) > 1})
            raise ValueError(f"panel {self.name!r} repeats gene(s) {duplicates}")
        if self.direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")

    def to_dict(self) -> dict:
        return {
            "panel": self.name, "n_genes": len(self.genes), "genes": list(self.genes),
            "description": self.description, "reference": self.reference,
            "direction": self.direction,
        }


# --- identity ---------------------------------------------------------------

NK_IDENTITY = MarkerPanel(
    name="nk_identity",
    genes=(
        "NCAM1",   # CD56
        "KLRD1",   # CD94
        "KLRF1",   # NKp80
        "NKG7", "GNLY", "PRF1",
        "FCGR3A",  # CD16
        "TYROBP", "KLRB1", "NCR3", "CD247",
    ),
    description="Canonical NK-cell identity genes; used to isolate NK cells from PBMCs",
    reference="Freud et al., Immunity 47:820 (2017); Crinier et al., Immunity 49:971 (2018)",
)

# NK cells are CD3-negative. Included explicitly because CD3+ NKT cells and
# cytotoxic CD8 T cells share much of the effector programme and are the main
# contaminant of a marker-only NK gate.
T_CELL_EXCLUSION = MarkerPanel(
    name="t_cell_exclusion",
    genes=("CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2", "CD8A", "CD4"),
    description="T-cell genes; NK cells must be negative for these",
    reference="Freud et al., Immunity 47:820 (2017)",
    direction="down",
)

# --- functional states ------------------------------------------------------

RESTING = MarkerPanel(
    name="resting",
    genes=(
        "SELL",    # CD62L, retained on circulating resting NK
        "IL7R", "CD27", "TCF7", "LEF1",
        "KLRC1",   # NKG2A, high on less-differentiated NK
        "XCL1", "GPR183", "CCR7", "TPT1",
    ),
    description=(
        "Resting / less-differentiated circulating NK: lymph-node homing "
        "receptors, NKG2A-high, low effector transcript load"
    ),
    reference="Crinier et al., Immunity 49:971 (2018); Smith et al., Cell Rep 32:107938 (2020)",
)

ACTIVATED = MarkerPanel(
    name="activated",
    genes=(
        "IFNG", "TNF",
        "CCL3", "CCL4", "CCL4L2", "XCL2",
        "CD69",    # early activation
        "NR4A1", "EGR1", "EGR2",  # immediate-early transcription factors
        "TNFAIP3", "REL", "NFKBIA",
    ),
    description=(
        "Recently activated NK: cytokine and chemokine output plus "
        "immediate-early transcription factors"
    ),
    reference="Smith et al., Cell Rep 32:107938 (2020); Collins et al., Cell 176:348 (2019)",
)

CYTOTOXIC = MarkerPanel(
    name="cytotoxic",
    genes=(
        "GNLY", "PRF1", "GZMB", "GZMH", "GZMA",
        "NKG7", "FGFBP2", "SPON2", "CX3CR1",
        "FCGR3A",  # CD16, on mature cytotoxic NK
        "KLRD1", "CST7", "CTSW", "S1PR5",
    ),
    description=(
        "Mature cytotoxic effector NK: granule contents, CD16-high, "
        "terminally differentiated surface phenotype"
    ),
    reference="Crinier et al., Immunity 49:971 (2018); Yang et al., Nat Commun 10:3931 (2019)",
)

EXHAUSTED = MarkerPanel(
    name="exhausted",
    genes=(
        "LAG3", "HAVCR2",  # TIM-3
        "PDCD1",           # PD-1
        "TIGIT", "CTLA4", "KLRG1", "CD160",
        "TOX", "ENTPD1",   # CD39
        "BATF", "IKZF2", "CD101",
    ),
    description=(
        "Dysfunctional / exhausted NK: inhibitory checkpoint receptors and "
        "the exhaustion-associated transcription factor programme"
    ),
    reference=(
        "Judge et al., Front Cell Dev Biol 8:49 (2020); "
        "Bi & Tian, Front Immunol 8:1999 (2017); Merino et al., JCI Insight 4:e128703 (2019)"
    ),
)

STATE_PANELS: dict[str, MarkerPanel] = {
    "resting": RESTING,
    "activated": ACTIVATED,
    "cytotoxic": CYTOTOXIC,
    "exhausted": EXHAUSTED,
}

# Proliferation is orthogonal to functional state and is scored separately so a
# cycling cytotoxic cell is not mistaken for a distinct state.
PROLIFERATION = MarkerPanel(
    name="proliferation",
    genes=("MKI67", "TOP2A", "STMN1", "TUBB", "PCNA", "TYMS", "CDK1", "CCNB1"),
    description="Cell-cycle genes; scored separately because proliferation is "
                "orthogonal to functional state",
    reference="Tirosh et al., Science 352:189 (2016)",
)

# Broader PBMC lineage panels, used to validate an NK gate by confirming that
# other lineages score low. Deliberately minimal.
PBMC_LINEAGES: dict[str, MarkerPanel] = {
    "T_cell": MarkerPanel(
        "T_cell", ("CD3D", "CD3E", "IL7R", "TRAC", "LTB"),
        "T lymphocytes", "Zheng et al., Nat Commun 8:14049 (2017)",
    ),
    "B_cell": MarkerPanel(
        "B_cell", ("MS4A1", "CD79A", "CD79B", "IGHM", "TCL1A"),
        "B lymphocytes", "Zheng et al., Nat Commun 8:14049 (2017)",
    ),
    "monocyte": MarkerPanel(
        "monocyte", ("LYZ", "CD14", "S100A8", "S100A9", "FCN1"),
        "Monocytes", "Zheng et al., Nat Commun 8:14049 (2017)",
    ),
    "NK_cell": MarkerPanel(
        "NK_cell", ("GNLY", "NKG7", "KLRD1", "NCAM1", "KLRF1"),
        "NK cells", "Zheng et al., Nat Commun 8:14049 (2017)",
    ),
    "dendritic": MarkerPanel(
        "dendritic", ("FCER1A", "CST3", "CLEC10A", "LYZ"),
        "Dendritic cells", "Zheng et al., Nat Commun 8:14049 (2017)",
    ),
    "megakaryocyte": MarkerPanel(
        "megakaryocyte", ("PPBP", "PF4", "GP9", "ITGA2B", "TUBB1"),
        "Megakaryocytes and platelets", "Zheng et al., Nat Commun 8:14049 (2017)",
    ),
}


# Cell-type labels as they appear in public annotations, mapped to the lineage
# panel they correspond to. Needed because annotation vocabularies differ
# between datasets ("CD4 T cells", "CD4+ T", "T_cell") while the marker panel
# does not, and an enrichment test that silently fails to match a label would
# report "no panel" for a cell type that plainly has one.
LINEAGE_ALIASES: dict[str, str] = {
    "t cell": "T_cell", "t cells": "T_cell", "t_cell": "T_cell",
    "cd4 t cells": "T_cell", "cd4+ t cells": "T_cell", "cd4 t": "T_cell",
    "cd8 t cells": "T_cell", "cd8+ t cells": "T_cell", "cd8 t": "T_cell",
    "b cell": "B_cell", "b cells": "B_cell", "b_cell": "B_cell",
    "monocyte": "monocyte", "monocytes": "monocyte",
    "cd14+ monocytes": "monocyte", "fcgr3a+ monocytes": "monocyte",
    "cd16+ monocytes": "monocyte",
    "nk cell": "NK_cell", "nk cells": "NK_cell", "nk": "NK_cell",
    "nk_cell": "NK_cell", "natural killer": "NK_cell",
    "dendritic": "dendritic", "dendritic cells": "dendritic", "dc": "dendritic",
    "megakaryocyte": "megakaryocyte", "megakaryocytes": "megakaryocyte",
    "platelet": "megakaryocyte", "platelets": "megakaryocyte",
}

# Every panel that can be tested for enrichment, functional states and
# lineages together.
ALL_PANELS: dict[str, MarkerPanel] = {**STATE_PANELS, **PBMC_LINEAGES}


def resolve_panel(label: str) -> str | None:
    """Return the panel name for a label, or ``None`` if there is none.

    Resolution order: exact panel name, then the alias table, then a
    case-insensitive match. Returns ``None`` rather than raising because many
    real cell-type labels legitimately have no panel here — megakaryocytes and
    erythrocytes in PBMC data, for instance — and those should be reported as
    untested, not treated as an error.
    """
    if label in ALL_PANELS:
        return label
    key = str(label).strip().lower()
    if key in LINEAGE_ALIASES:
        return LINEAGE_ALIASES[key]
    for name in ALL_PANELS:
        if name.lower() == key:
            return name
    return None


def all_state_genes() -> tuple[str, ...]:
    """Every gene appearing in any functional-state panel, deduplicated."""
    seen: list[str] = []
    for panel in STATE_PANELS.values():
        for gene in panel.genes:
            if gene not in seen:
                seen.append(gene)
    return tuple(seen)


def panel_overlap() -> dict[str, dict[str, int]]:
    """Pairwise shared-gene counts between the state panels.

    Reported rather than eliminated. Cytotoxic and activated NK cells share
    effector genes and exhausted cells retain cytotoxic machinery, so disjoint
    panels would misrepresent the biology. Quantifying the overlap is what lets
    a reader judge how separable the marker scores can possibly be.
    """
    names = list(STATE_PANELS)
    return {
        a: {
            b: len(set(STATE_PANELS[a].genes) & set(STATE_PANELS[b].genes))
            for b in names if b != a
        }
        for a in names
    }


def genes_present(panel: MarkerPanel, available: set[str]) -> tuple[list[str], list[str]]:
    """Split a panel into genes present in the dataset and genes missing.

    A panel with most of its genes missing cannot support a score, and silently
    scoring the remainder would produce a number that looks like the others but
    measures something else. Callers check the coverage.
    """
    present = [g for g in panel.genes if g in available]
    missing = [g for g in panel.genes if g not in available]
    return present, missing


def panel_coverage(available: set[str]) -> dict[str, dict]:
    """Per-panel gene coverage against a dataset's gene list."""
    report: dict[str, dict] = {}
    for name, panel in {**STATE_PANELS, "nk_identity": NK_IDENTITY,
                         "proliferation": PROLIFERATION}.items():
        present, missing = genes_present(panel, available)
        report[name] = {
            "n_panel_genes": len(panel.genes),
            "n_present": len(present),
            "n_missing": len(missing),
            "coverage": len(present) / len(panel.genes),
            "present": present,
            "missing": missing,
        }
    return report
