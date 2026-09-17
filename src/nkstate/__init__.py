"""NK-cell functional state classification from single-cell RNA-seq.

Layers:

``nkstate.data``
    AnnData loading, quality control, normalisation, highly-variable-gene
    selection, and **donor-grouped splits**. Cells from one donor share that
    donor's genotype, ambient RNA and sequencing batch, so a cell-level random
    split evaluates memorisation. Split construction is assertion-guarded.

``nkstate.markers`` / ``nkstate.scoring``
    Published marker panels for the four NK functional states, and
    marker-based state scoring. Scores derived this way are **pseudo-labels**,
    not ground truth, and every output records that.

``nkstate.models``
    A baseline classifier stack that runs on expression alone, plus the scGPT
    integration layer (gene tokenisation, expression binning, fine-tuning
    command construction). scGPT needs pretrained weights that are not
    redistributable, so the baselines are what actually executes here.

``nkstate.evaluate`` / ``nkstate.interpret``
    F1, AUROC and confusion matrices at cell *and* donor level, and
    differential expression with marker enrichment for interpreting what a
    classifier learned.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
