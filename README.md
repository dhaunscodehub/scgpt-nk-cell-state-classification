# NK cell functional state classification

Single-cell RNA-seq classification of natural killer cell states, built around
one question that most cell-state papers answer badly: **is the reported
accuracy a property of the biology, or of the donor the cells came from?**

The repository's central measurement is a controlled experiment, not a
leaderboard number. On a synthetic cohort where **no gene carries any
information about the state label**, a conventional cell-random split reaches
**0.655 balanced accuracy** against a chance level of 0.250. A donor-held-out
split on the same data reaches **0.252**. The entire difference —
**0.403 balanced accuracy** — is the model recognising which donor a cell came
from.

That is the failure mode this code is designed to make impossible to commit
silently.

```bash
pip install -e ".[all]"
python scripts/smoke_test.py            # 38 checks, under a minute, no network
python scripts/validate_controls.py     # the 8 control experiments
nkstate run -c configs/pbmc_reference.yaml | jq .comparison
```

---

## What is measured, and what is not

| | |
|---|---|
| **Status** | `IMPLEMENTED_NOT_FULLY_EXECUTED` |
| **Tests** | 300 passing |
| **Smoke test** | 38/38 checks |
| **Controls** | 8/8 passing |
| **Real-data task** | 8-class PBMC cell typing, curated labels, macro-F1 **0.966** |
| **scGPT fine-tuning** | `NOT_REPRODUCED` — needs non-redistributable weights and a GPU |

Every number in this repository is tagged with its provenance
(`VERIFIED_REPRODUCED`, `NOT_REPRODUCED`, `REFERENCE_RESULT`). See
[docs/VALIDATION.md](docs/VALIDATION.md).

---

## Results

### Real data: 8-class PBMC cell typing

`scanpy` pbmc3k — 2638 cells, curated labels from Zheng et al. (2017) that are
**independent of anything in this package**. All `VERIFIED_REPRODUCED`.

| Model | macro-F1 | balanced acc. | accuracy | macro AUROC | Cohen κ |
|---|---|---|---|---|---|
| logistic regression | **0.966** | 0.963 | 0.970 | 0.997 | 0.960 |
| random forest | 0.955 | 0.944 | 0.960 | 0.998 | 0.947 |
| MLP | 0.925 | 0.904 | 0.941 | 0.995 | 0.921 |
| `marker_score` (no fitting) | 0.478 | 0.695 | 0.710 | 0.911 | 0.629 |

Majority-class accuracy 0.428. All three learned models beat the
no-fitting marker floor, which is the comparison that decides whether learning
added anything.

**This run is single-donor**, so `split.kind` is `cell_random` and the numbers
describe within-donor separability. Given the control experiment above, they
are not an estimate of performance on a new donor — and the pipeline attaches
that caveat to its own output rather than leaving it to this README.

### A negative result, reported as one

Marker enrichment of each class's own differentially expressed genes **fails**
for two classes:

| Class | Overlap | Fold | p |
|---|---|---|---|
| CD8 T cells | 0/2 | 0.0 | 1.00 |
| FCGR3A+ Monocytes | 0/5 | 0.0 | 1.00 |

This is correct behaviour. The panels are lineage-level, so CD4 and CD8 T cells
resolve to the same `T_cell` panel; under one-vs-rest testing the shared
T-lineage genes are not differentially expressed for CD8-vs-rest, because the
comparison group expresses them too. The same holds for non-classical vs
classical monocytes.

It has not been fixed by adding `CD8A`/`GZMK` to the panel. That would make the
table pass while changing what is being tested. The finding stands: **lineage
panels do not validate subtype distinctions under one-vs-rest testing.**

---

## How leakage is prevented

Four mechanisms, each tested:

**Donor-grouped splits.** `donor_split` partitions donors, not cells, and
**refuses fewer than three donors** rather than silently degenerating into a
cell-level split. `assert_no_donor_leakage` runs inside the constructor *and*
again in the pipeline before training.

**Training-fold HVG selection.** Highly variable genes are chosen on the
training fold only. Selecting across all cells uses the test cells' expression
to decide what the model may see — a real leak, and an easy one to miss, since
scanpy tutorials place HVG selection before any split. With ≥2 donors,
selection is per donor and the union is taken, because pooled selection ranks
donor-specific variation highly and that variation is exactly the confound the
split exists to exclude.

**Training-fold standardisation.** Scalers are fitted on the training fold.
On single-cell data the dominant axis of variation is technical, so fitting on
all cells leaks a substantial amount.

**A distinct exit code.** Detected leakage exits **5**, separate from bad input
(2) or a crash. A leaking split is not an error — it is a scientifically invalid
run, and a batch script should be able to tell the difference without parsing
messages.

---

## Honest labelling of labels

No public dataset used here carries curated resting/activated/cytotoxic/
exhausted annotations. Marker-derived state labels are therefore always tagged
`label_type = "marker_pseudo_label"` and carry an explicit caveat.

This matters because a model scored against marker pseudo-labels is measured on
**agreement with the marker rule** — an upper bound on its agreement with true
cell state, not an accuracy. Reporting such a score as accuracy is the most
common way to overstate a cell-state classifier, and the code refuses to let it
happen quietly.

The `marker_score` baseline does no fitting at all. Any learned model must beat
it to have earned its complexity, and `compare_models` reports
`beats_marker_score_floor` explicitly. A `False` there is a result.

---

## Commands

```bash
nkstate run      -c configs/pbmc_reference.yaml   # full pipeline
nkstate markers  -c configs/pbmc_reference.yaml   # panels + dataset coverage
nkstate scgpt    -c configs/pbmc_reference.yaml   # checkpoint status, fine-tune command
nkstate tokenise -c configs/pbmc_reference.yaml   # validate scGPT input encoding
nkstate validate                                  # the control experiments
```

Status messages go to **stderr**, JSON to **stdout**, so
`nkstate run -c config.yaml | jq .comparison` works and redirecting stdout
yields a valid JSON document.

Exit codes: `0` ok, `2` bad input, `3` missing dependency, `4` external tool
unavailable, `5` **leakage detected**.

---

## Configuration

YAML, with **unknown keys and sections rejected as errors**. A misspelled
`n_top_genes` that is silently ignored produces a run whose config file
contradicts what actually happened:

```
[data] unknown key(s) ['n_top_gene']; valid keys are ['cell_type_key',
'donor_key', 'label_key', 'max_mito_fraction', 'min_cells_per_gene', ...]
```

Paths resolve relative to the config file, so configs are portable across
working directories. Two are shipped:

| Config | Purpose |
|---|---|
| `configs/pbmc_reference.yaml` | Real data, curated labels, single donor |
| `configs/synthetic_donor_split.yaml` | 12 donors, genuine donor-grouped split and donor-level metrics |

---

## scGPT

The scGPT stage is **implemented but not executed**. Fine-tuning needs a
pretrained checkpoint that the authors distribute under their own terms — not
redistributed here — plus a GPU. **No transformer metric appears anywhere in
this repository.**

What *is* implemented and covered by 31 tests is the input transformation,
which is where the errors that silently ruin a fine-tune actually live:

* **Vocabulary mapping** reports dropped **marker** genes separately from
  dropped background genes — losing GNLY from a cytotoxic task is fatal,
  losing an anonymous background gene is not, and one coverage percentage
  hides the difference.
* **Expression binning** is per-cell over non-zero values only, matching
  scGPT's scheme. Binning globally would make the representation depend on
  library size; including zeros would push ~90% of genes into the lowest bin.
* **Sequence truncation** pins marker genes, which are often not highly
  expressed and would otherwise be truncated away — leaving the model to
  classify functional states without seeing the genes that define them.

See [docs/SCGPT.md](docs/SCGPT.md) to run it yourself.

---

## Layout

```
src/nkstate/
  markers.py          curated panels, lineage aliases, coverage reporting
  scoring.py          control-corrected panel scoring, ambiguity margins
  evaluate.py         cell-level and donor-level metrics, kept separate
  interpret.py        Wilcoxon DE, BH correction, hypergeometric enrichment
  pipeline.py         orchestration, in the order leakage control requires
  validation.py       the 8 control experiments
  config.py           typed config; unknown keys are errors
  cli.py              stderr for status, stdout for JSON
  testing.py          synthetic cohorts with controllable confounds
  data/loading.py     QC with per-rule attribution, HVG strategy
  data/splits.py      donor grouping and the leakage guards
  models/baselines.py logistic / forest / MLP / marker floor
  models/scgpt.py     tokenisation, binning, command construction
scripts/
  smoke_test.py       38 checks, no network
  validate_controls.py  the control experiments, PASS/FAIL table
docs/                 VALIDATION.md, METHODS.md, SCGPT.md
hpc/slurm/            CPU pipeline job, GPU fine-tune job (unexecuted)
```

---

## Documentation

* [docs/VALIDATION.md](docs/VALIDATION.md) — every measurement, its provenance
  tag, and the eight bugs the controls caught
* [docs/METHODS.md](docs/METHODS.md) — panels, splitting, metrics, statistics
* [docs/SCGPT.md](docs/SCGPT.md) — what is missing and how to run it
* [data/README.md](data/README.md) — what is downloaded, and the pbmc3k `.X`
  trap
* [PROJECT_STATE.md](PROJECT_STATE.md) — status, limitations, next steps

## Citation and attribution

Marker panels are drawn from published literature; every panel carries its
reference in `markers.py` and they are listed in
[docs/METHODS.md](docs/METHODS.md).

This repository contains **no scGPT source code and no scGPT weights**. scGPT
is the work of Cui et al., *Nature Methods* 21:1470 (2024).

Datasets are downloaded from their original distributors and are not vendored
here. See [data/README.md](data/README.md).

## License

MIT — see [LICENSE](LICENSE).
