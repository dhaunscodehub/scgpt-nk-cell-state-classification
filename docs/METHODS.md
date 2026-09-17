# Methods

## Marker panels

Four NK functional states, each defined by a curated panel with a literature
reference:

| State | Genes | Reference |
|---|---|---|
| resting | SELL, IL7R, TCF7, KLF2, CXCR4, GZMK, XCL1, CD27, LTB, S1PR5 | Crinier et al., *Immunity* 49:971 (2018) |
| activated | CD69, IFNG, TNF, CCL3, CCL4, CCL4L2, XCL2, NR4A1, EGR1, EGR2, REL, NFKBIA, TNFAIP3 | Smith et al., *Cell Rep* 33:108384 (2020) |
| cytotoxic | GNLY, PRF1, GZMB, GZMH, GZMA, NKG7, KLRD1, FGFBP2, FCGR3A, SPON2, CST7, CTSW, S1PR5, ADGRG1 | Freud et al., *Immunity* 47:820 (2017) |
| exhausted | LAG3, HAVCR2, PDCD1, TIGIT, CTLA4, KLRG1, CD160, TOX, ENTPD1, BATF, IKZF2, CD101 | Judge et al., *Front Immunol* 11:1989 (2020); Bi & Tian, *Front Immunol* 8:1999 (2017) |

Five lineage panels (T cell, B cell, monocyte, NK cell, dendritic) and a
megakaryocyte panel support the cell-type task, all from Zheng et al., *Nat
Commun* 8:14049 (2017).

Panels overlap by design — `S1PR5` appears in both resting and cytotoxic,
`FCGR3A` in cytotoxic — and `panel_overlap()` reports the overlap rather than
hiding it. Overlap is why assignment uses a confidence margin instead of a
bare argmax.

### Scoring

Per-cell panel scores use `scanpy.tl.score_genes` (Tirosh et al., *Science*
352:189, 2016): the mean expression of the panel minus the mean of an
expression-matched control set. The control correction matters because a panel
of highly expressed genes scores highly in every cell without it.

Two guards:

* **Coverage floor** (`MIN_PANEL_COVERAGE = 0.4`). A score computed from 3 of
  12 panel genes is not the panel's score, and putting it next to a fully
  covered panel's score invites a comparison that is not valid. Below the floor
  the call is refused.
* **Background pool** (`MIN_BACKGROUND_GENES = 50`). The control set is sampled
  from expression-matched bins of non-panel genes; too small a pool makes some
  bin empty. Without this guard scanpy raises a message that does not identify
  the cause.

### Assignment

Scores are standardised per state before comparison — panels differ in size and
baseline expression, so raw scores are not on a common scale. A cell whose best
and second-best standardised scores differ by less than `margin` (default 0.25)
is labelled `"ambiguous"` rather than forced into a class.

Every assignment carries `label_type = "marker_pseudo_label"` and an explicit
caveat. This is not decoration: a model scored against marker pseudo-labels is
measured on **agreement with the marker rule**, which is an upper bound on its
agreement with true cell state. Treating such a score as accuracy is the single
most common way to overstate a cell-state classifier.

---

## Splitting

### Donor-grouped

`donor_split` partitions **donors**, then takes all of each donor's cells. It
refuses fewer than three donors: with two, there is no way to form disjoint
train, validation and test donor groups, and silently degenerating into a
cell-level split would invalidate every downstream metric without saying so.

`assert_no_donor_leakage` is called inside the constructor **and** again in the
pipeline before training. Checking twice is nearly free, and this is the
invariant every reported number depends on.

### Cell-random

`cell_split` exists only as a comparison. It shares donors across folds by
construction, records `shared_donors_train_test` in its own metadata, and the
evaluation layer attaches a caveat to any result computed on it. Control
experiment 3 quantifies what it costs: 0.403 balanced accuracy of pure
optimism on a cohort with no biological signal.

### Highly variable gene selection

HVGs are selected **on the training fold only**. Selecting across all cells
uses the test cells' expression to decide which genes the model may see. This
is a real leak and an easy one to miss, since scanpy tutorials place HVG
selection in preprocessing, before any split.

With two or more donors, selection is per donor and the **union** is taken.
Pooled selection ranks donor-specific variation highly; that variation is real,
but it is precisely the confound a donor-grouped split exists to exclude.

Marker-panel genes are forced back in regardless of variance rank. This is a
disclosed injection of prior knowledge, not a leak: the panels come from
published literature, not from this dataset's test cells. Without it, selection
sometimes drops the genes the states are defined by, leaving the classifier to
recover cytotoxicity without seeing GNLY.

---

## Models

| Model | Role |
|---|---|
| `logistic` | Multinomial logistic regression on standardised expression. The reference. Signed per-class coefficients make the interpretation step meaningful. |
| `random_forest` | Non-linear, robust to heavy-tailed expression, supplies feature importances. |
| `mlp` | Small dense network — the closest baseline in family to a fine-tuned transformer head. |
| `marker_score` | Assigns the highest-scoring panel. **No fitting.** The floor. |

Features are standardised with training-fold statistics only; fitting the
scaler on all cells leaks the test fold's expression distribution, which on
single-cell data is substantial because the dominant axis of variation is
technical.

Class weighting is on by default. NK functional states are strongly imbalanced
in any real sample, and an unweighted model on a 70/10/10/10 split can reach
70% accuracy while never predicting three of the four states.

### Why `marker_score` is in the list

A foundation model, or any learned model, that cannot beat reading the marker
panels directly has not earned its cost. Without the floor on the same axis,
that is invisible. `compare_models` reports `beats_marker_score_floor`
explicitly, and a `False` there is a result to be reported, not a bug.

On real data the floor is informative: `marker_score` reaches 0.478 macro-F1
against logistic regression's 0.966.

---

## Evaluation

Two levels, reported separately and never averaged together.

**Cell level** — accuracy, balanced accuracy, macro-F1, weighted F1, Cohen's κ,
macro one-vs-rest AUROC, and the full confusion matrix. Macro-F1 is the
headline rather than accuracy, because accuracy on a sample that is 70% one
state rewards a model that only ever predicts that state. Majority-class
accuracy is reported alongside as the chance level.

**Donor level** — the model is scored per held-out donor and the spread across
donors is reported. This is the number that predicts behaviour on the next
donor. Donors with fewer than 10 cells are excluded and **listed**: a macro-F1
over four cells is noise, and averaging it in lets the smallest donors dominate
the variance estimate. Population standard deviation is used, since the
quantity describes the donors observed and with 2–5 held-out donors the sample
correction is a large and arbitrary inflation.

`cell_minus_donor_macro_f1` is reported for every result. It is the quantity a
cell-random split is designed not to reveal.

AUROC is `None`, not a number, when a class has no positive example in the
evaluation set — one-vs-rest AUROC is undefined there. Similarly, an undefined
Cohen's κ comes back as NaN from scikit-learn and is serialised as JSON `null`.

---

## Interpretation

A cell-state classifier reporting only macro-F1 is unfalsifiable as biology.
The question is whether the genes the model relies on are the genes that should
define these states. A classifier separating NK states by ribosomal and
mitochondrial content has learned library size, and its F1 will not say so.

**Differential expression.** One-vs-rest Wilcoxon rank-sum tests with
Benjamini–Hochberg FDR control. Rank-sum rather than a t-test because
log-normalised single-cell expression is zero-inflated and far from normal;
this also matches `scanpy.tl.rank_genes_groups`'s default, keeping results
comparable to the standard workflow. Fold changes are computed on `expm1` of
the log-normalised values, so a reported log2 fold change is a ratio of mean
normalised expression rather than a difference of log means — the latter is not
a fold change of anything.

**Marker enrichment.** Hypergeometric over-representation (one-sided) of a
class's panel in its own ranked gene list. The universe is the set of genes
actually measured and available to the model, **not** the full transcriptome:
testing against ~20000 genes when the model saw 2000 HVGs inflates every
p-value, because HVG selection has already enriched for informative genes.

`model_gene_enrichment` is the stronger version — it tests the genes the fitted
model actually weights. Unsigned importances shared across classes (tree
ensembles give one vector for all classes) cannot be attributed to a single
state and are reported as untestable rather than tested against an arbitrary
panel.

Genes with the prefixes `MT-`, `RPS`, `RPL`, `MRPS`, `MRPL`, `HB` are flagged
as technical and reported. They are never silently filtered.

---

## scGPT integration

scGPT (Cui et al., *Nat Methods* 21:1470, 2024) is a transformer pretrained on
33 million cells. Fine-tuning it requires the `scgpt` package, a pretrained
checkpoint (~1–2 GB, distributed by the authors under their own terms), the
checkpoint's gene vocabulary, and a GPU.

None of that is available here, so this module implements the parts that can be
validated without weights — the **input transformation**, which is where the
errors that silently ruin a fine-tune actually live — and constructs the
documented command line for the parts that cannot.

Three transformations, all tested:

1. **Vocabulary mapping.** Genes outside the checkpoint's vocabulary have no
   embedding and are invisible to the model. Dropped **marker** genes are
   reported separately from dropped background genes, because losing GNLY from
   a cytotoxic-state task is fatal and losing an anonymous background gene is
   not; a single coverage percentage would hide the difference.

2. **Expression binning.** scGPT discretises expression into value bins
   computed **per cell** over **non-zero values only**. Both details matter:
   per-cell so the token encodes where a gene sits in that cell's own
   distribution rather than depending on library size, and non-zero-only
   because in a typical cell ~90% of genes are zero — including them would put
   almost every gene in the lowest bin and compress all signal into the top few.

3. **Sequence truncation.** The input is fixed-length, so genes are ranked by
   mean expression and truncated. Marker genes are **pinned**, because they are
   often not among the most highly expressed and truncating them away leaves
   the model classifying functional states without seeing the genes that
   define them.

With no checkpoint configured, the stage reports
`IMPLEMENTED_NOT_FULLY_EXECUTED`, the command it would run, and why it did not.
It never produces a transformer metric. See [SCGPT.md](SCGPT.md).

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | bad input or invalid configuration |
| 3 | a required optional dependency is missing |
| 4 | an external tool (scGPT) is unavailable or failed |
| 5 | **data leakage detected** |

Code 5 is distinct deliberately: a leaking split is not a crash and not a bad
config, it is a scientifically invalid run, and a caller in a batch script
should be able to tell that case apart without parsing a message.

Status messages go to **stderr** and JSON to **stdout**, so
`nkstate run -c config.yaml | jq .comparison` works and redirecting stdout
gives a valid JSON document with no log lines mixed in.
