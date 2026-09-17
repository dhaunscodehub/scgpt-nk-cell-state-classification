# Data

**No datasets are committed to this repository.** Everything is downloaded on
demand into `data/`, which is gitignored. This document states exactly what is
fetched, from where, and under what terms.

## Reference dataset: pbmc3k

| | |
|---|---|
| Identifier | `scanpy.datasets.pbmc3k_processed()` |
| Origin | 10x Genomics, "3k PBMCs from a Healthy Donor" |
| Citation | Zheng, G. X. Y., et al. (2017). Massively parallel digital transcriptional profiling of single cells. *Nature Communications*, 8, 14049. |
| Size | 2638 cells × 13714 genes (after the authors' QC) |
| Download | ~23.5 MB, cached in `data/scanpy/` |
| Labels | 8 curated cell types, annotated by the scanpy maintainers from Louvain clusters |
| Donors | **1** |
| Terms | 10x Genomics public datasets; freely redistributable for research |

Fetched automatically by:

```bash
nkstate run -c configs/pbmc_reference.yaml
```

### Two things to know before using it

**It is a single donor.** A donor-grouped split is impossible on this dataset,
and the code refuses to pretend otherwise: `donor_split` raises rather than
degenerating into a cell-level split. Metrics from this dataset describe
within-donor separability only. Given that a cell-random split can overstate
balanced accuracy by 0.403 on confounded data (see
[VALIDATION.md](../docs/VALIDATION.md) control 3), they are not an estimate of
performance on a new donor.

**The distributed `.X` is z-scored, not log-normalised.** It has been scaled to
unit variance and clipped at 10, over only 1838 highly variable genes, and
contains negative values. This repository takes `.raw` instead — log1p of
CPM-normalised counts over all 13714 genes — and records the substitution in
`uns["nkstate_provenance"]`. Using `.X` would produce plausible-looking
nonsense: fold changes computed on z-scores are not fold changes, and marker
scoring against a scaled matrix compares genes whose scales have already been
equalised.

## Synthetic cohorts

Generated in memory by `nkstate.testing`, never written to disk, fully
determined by a seed. Counts are drawn from a negative binomial
(dispersion 0.5), the standard model for scRNA-seq overdispersion.

Controllable knobs:

| Knob | Effect |
|---|---|
| `state_effect` | How recoverable the biology is. `0.0` = unlearnable (negative control), `0.8` = strongly separable (positive control). |
| `donor_effect` | Per-donor multiplicative expression shift — a batch effect. |
| `donor_confound` | Makes each donor's state composition differ, so a cell-random split can exploit donor identity. |

These supply the multi-donor structure pbmc3k cannot, and they are the only
place in this repository where ground truth is known exactly. Every number
derived from them is a statement about the code, not about NK biology.

## Bringing your own data

```yaml
data:
  source: h5ad
  path: ../data/my_cohort.h5ad
  donor_key: donor_id      # REQUIRED for a donor-grouped split
  label_key: cell_type     # curated labels, if you have them
```

Requirements:

* `.X` must hold **raw counts** or log-normalised values. If already
  normalised, set `uns["nkstate_normalisation"] = {"log1p": True}` so
  `normalise()` does not log-transform a second time — it checks, and a second
  log1p flattens every fold change while leaving the data looking intact.
* `obs[donor_key]` must identify the biological donor, not the sequencing run.
  Splitting on a technical batch id while calling it a donor split gives false
  assurance: if one donor spans several batches, a "batch-grouped" split still
  shares that donor across folds.
* At least **3 donors** for a donor-grouped split; the code refuses fewer with
  an explanatory message.
* Gene symbols should be HGNC symbols. The marker panels use symbols, not
  Ensembl ids, and `nkstate markers -c your_config.yaml` reports panel coverage
  so a symbol-convention mismatch is visible before you train anything.

## scGPT checkpoints

Not downloaded, and not redistributable here — the scGPT authors distribute the
weights under their own terms. See [SCGPT.md](../docs/SCGPT.md). With no
checkpoint configured, the scGPT stage reports what it would run and why it
did not, and no transformer metric is produced.
