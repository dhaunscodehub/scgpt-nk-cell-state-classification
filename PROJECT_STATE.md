# Project state

**Status:** `IMPLEMENTED_NOT_FULLY_EXECUTED`

The baseline stack, evaluation, interpretation and control experiments all
execute end to end on real and synthetic data and are covered by 300 passing
tests. The scGPT fine-tuning stage is implemented and tested at the input-
transformation level but cannot be executed here: it needs a pretrained
checkpoint the authors distribute under their own terms plus a GPU.

## What runs

| Component | Status | Evidence |
|---|---|---|
| Marker panels and alias resolution | `IMPLEMENTED_AND_TESTED` | 22 tests |
| Control-corrected panel scoring | `IMPLEMENTED_AND_TESTED` | 16 tests |
| Donor-grouped splitting and leakage guards | `IMPLEMENTED_AND_TESTED` | 17 tests |
| Baseline classifiers (4) | `IMPLEMENTED_AND_TESTED` | 22 tests |
| Cell- and donor-level evaluation | `IMPLEMENTED_AND_TESTED` | 26 tests |
| DE and marker enrichment | `IMPLEMENTED_AND_TESTED` | 28 tests |
| Configuration and CLI | `IMPLEMENTED_AND_TESTED` | 44 tests |
| Control experiments | `IMPLEMENTED_AND_TESTED` | 8/8 passing |
| scGPT input transformation | `IMPLEMENTED_AND_TESTED` | 31 tests |
| scGPT fine-tuning | `IMPLEMENTED_NOT_FULLY_EXECUTED` | no weights, no GPU |

## Headline measurements

All `VERIFIED_REPRODUCED`; see [docs/VALIDATION.md](docs/VALIDATION.md).

* **Donor-confound optimism: 0.403 balanced accuracy.** On a cohort with no
  biological state signal, a cell-random split reaches 0.655 while a
  donor-held-out split reaches 0.252 (chance 0.250), averaged over 5 seeds.
* **Real-data cell typing: macro-F1 0.966** (logistic regression, 8 classes,
  curated pbmc3k labels), against a no-fitting marker floor of 0.478 and a
  majority-class accuracy of 0.428.
* **300 tests, 38/38 smoke checks, 8/8 controls.**

## Known limitations

1. **No multi-donor real dataset.** Donor-grouped evaluation is demonstrated on
   synthetic cohorts, where ground truth is known by construction. The
   real-data run is single-donor (pbmc3k) and therefore uses a cell-random
   split, which — by this repository's own control experiment — cannot estimate
   generalisation to a new donor. This is the most significant gap.

2. **NK functional-state accuracy is not measured on real cells.** No public
   dataset used here carries curated resting/activated/cytotoxic/exhausted
   labels. Marker-derived labels are tagged `marker_pseudo_label` throughout: a
   model scored against them measures agreement with the marker rule, not
   accuracy. The real-data task is therefore **cell typing**, where the labels
   are curated and independent.

3. **No scGPT metric.** Not a single transformer number appears in this
   repository. The integration layer is real and tested; the fine-tune is not
   run. See [docs/SCGPT.md](docs/SCGPT.md).

4. **Lineage panels do not resolve subtypes.** Marker enrichment fails for CD8
   T cells and FCGR3A+ monocytes because each shares its panel with a sibling
   subtype and enrichment is computed one-vs-rest. This is reported as a
   finding, not patched by widening the panels.

5. **Synthetic model comparison is tautological.** The generator plants state
   signal on the marker-panel genes, so `marker_score` is optimal there by
   construction. Only the real-data comparison is informative about models.

## Next steps, in order of value

1. **Acquire a multi-donor NK dataset** with per-donor identifiers — e.g. a
   healthy-donor PBMC panel or a tumour-infiltrating NK cohort — and run the
   existing donor-grouped pipeline unchanged. Everything needed is already
   implemented; this is a data problem, not a code problem, and it would
   convert limitation 1 into a measurement.
2. **Obtain curated functional-state annotations** (or FACS-sorted
   populations) to replace pseudo-labels for the state task, which would make
   state accuracy reportable as accuracy.
3. **Run the scGPT fine-tune** on a GPU with a downloaded checkpoint, and
   compare it against the baselines on the same donor-grouped split. Check
   vocabulary coverage first (`nkstate tokenise`): a marker gene absent from
   the vocabulary compromises the run before it starts.
4. **Add subtype-resolution panels** (CD8A/GZMK, FCGR3A/MS4A7) as a *separate*
   panel set rather than widening the lineage panels, so both resolutions can
   be tested independently.

## Reproducing

```bash
pip install -e ".[all]"
python scripts/smoke_test.py                                   # 38/38
python scripts/validate_controls.py --json results/controls.json  # 8/8
python -m pytest tests/                                        # 300 passed
nkstate run -c configs/pbmc_reference.yaml
nkstate run -c configs/synthetic_donor_split.yaml
```

Committed reference outputs are under `results/reference/`, and every run
records its own `provenance` block with package versions, because scanpy's HVG
selection and `score_genes` defaults have changed between minor releases.

Environment used for the reported numbers: Python 3.14.6, numpy 2.5.1,
scipy 1.18.0, pandas 3.0.3, scikit-learn 1.9.0, scanpy 1.12.4,
anndata 0.13.3.post0, macOS arm64.
