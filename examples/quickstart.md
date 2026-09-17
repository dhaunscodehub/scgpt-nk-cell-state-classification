# Quickstart

## Install

```bash
pip install -e ".[all]"
```

## 1. Check the installation works

```bash
python scripts/smoke_test.py
```

38 checks, under a minute, no network access. Expect `38/38 checks passed`.

## 2. Run the control experiments

```bash
python scripts/validate_controls.py
```

This is the step worth understanding before looking at any metric:

```
1   PASS   negative control: no state signal, donor-groupe      0.247    0.250
2   PASS   positive control: strong state signal, donor-gr      1.000    0.250
3   PASS   donor confound, cell-random split: scores above      0.655    0.250
4   PASS   donor confound, donor-held-out split: falls bac      0.252    0.250
5   PASS   cell-random splitting overstates accuracy by a       0.403    0.250
6   PASS   single-donor donor-grouped split is refused              -        -
7   PASS   marker pseudo-labels recover the generating sta      1.000        -
8   PASS   leakage guard rejects a corrupted donor split            -        -
```

Rows 3 and 4 use the **same data**, in which no gene carries information about
the state label. The cell-random split scores 0.655; the donor-held-out split
scores 0.252 against chance 0.250. The 0.403 gap is donor identity being
mistaken for biology.

If row 1 ever scores above chance, stop — something is leaking and nothing else
the pipeline reports is trustworthy.

## 3. Inspect the marker panels

```bash
nkstate markers | jq '.panels | keys'
```

Coverage against a real dataset, which is how you catch a gene-symbol
convention mismatch before training anything:

```bash
nkstate markers -c configs/pbmc_reference.yaml | jq '.coverage.cytotoxic'
```

## 4. Run on real data

```bash
nkstate run -c configs/pbmc_reference.yaml > result.json
```

Status goes to stderr, JSON to stdout:

```
[nkstate] 2638 cells, 2022 genes, 1 donors
[nkstate] logistic       cell macro-F1 0.966  donor-level: not computed
[nkstate] random_forest  cell macro-F1 0.955  donor-level: not computed
[nkstate] mlp            cell macro-F1 0.925  donor-level: not computed
[nkstate] marker_score   cell macro-F1 0.478  donor-level: not computed
[nkstate] warning: marker enrichment failed for ['CD8 T cells', ...]
```

Note `donor-level: not computed` — pbmc3k is a single donor, so donor-level
metrics are impossible and the code says so rather than reporting a number that
looks like generalisation.

Did learning beat simply reading the markers?

```bash
jq .comparison.beats_marker_score_floor result.json
# { "logistic": true, "random_forest": true, "mlp": true }
```

## 5. Get donor-level metrics

pbmc3k cannot support them. The synthetic config has 12 donors:

```bash
nkstate run -c configs/synthetic_donor_split.yaml 2>&1 >/dev/null | grep macro-F1
```

```
[nkstate] logistic       cell macro-F1 0.998  donor macro-F1 0.998 +/- 0.003
[nkstate] random_forest  cell macro-F1 0.993  donor macro-F1 0.992 +/- 0.011
[nkstate] marker_score   cell macro-F1 1.000  donor macro-F1 1.000 +/- 0.000
```

`marker_score` winning here is an **artefact**: the synthetic generator plants
the state signal on the marker-panel genes, so reading those panels is reading
the generative model. This config validates the donor-level metric plumbing,
not model quality. Section 4 on real data is the informative comparison.

## 6. Try to break it

The config rejects typos rather than ignoring them:

```bash
printf 'data:\n  n_top_gene: 500\n' > /tmp/bad.yaml
nkstate run -c /tmp/bad.yaml; echo "exit=$?"
# config error: [data] unknown key(s) ['n_top_gene']; valid keys are [...]
# exit=2
```

A donor-grouped split on one donor is refused:

```bash
python -c "
import numpy as np
from nkstate.data.splits import donor_split
donor_split(np.array(['d1']*100), test_fraction=0.25)
"
# SplitError: a donor-grouped split needs at least 3 donors, found 1 (['d1'])
```

## 7. Inspect the scGPT stage

No weights are needed to see what it would do:

```bash
nkstate scgpt -c configs/pbmc_reference.yaml --dry-run | jq '.checkpoint.available, .executed'
# false
# false
```

```bash
nkstate scgpt -c configs/pbmc_reference.yaml | jq -r .finetune_command.command
```

The command is constructed but never run here, and no transformer metric is
produced. See [../docs/SCGPT.md](../docs/SCGPT.md).
