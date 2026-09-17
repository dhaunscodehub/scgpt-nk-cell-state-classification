# Using scGPT with this repository

The scGPT stage is **implemented but not executed here**. This document states
exactly what is missing, why, and how to run it yourself.

## What is missing and why

| Requirement | Status | Reason |
|---|---|---|
| `scgpt` Python package | not installed | Optional extra: `pip install -e ".[scgpt]"` |
| Pretrained checkpoint (~1–2 GB) | **not redistributed** | The scGPT authors distribute the weights themselves, under their own terms. Redistributing them here would strip them from those terms. |
| Gene vocabulary (`vocab.json`) | not redistributed | Ships inside the checkpoint. |
| GPU | not available | A 1200-gene sequence per cell over a ~50M-parameter encoder is not practical on CPU. |

Because of this, **no transformer metric appears anywhere in this repository**.
The baseline stack is what actually runs, and it is reported as such.

## What *is* implemented and tested

31 tests cover the input transformation — the part where silent errors live:

* `map_to_vocabulary` — reports coverage and, separately, which **marker** genes
  were dropped.
* `bin_expression` — per-cell, non-zero-only quantile binning, matching scGPT's
  scheme. Rejects negative (scaled) input.
* `tokenise` — ranks genes, truncates to `max_seq_len`, and **pins** marker
  genes so the task's own signal is not truncated away.
* `build_finetune_command` — constructs the `cell_annotation` invocation.
* `checkpoint_report` — reports what is present and what is missing, never
  assumes availability.

Inspect them without any weights:

```bash
nkstate scgpt -c configs/pbmc_reference.yaml --dry-run
```

## Running the fine-tune yourself

1. Install the extra:

```bash
pip install -e ".[scgpt]"
```

2. Obtain a checkpoint from the scGPT repository
   (<https://github.com/bowang-lab/scGPT>) under the authors' terms. The
   `whole-human` checkpoint matches the cell-annotation tutorial. You need a
   directory containing `args.json`, `vocab.json` and `best_model.pt`.

3. Point the config at it:

```yaml
scgpt:
  checkpoint_dir: /path/to/scGPT_human
  vocabulary_path: /path/to/scGPT_human/vocab.json
  max_seq_len: 1200
  n_bins: 51
  freeze_encoder: true
```

4. Check the vocabulary before spending GPU time. This is worth doing first —
   if marker genes are missing from the vocabulary, the model cannot see them
   and the run is compromised before it starts. Substitute your own config
   path for `MY_CONFIG` throughout; no such file ships with the repository:

```bash
MY_CONFIG=configs/pbmc_reference.yaml   # your config, with scgpt.* filled in
nkstate tokenise -c "$MY_CONFIG" | jq '.n_pinned_marker_genes, .seq_len'
```

5. Build and inspect the command, then run it:

```bash
nkstate scgpt -c "$MY_CONFIG" | jq -r .finetune_command.command
```

## A note on `freeze_encoder`

With a few thousand cells, `freeze_encoder: true` (a linear probe) is usually
the honest choice. A full fine-tune of a ~50M-parameter encoder on 2000 cells
overfits, and the resulting number says more about the fine-tuning set than
about the model. The config default is `true` for that reason.

## Attribution

scGPT is the work of its authors:

> Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024).
> scGPT: toward building a foundation model for single-cell multi-omics using
> generative AI. *Nature Methods*, 21, 1470–1480.

This repository contains **no scGPT source code and no scGPT weights**. It
contains an integration layer that calls the `scgpt` package's public entry
point. The tokenisation reimplements the documented input scheme so it can be
tested independently; it is not copied from the scGPT codebase.
