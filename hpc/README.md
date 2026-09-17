# HPC

Two SLURM scripts. Both resolve the repository root from their own location, so
they work from any submission directory.

## `run_pipeline.sbatch` — the baseline pipeline

```bash
sbatch hpc/slurm/run_pipeline.sbatch configs/pbmc_reference.yaml
```

CPU only, and deliberately so: the baselines are scikit-learn models and a GPU
allocation would sit idle. 8 cores / 32 GB / 2 h is comfortable for ~50k cells.
Memory is dominated by the dense expression matrix and scales roughly linearly
in cells.

BLAS thread counts are pinned to `$SLURM_CPUS_PER_TASK`. scikit-learn already
parallelises across the allocation, and nested threading measurably slows the
job.

The script runs `scripts/validate_controls.py` **first** and fails the job if
any control fails. If the negative control scores above chance, nothing the
pipeline reports afterwards is trustworthy, so there is no point spending the
rest of the allocation.

## `finetune_scgpt.sbatch` — scGPT fine-tuning

**This script has not been executed.** It needs a pretrained checkpoint the
scGPT authors distribute under their own terms, which is not redistributed
here. No transformer metric appears anywhere in this repository. See
[../docs/SCGPT.md](../docs/SCGPT.md).

It validates the checkpoint and the gene vocabulary before consuming GPU hours:
if marker genes are missing from the vocabulary, the model cannot see them and
the run is compromised before the first optimisation step.

GPU memory: ~24 GB for a full fine-tune at `max_seq_len=1200`, batch 32; ~16 GB
with `freeze_encoder: true`. Lower `--batch-size` before `--max-seq-len` —
truncating the input discards genes and changes what the model can see, while a
smaller batch only changes the optimisation.

## Adapting to another scheduler

Both scripts are ordinary bash. The SLURM-specific parts are the `#SBATCH`
directives and `$SLURM_CPUS_PER_TASK` / `$SLURM_JOB_ID`, each of which falls
back to a sensible default (`1` and `local`) when unset, so the scripts also
run directly:

```bash
bash hpc/slurm/run_pipeline.sbatch configs/pbmc_reference.yaml
```
