#!/bin/bash
#SBATCH --job-name=nkstate
#SBATCH --output=logs/nkstate_%j.out
#SBATCH --error=logs/nkstate_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

# Baseline pipeline. CPU only: the baselines are scikit-learn models and a GPU
# would sit idle. Request one only for the scGPT stage (see finetune_scgpt.sbatch).
#
#   sbatch hpc/slurm/run_pipeline.sbatch configs/pbmc_reference.yaml
#
# Memory: the dense expression matrix dominates. 32G is comfortable for
# ~50k cells x 2000 genes in float64; scale roughly linearly in cells.

set -euo pipefail

CONFIG="${1:?usage: sbatch run_pipeline.sbatch <config.yaml>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

module load anaconda3 2>/dev/null || true
source activate nkstate 2>/dev/null || true

# Keep BLAS from oversubscribing: scikit-learn already parallelises across the
# allocated cores, and nested threading slows the job down measurably.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host:    $(hostname)"
echo "config:  $CONFIG"
echo "started: $(date -Iseconds)"

# Controls first. If the negative control scores above chance, nothing the
# pipeline reports afterwards is trustworthy, so fail the job here.
python scripts/validate_controls.py --json "results/controls_${SLURM_JOB_ID:-local}.json"

nkstate run -c "$CONFIG" \
  > "results/run_${SLURM_JOB_ID:-local}.json"

echo "finished: $(date -Iseconds)"
