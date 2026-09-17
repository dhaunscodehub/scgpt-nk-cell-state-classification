#!/bin/bash
#SBATCH --job-name=nkstate-scgpt
#SBATCH --output=logs/scgpt_%j.out
#SBATCH --error=logs/scgpt_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# scGPT fine-tuning.
#
# This script is provided for completeness and HAS NOT BEEN EXECUTED. It
# requires a pretrained checkpoint that the scGPT authors distribute under
# their own terms and which is not redistributed by this repository. No
# transformer metric appears anywhere in this repository. See docs/SCGPT.md.
#
#   sbatch hpc/slurm/finetune_scgpt.sbatch configs/your_config.yaml
#
# GPU memory: a 1200-gene sequence at batch size 32 needs roughly 24 GB for a
# full fine-tune. With freeze_encoder: true (a linear probe, and usually the
# honest choice on a few thousand cells) 16 GB is enough. Lower --batch-size
# before lowering --max-seq-len: truncating the input discards genes, which
# changes what the model can see, whereas a smaller batch only changes the
# optimisation.

set -euo pipefail

CONFIG="${1:?usage: sbatch finetune_scgpt.sbatch <config.yaml>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

module load anaconda3 cuda 2>/dev/null || true
source activate nkstate 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Verify the checkpoint and the gene vocabulary BEFORE requesting hours of GPU
# time. If marker genes are absent from the vocabulary the model cannot see
# them, and the run is compromised before the first step.
nkstate scgpt -c "$CONFIG" > "results/scgpt_plan_${SLURM_JOB_ID:-local}.json"
nkstate tokenise -c "$CONFIG" > "results/scgpt_tokens_${SLURM_JOB_ID:-local}.json"

COMMAND=$(python -c "
import json, sys
plan = json.load(open('results/scgpt_plan_${SLURM_JOB_ID:-local}.json'))
if not plan['checkpoint']['available']:
    sys.exit('no scGPT checkpoint configured; see docs/SCGPT.md')
print(plan['finetune_command']['command'])
")

echo "running: $COMMAND"
eval "$COMMAND"
