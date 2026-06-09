#!/bin/bash
# Single-node, 4-GPU DDP training on Perlmutter.
#
# Submit:
#   cd $HOME/tricor
#   sbatch scripts/macerelax/train_perlmutter.sbatch
#
# For a debug-QoS smoke (30 min, 1 node):
#   sbatch --qos=debug --time=00:30:00 \
#       scripts/macerelax/train_perlmutter.sbatch

#SBATCH --job-name=macerelax_train
#SBATCH --account=m5241
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
module load conda
conda activate tricor

# Absolute path to python — sidesteps PATH issues under srun/DDP.
PYTHON_BIN=/global/common/software/m5020/ehrdt/tricor/bin/python

# Allocator tuning — bf16-mixed training in DDP can fragment the cache.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DDP networking — Lightning uses these to set up the process group.
# IB on Perlmutter is the right backend; gloo is the fallback.
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN              # set to INFO for connectivity debugging

mkdir -p logs

echo "=== job $SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES  $(date) ==="
echo "  conda env : $CONDA_DEFAULT_ENV"
echo "  cwd       : $(pwd)"
echo "  host      : $(hostname)"
echo "  python    : $PYTHON_BIN"
echo "  GPUs      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo

# ── Launch ──────────────────────────────────────────────────────────────────
# IMPORTANT: --ntasks-per-node MUST match NUM_GPUS in train_perlmutter.py.
# Each rank is one task and Lightning's DDP assigns it one GPU.
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=none \
    $PYTHON_BIN $HOME/tricor/scripts/macerelax/train_perlmutter.py

echo
echo "=== job $SLURM_JOB_ID  finished $(date) ==="
