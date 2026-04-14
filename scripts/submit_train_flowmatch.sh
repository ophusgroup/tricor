#!/bin/bash
#SBATCH -A m5241
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J flowmatch-si3n4
#SBATCH -o logs/flowmatch-%j.out
#SBATCH -e logs/flowmatch-%j.err

set -euo pipefail

module load conda
conda activate /global/common/software/m5020/ehrdt/tricor

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=cores

REPO=/pscratch/sd/e/ehrdt/tricor
cd "$REPO"

mkdir -p scripts/logs

srun python scripts/train_flowmatch.py
