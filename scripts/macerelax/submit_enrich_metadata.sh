#!/bin/bash
# SLURM batch script for the cheap-enrichment pass on Perlmutter.
#
# Layout: 1 CPU node × 16 ranks × 8 threads/rank = 128 cores total (one whole
# Perlmutter CPU node).  Each rank picks up a round-robin slice of the
# trajectory list — the script handles partitioning via SLURM_LOCALID /
# SLURM_PROCID / SLURM_NTASKS in its rank-detection helper.  Per-rank output
# is enrichment.rank{N}.csv under OUTPUT_ROOT; build_dataset_table.py globs
# them automatically.
#
# Why CPU node and not GPU node:
#   The pass is pure scipy.cKDTree + numba g3 + matplotlib.  No CUDA.
#   CPU nodes are cheaper, more available, and the workload is embarrassingly
#   parallel at the trajectory level.
#
# Why 16×8 and not 4×32 / 64×2:
#   Numba's g3 kernel uses prange over origin atoms, so per-call thread
#   scaling saturates around ~8 threads for typical n_origin sample sizes.
#   Going wider per-rank (32 threads) gives diminishing g3 speedup; going
#   narrower (2 threads) underuses the cores.  16 ranks × 8 threads keeps
#   both the file-level and intra-call parallelism balanced.  Override below
#   if your trajectory size distribution suggests otherwise.
#
# RESUME-SAFE: each rank reads its existing enrichment.rank{N}.csv at start
# and skips trajectories already in there.  Killing + restarting picks up
# where the previous run left off — no manual bookkeeping.

#SBATCH --job-name=enrich_meta
#SBATCH --nodes=1                               # → CHANGE to N for multi-node (linear speedup)
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --qos=regular                           # max 12 h on CPU; switch to "preempt" for ≤24 h
#SBATCH -A m5241
#SBATCH -C cpu
#SBATCH -t 04:00:00                             # ~14k trajs × 8 s / 16 ranks ≈ 2 h; pad for variance
#SBATCH --output=logs/enrich-meta-%j.out
#SBATCH --error=logs/enrich-meta-%j.err
# Optional auto-chain: uncomment to resubmit this same script when the
# current job ends.  Combined with the skip-if-done logic in the
# enrichment script, this gives unattended full-corpus passes across
# multiple time slots.
##SBATCH --dependency=singleton

set -euo pipefail

SCRIPT=/global/u2/e/ehrdt/tricor/scripts/macerelax/enrich_metadata.py

# ── Threading ──────────────────────────────────────────────────────────────
# Override enrich_metadata.py's NUM_THREADS=4 default via env vars.  The
# script uses os.environ.setdefault(), which respects anything we set here.
# 8 threads/rank × 16 ranks = 128 cores, the whole Perlmutter CPU node.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Matplotlib ─────────────────────────────────────────────────────────────
# Headless backend — Perlmutter compute nodes have no display.  The script
# also forces Agg internally; this is belt-and-braces.
export MPLBACKEND=Agg

mkdir -p logs

module load conda
conda activate /global/common/software/m5020/ehrdt/tricor/

echo "=== job $SLURM_JOB_ID  starting $(date) ==="
echo "  nodes         : $SLURM_JOB_NUM_NODES"
echo "  ranks per node: $SLURM_NTASKS_PER_NODE"
echo "  cores per rank: $SLURM_CPUS_PER_TASK"
echo "  total ranks   : $SLURM_NTASKS"
echo "  total cores   : $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "  nodelist      : $SLURM_JOB_NODELIST"
echo "  script        : $SCRIPT"
echo

# Each srun task becomes one rank.  The Python script reads
# SLURM_LOCALID / SLURM_PROCID / SLURM_NTASKS to partition the file list.
# -l prefixes each line of output with the rank, so log greps work:
#     grep "^ 3:" logs/enrich-meta-12345.out   # only rank 3
srun -l python "$SCRIPT"

echo
echo "=== job $SLURM_JOB_ID  finished $(date) ==="
