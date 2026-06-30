#!/bin/bash
# SLURM batch script for the MACE multi-crop energy enrichment pass on Perlmutter.
#
# Layout: 1 GPU node × 4 A100 80 GB by default.  Each rank loads MACE-MPA-0
# on its own GPU and processes a round-robin slice of trajectories.  Per-rank
# output is mace_enrichment.rank{N}.csv under OUTPUT_ROOT; build_dataset_table.py
# globs them automatically.
#
# Cost model:
#   ~4 crops × ~7 s/crop = 30 s per trajectory at production cell size.
#   14k trajectories × 30 s / 16 GPUs (4 nodes × 4 A100) ≈ 7.3 wall-hours.
#   On 1 node × 4 A100, the same corpus is ~30 wall-hours — needs multiple
#   12h slots.  The resume logic + --dependency=singleton (uncomment below)
#   chains slots automatically.
#
# Launch pattern (mirrors submit_generate.sh):
#   --ntasks-per-node=4   — one SLURM task per GPU
#   --gpus-per-node=4     — all 4 A100s
#   --gpu-bind=none       — each task sees all GPUs; the script pins via
#                           torch.cuda.set_device(_LOCAL_RANK)
#   --cpus-per-task=16    — 64-core node / 4 ranks for MACE worker loaders
#
# RESUME-SAFE: each rank reads its own mace_enrichment.rank{N}.csv at start
# and skips trajectories already in there.  Killing + restarting picks up
# where the previous run left off.

#SBATCH --job-name=mace_enrich
#SBATCH --nodes=1                              # → CHANGE to N for multi-node (linear speedup)
#SBATCH --qos=regular                          # max 12 h; switch to "preempt" for ≤72 h
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=16
#SBATCH -A m5241
#SBATCH -C "gpu&hbm80g"
#SBATCH -t 12:00:00                            # full corpus → use --dependency=singleton + resubmit
#SBATCH --signal=B:USR1@300                    # send SIGUSR1 5 min before walltime
#SBATCH --output=logs/mace-enrich-%j.out
#SBATCH --error=logs/mace-enrich-%j.err
# Optional auto-chain: uncomment to resubmit this same script when the
# current job ends.  Combined with the skip-if-done logic in the
# enrichment script, this gives unattended full-corpus passes across
# multiple 12 h slots.
##SBATCH --dependency=singleton

set -euo pipefail

SCRIPT=/global/u2/e/ehrdt/tricor/scripts/macerelax/enrich_mace_energy.py

# ── Threading ──────────────────────────────────────────────────────────────
# 4 ranks × 16 cores/rank = 64 cores (full node).  enrich_mace_energy.py
# uses os.environ.setdefault for OMP/MKL/etc., so what we export here wins.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── CUDA allocator ─────────────────────────────────────────────────────────
# Same settings as submit_generate.sh — MACE-MPA-0 holds ~12 GB resident,
# clusters are 10-25 k atoms, expandable_segments handles the varying cluster
# size across the trajectory distribution.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

mkdir -p logs

module load conda
conda activate /global/common/software/m5020/ehrdt/tricor/

echo "=== job $SLURM_JOB_ID  starting $(date) ==="
echo "  nodes        : $SLURM_JOB_NUM_NODES"
echo "  gpus per node: $SLURM_GPUS_PER_NODE"
echo "  total ranks  : $SLURM_NTASKS"
echo "  cores per rank: $SLURM_CPUS_PER_TASK"
echo "  nodelist     : $SLURM_JOB_NODELIST"
echo "  script       : $SCRIPT"
echo

# Each srun task becomes one rank.  enrich_mace_energy.py reads
# SLURM_LOCALID / SLURM_PROCID / SLURM_NTASKS in _detect_rank_from_env(),
# pins to GPU LOCAL_RANK via torch.cuda.set_device, and processes its
# round-robin slice of the trajectory file list.
# -l prefixes each line of output with the rank, so log greps work:
#     grep "^ 2:" logs/mace-enrich-12345.out   # only rank 2
srun -l python "$SCRIPT"

echo
echo "=== job $SLURM_JOB_ID  finished $(date) ==="
