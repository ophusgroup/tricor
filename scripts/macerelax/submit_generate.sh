#!/bin/bash
# SLURM batch script for student-model structure generation on Perlmutter.
#
# Layout: 1 node × 4 GPUs by default.  Each rank picks up a round-robin
# slice of the CIF list (handled inside generate_with_student.py via
# LOCAL_RANK / WORLD_SIZE), so 4 GPUs ≈ 4× throughput.
#
# The script is RESUME-SAFE: each rank checks OUTPUT_ROOT for existing
# XYZ files and skips trajectories already done.  Killing + restarting
# this job (or auto-chaining via --dependency=singleton, see below) just
# picks up where the previous run left off — no manual bookkeeping.
#
# Launch pattern:
#   * --ntasks-per-node=1   — one SLURM task; torchrun forks 4 worker procs
#   * --gpus-per-node=4     — all 4 A100 80 GB GPUs available to torchrun
#   * --gpu-bind=none       — disable Perlmutter's default per-task GPU mask
#   * --cpus-per-task=64    — whole node's CPUs for the per-rank dataloaders
#
# To scale to multiple nodes, set --nodes=N below.  Throughput scales
# linearly since CIF slices are disjoint across world-size ranks.

#SBATCH --job-name=macegen_v1
#SBATCH --nodes=1                              # → CHANGE to N for multi-node
#SBATCH --qos=regular                          # max 24 h; switch to "preempt" for ≤72 h
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=64
#SBATCH -A m5241
#SBATCH -C "gpu&hbm80g"
#SBATCH -t 12:00:00                            # adjust per corpus size; see throughput notes
#SBATCH --signal=B:USR1@300                    # send SIGUSR1 5 min before walltime
#SBATCH --output=logs/gen-%j.out
#SBATCH --error=logs/gen-%j.err
# Optional auto-chain: uncomment to resubmit this same script when the
# current job ends.  Combined with the skip-if-done logic in the
# generation script, this gives unattended full-corpus runs across
# multiple 12 h slots.
##SBATCH --dependency=singleton

set -euo pipefail

SCRIPT=/global/u2/e/ehrdt/tricor/scripts/macerelax/generate_with_student.py

# ── torchrun rendezvous ─────────────────────────────────────────────────────
# Single-node: localhost is fine.  Multi-node: scontrol resolves nodelist.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501                       # different from train (29500) to avoid collisions

# ── Threading ───────────────────────────────────────────────────────────────
# 4 ranks × NUM_THREADS=4 in the script's CONFIG = 16 OMP threads, well
# under the 64 cores allocated.  generate_with_student.py also pins
# OMP/MKL/etc. itself; this is belt-and-braces.
export OMP_NUM_THREADS=4

# ── CUDA allocator ──────────────────────────────────────────────────────────
# expandable_segments handles the variable-shape edge tensors across the
# CIF distribution; garbage_collection_threshold:0.8 triggers proactive
# cache release at 80% memory.  Fe2N-like dense systems peak ~76 GB with
# EDGE_CHUNK_SIZE=2_000_000 so the headroom matters.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# ── NCCL tuning (no-ops for round-robin generation but cost nothing) ────────
# torchrun sets up NCCL even though generate_with_student.py doesn't
# actually communicate between ranks.  Keep the standard Slingshot
# tuning in case a future multi-node version needs it.
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=0
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DEFAULT_CQ_SIZE=131072
export NCCL_DEBUG=WARN

mkdir -p logs

module load conda
conda activate /global/common/software/m5020/ehrdt/tricor/

echo "=== job $SLURM_JOB_ID  starting $(date) ==="
echo "  nodes        : $SLURM_JOB_NUM_NODES"
echo "  gpus per node: $SLURM_GPUS_PER_NODE"
echo "  total ranks  : $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "  master       : $MASTER_ADDR:$MASTER_PORT"
echo "  nodelist     : $SLURM_JOB_NODELIST"
echo "  script       : $SCRIPT"
echo

srun -l torchrun \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --nproc-per-node="$SLURM_GPUS_PER_NODE" \
     --rdzv-backend=c10d \
     --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     "$SCRIPT"

echo
echo "=== job $SLURM_JOB_ID  finished $(date) ==="
