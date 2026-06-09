#!/bin/bash
# SLURM batch script for pure-PyTorch DDP training on Perlmutter.
# Single-node by default; for multi-node change --nodes=N below.
#
# Launch pattern:
#   * --ntasks-per-node=1   — one SLURM task per node; torchrun forks 4 workers
#   * --gpus-per-node=4     — all 4 GPUs per node available
#   * --gpu-bind=none       — disable Perlmutter's default per-task GPU mask
#   * --cpus-per-task=64    — whole node's CPUs available to the dataloader pool
#
# torchrun sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_* in each worker's env,
# which train_perl_ddp.py's setup_distributed() reads.

#SBATCH --job-name=macetrain_ddp
#SBATCH --nodes=1                              # → CHANGE to N for multi-node
#SBATCH --qos=regular                          # "debug" allows ≤30 min, ≤2 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=64
#SBATCH -A m5241
#SBATCH -C "gpu&hbm80g"
#SBATCH -t 12:00:00                            # adjust walltime per epoch * MAX_EPOCHS
#SBATCH --signal=B:USR1@300                    # send SIGUSR1 5 min before walltime
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
# Optional auto-chain: enable by uncommenting next line.  Resubmits this same
# script when this job ends, picking up from last.pt automatically.
##SBATCH --dependency=singleton

set -euo pipefail

SCRIPT=/global/u2/e/ehrdt/tricor/scripts/macerelax/train_perl_ddp.py

# ── torchrun rendezvous ─────────────────────────────────────────────────────
# MASTER_ADDR = first node's hostname.  All ranks (on any node) torchrun-
# connect to this endpoint to bootstrap; NCCL traffic itself uses Slingshot.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# ── Threading ───────────────────────────────────────────────────────────────
# Each node has 4 ranks × NUM_WORKERS dataloader procs.  With 64 cores and
# NUM_WORKERS=4 in the script → 16 worker procs; 4 OMP threads each keeps
# total CPU use ≤ 64 cores.
export OMP_NUM_THREADS=4

# ── CUDA allocator ──────────────────────────────────────────────────────────
# expandable_segments handles the variable-shape graphs;
# garbage_collection_threshold:0.8 triggers proactive cache release at 80%
# memory, which helps avoid spikes on monster-cell batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# ── NCCL tuning for Perlmutter Slingshot ────────────────────────────────────
# Slingshot interface is hsn0 (Cassini NIC).  Tell NCCL to use it for
# inter-node collectives.  These are no-ops on single-node but cost nothing.
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=0              # use Slingshot/IB (not TCP loopback)
export NCCL_CROSS_NIC=1               # allow cross-NIC P2P (multi-node A100 boxes)
export NCCL_NET_GDR_LEVEL=PHB         # GPUDirect RDMA at PCI-host-bridge level
export FI_CXI_DEFAULT_CQ_SIZE=131072  # libfabric completion-queue size (avoids stalls)
export NCCL_DEBUG=WARN                # bump to INFO if multi-node hangs at init

mkdir -p logs

module load conda
conda activate /global/common/software/m5020/ehrdt/tricor/

# ── Auto-resume from last.pt if a previous job for this RUN_NAME exists ─────
# Reads RUN_NAME from train_perl_ddp.py and looks for last.pt in any prior run
# subdir under $LOG_DIR/$RUN_NAME.  If found, sets RESUME_CKPT_OVERRIDE env
# var which the Python script reads (see CONFIG block: fallback if env var
# set).  Skip entirely by setting NO_AUTO_RESUME=1 before sbatch.
if [[ -z "${NO_AUTO_RESUME:-}" ]]; then
    RUN_NAME=$(grep -E "^RUN_NAME\s*=" "$SCRIPT" | head -n 1 | sed -E 's/.*"([^"]+)".*/\1/')
    LOG_DIR=$(grep -E "^LOG_DIR\s*=" "$SCRIPT" | head -n 1 | sed -E 's/.*"([^"]+)".*/\1/')
    LATEST=$(ls -1t "$LOG_DIR/$RUN_NAME"/*/checkpoints/last.pt 2>/dev/null | head -n 1)
    if [[ -n "$LATEST" ]]; then
        export RESUME_CKPT_OVERRIDE="$LATEST"
        echo "[auto-resume] resuming from $LATEST"
    else
        echo "[auto-resume] no prior last.pt found under $LOG_DIR/$RUN_NAME — starting fresh"
    fi
fi

echo "=== job $SLURM_JOB_ID  starting $(date) ==="
echo "  nodes        : $SLURM_JOB_NUM_NODES"
echo "  gpus per node: $SLURM_GPUS_PER_NODE"
echo "  total ranks  : $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "  master       : $MASTER_ADDR:$MASTER_PORT"
echo "  nodelist     : $SLURM_JOB_NODELIST"
echo

srun -l torchrun \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --nproc-per-node="$SLURM_GPUS_PER_NODE" \
     --rdzv-backend=c10d \
     --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     "$SCRIPT"

echo
echo "=== job $SLURM_JOB_ID  finished $(date) ==="
