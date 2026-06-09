#!/bin/bash
# Multi-node MACE+wall trajectory generation on Perlmutter.
#
# One srun task per node; each task runs
# generate_mace_trajectories_perlmutter.py which:
#   (1) reads SLURM_NODEID + SLURM_JOB_NUM_NODES to take a round-robin
#       slice of the CIF list,
#   (2) spawns 4 GPU workers via _spawn_workers, each handling a further
#       round-robin slice across its share of CIFs.
#
# Resume on the next sbatch is automatic: per-CIF manifests + the shared
# calibrated_cells.csv mean re-submitting picks up where the previous job
# stopped at the 24-hour walltime cap.  Submit several in a row (chain via
# `sbatch --dependency=afterany:<jobid>`) for multi-day runs.
#
# Submit:
#   cd $HOME/macerelax            # or wherever the repo root is
#   sbatch scripts/macerelax/generation/perlmutter_generate.sbatch
#
# Edit before first submission:
#   - --account=<FILL_ME>         project allocation
#   - --nodes=N                   tune for queue + dataset size
#   - DATASET_ROOT inside the .py change to a $SCRATCH path
#   - CIF_DIR inside the .py change to your Perlmutter CIF library path

#SBATCH --job-name=macerelax_gen
#SBATCH --account=m5241
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Note on the constraint: `gpu&hbm80g` requests A100-80GB nodes
# specifically.  These have ~2× the memory of regular A100-40GB nodes,
# which is what lets CELL=50 fit for nearly every CIF in the library.
# Drop `&hbm80g` if your project is paying a premium for 80G and you'd
# rather accept some shrinks on 40G nodes (less queue contention).

set -euo pipefail

# ── Environment ─────────────────────────────────────────────────────────────
# Adjust the conda env name to whichever holds your mace + tricor + cueq
# install — typically the existing `tricor` env after the two extra pip
# installs (see PERLMUTTER_SETUP.md §1a), or a fresh `macerelax_gen` env
# (§1b).
module load conda
conda activate tricor          # <-- change if your env name differs

# Repo root (the directory containing scripts/macerelax/).  $SLURM_SUBMIT_DIR
# is where `sbatch` was invoked, which is the natural choice.
cd "$SLURM_SUBMIT_DIR"

# Belt-and-suspenders allocator config — also set inside the script for any
# bare-python launches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure the logs/ subdir exists for slurm-*.out files.
mkdir -p logs

echo "=== job $SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES  $(date) ==="
echo "  conda env: $CONDA_DEFAULT_ENV"
echo "  cwd:       $(pwd)"
echo "  host:      $(hostname)"
echo

# ── Launch ──────────────────────────────────────────────────────────────────
# One task per node; each task uses its 4 GPUs via the script's internal
# _spawn_workers.  --cpu-bind avoids srun stepping on the script's own
# thread caps.
#
# Absolute path to python sidesteps any PATH issues inside srun's per-task
# shells (sbatch contexts don't always inherit the activated conda env's
# PATH cleanly).  The conda env lives in NERSC's project-shared software
# area, not the user's home, so adjust this path if you've installed your
# env elsewhere.
PYTHON_BIN=/global/common/software/m5020/ehrdt/tricor/bin/python

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --cpu-bind=none \
    $PYTHON_BIN scripts/macerelax/generation/generate_mace_trajectories_perlmutter.py

echo
echo "=== job $SLURM_JOB_ID  finished $(date) ==="
