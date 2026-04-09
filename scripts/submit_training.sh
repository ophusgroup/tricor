#!/bin/bash
#SBATCH --job-name=glass-train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=glass-train-%j.out

# ──────────────────────────────────────────────────────
# GLASS score model training on Perlmutter
#
# Edit the variables below, then submit:
#   sbatch submit_training.sh
# ──────────────────────────────────────────────────────

# === Configuration (EDIT THESE) ===
DATA_DIR="$SCRATCH/structures/si3n4_25A/"   # directory of xyz files
SPECIES="7 14"                               # atomic numbers (N Si)
RUN_NAME="glass-si3n4"

# === Optional overrides ===
DIM=200
NUM_CONVS=5
BATCH_SIZE=32
MAX_EPOCHS=12000
LR=1e-3
DUP=128
CUTOFF=5.0
K=0.8

# === Environment ===
module load conda
conda activate mlstructgen

# === Run ===
srun python $SCRATCH/tricor/scripts/train_score_model.py \
    --data_dir "$DATA_DIR" \
    --species $SPECIES \
    --dim $DIM \
    --num_convs $NUM_CONVS \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --dup $DUP \
    --cutoff $CUTOFF \
    --k $K \
    --num_workers 8 \
    --gpus 1 \
    --log_dir "$SCRATCH/glass_logs" \
    --run_name "$RUN_NAME"
