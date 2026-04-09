#!/bin/bash
#SBATCH --job-name=glass-gen
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=glass-gen-%j.out

# ──────────────────────────────────────────────────────
# GLASS conditional structure generation on Perlmutter
#
# Edit the variables below, then submit:
#   sbatch submit_generation.sh
# ──────────────────────────────────────────────────────

# === Configuration (EDIT THESE) ===
CHECKPOINT="$SCRATCH/glass_logs/glass-si3n4/version_0/checkpoints/last.ckpt"
TARGET_STRUCTURE="$SCRATCH/structures/reference_target.xyz"
SPECIES="7 14"
NUM_ATOMS=400
CELL_SIZE=25.0
ATOM_FRACTIONS="0.4286 0.5714"   # Si3N4: 3/7 N, 4/7 Si

# === Guidance parameters ===
W=3000
R_MAX=10.0
R_STEP=0.05
PDF_WEIGHT=1.0
ADF_WEIGHT=1.0

# === Sampling ===
NUM_RUNS=10
UNCOND_STEPS=512
COND_STEPS=512

# === Environment ===
module load conda
conda activate mlstructgen

# === Run ===
srun python $SCRATCH/tricor/scripts/generate_structures.py \
    --checkpoint "$CHECKPOINT" \
    --target_structure "$TARGET_STRUCTURE" \
    --species $SPECIES \
    --num_atoms $NUM_ATOMS \
    --cell_size $CELL_SIZE \
    --atom_fractions $ATOM_FRACTIONS \
    --w $W \
    --r_max $R_MAX \
    --r_step $R_STEP \
    --pdf_weight $PDF_WEIGHT \
    --adf_weight $ADF_WEIGHT \
    --uncond_steps $UNCOND_STEPS \
    --cond_steps $COND_STEPS \
    --num_runs $NUM_RUNS \
    --output_dir "$SCRATCH/generated/si3n4/" \
    --device cuda
