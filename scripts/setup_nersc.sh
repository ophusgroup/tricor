#!/bin/bash
# Setup script for NERSC (Perlmutter).
# Run once to create the conda environment and install dependencies.
#
# Usage:
#   bash setup_nersc.sh

set -e

ENV_NAME="mlstructgen"

echo "=== Creating conda environment: ${ENV_NAME} ==="
module load conda
conda create -n ${ENV_NAME} python=3.11 -y
conda activate ${ENV_NAME}

echo "=== Installing PyTorch (CUDA) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing PyG, Lightning, and other dependencies ==="
pip install torch-geometric lightning pandas scikit-learn

echo "=== Cloning and installing graphite ==="
cd $SCRATCH
git clone https://github.com/LLNL/graphite.git
pip install -e graphite/

echo "=== Installing tricor ==="
# Assumes tricor is cloned to $SCRATCH/tricor
cd $SCRATCH
pip install -e tricor/

echo "=== Verifying installation ==="
python -c "
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import torch_geometric; print(f'PyG {torch_geometric.__version__}')
import lightning; print(f'Lightning {lightning.__version__}')
import graphite; print('graphite OK')
from tricor.glass import LitScoreNet, StructureDataModule
from tricor.glass.sampler import generate
from tricor.differentiable_pdf import DifferentiablePDFADF
print('All tricor.glass imports OK')
"

echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
