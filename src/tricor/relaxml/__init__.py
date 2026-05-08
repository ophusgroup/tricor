"""Step-by-step surrogate for tricor's Supercell.generate relaxation.

Trains a GNN to predict *k-step* atomic displacements during
``shell_relax``.  At inference, the model is applied iteratively to its
own output until the structure stops moving, mimicking tricor's
iterative relaxation but with far larger strides per application.

Modules
-------
data  -- ``RelaxMLDataset`` + ``RelaxMLDataModule`` reading the .npz
         trajectories produced by
         ``scripts/flowmatch_guided/generate_surrogate_trajectories.py``.
model -- ``RelaxMLModel`` (MGN backbone + weight-vector conditioning) and
         ``LitRelaxML`` PyTorch Lightning training wrapper.
"""

from .data import (
    RelaxMLDataset,
    RelaxMLDataModule,
    WEIGHT_FEATURE_KEYS,
    WEIGHT_FEATURE_SCALES,
    NUM_WEIGHT_FEATURES,
)
from .model import RelaxMLModel, LitRelaxML

__all__ = [
    "RelaxMLDataset",
    "RelaxMLDataModule",
    "RelaxMLModel",
    "LitRelaxML",
    "WEIGHT_FEATURE_KEYS",
    "WEIGHT_FEATURE_SCALES",
    "NUM_WEIGHT_FEATURES",
]
