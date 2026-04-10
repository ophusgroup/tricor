"""Conditional flow matching for atomic structure generation.

Generates 3D atomic structures conditioned on target pair distribution
functions (g2) and angular distribution functions (ADF). Uses a
MeshGraphNets GNN with FiLM conditioning and ODE-based inference.

Compared to GLASS (score-based diffusion with inference-time guidance):
  - 20-100x faster inference (deterministic ODE, no gradient computation)
  - Single model supports multiple compositions
  - Requires paired (structure, g2, ADF) training data
"""

from .velocity_model import VelocityModel, LitFlowMatch
from .data import FlowMatchDataset, FlowMatchDataModule
from .sampler import generate, positions_to_atoms

__all__ = [
    "VelocityModel",
    "LitFlowMatch",
    "FlowMatchDataset",
    "FlowMatchDataModule",
    "generate",
    "positions_to_atoms",
]
