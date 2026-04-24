"""Flow matching for atomic structure generation.

Two approaches available:

1. **Conditional** (velocity_model.py, data.py, sampler.py):
   FiLM-conditioned model that takes target g2/ADF as input during training.
   Fast inference with no gradients, but limited to in-distribution targets.

2. **Unconditional + guidance** (velocity_model_uncond.py, data_uncond.py, sampler_guided.py):
   Unconditional velocity model with GLASS-style inference-time gradient guidance.
   Handles out-of-distribution targets, faster than diffusion-based GLASS.
"""

# Conditional
from .velocity_model import VelocityModel, LitFlowMatch
from .data import FlowMatchDataset, FlowMatchDataModule
from .sampler import generate, positions_to_atoms

# Unconditional + guidance
from .velocity_model_uncond import UncondVelocityModel, LitUncondFlowMatch
from .data_uncond import UncondFlowMatchDataset, UncondFlowMatchDataModule
from .sampler_guided import generate_unconditional, generate_guided, generate_hybrid

__all__ = [
    # Conditional
    "VelocityModel",
    "LitFlowMatch",
    "FlowMatchDataset",
    "FlowMatchDataModule",
    "generate",
    "positions_to_atoms",
    # Unconditional + guidance
    "UncondVelocityModel",
    "LitUncondFlowMatch",
    "UncondFlowMatchDataset",
    "UncondFlowMatchDataModule",
    "generate_unconditional",
    "generate_guided",
    "generate_hybrid",
]
