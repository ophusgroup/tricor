"""ODE-based inference for conditional flow matching.

Integrates the learned velocity field from t=0 (noise) to t=1 (data)
using Euler or midpoint methods. No gradient computation needed —
all conditioning is baked into the model at training time.

Typical usage:
    pos = generate(cell, num_atoms, z, velocity_model,
                   g2_target, adf_target, comp_frac,
                   num_steps=30)
"""

import torch
import ase
from torch import Tensor
from typing import Optional

from .flow_utils import wrap_periodic, sample_uniform_in_cell, periodic_radius_graph_chunked

_GRAPH_CHUNK = 1024


def _build_graph(pos: Tensor, cell: Tensor, cutoff: float):
    edge_index, edge_vec = periodic_radius_graph_chunked(
        pos, cutoff, cell, chunk=_GRAPH_CHUNK,
    )
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    return edge_index, edge_attr


@torch.no_grad()
def generate(
    cell: Tensor,
    num_atoms: int,
    z: Tensor,
    velocity_model,
    g2_target: Tensor,
    adf_target: Tensor,
    comp_frac: Tensor,
    *,
    cutoff: float = 5.0,
    num_steps: int = 30,
    method: str = "midpoint",
    return_trajectory: bool = False,
) -> Tensor:
    """Generate a structure by integrating the conditional velocity field.

    Args:
        cell: (3, 3) periodic cell matrix.
        num_atoms: Number of atoms to generate.
        z: (num_atoms, num_species) one-hot species encoding.
        velocity_model: Trained velocity network (EMA model).
        g2_target: (1, ns, ns, nr) target PDF (unsqueezed batch dim).
        adf_target: (1, nt, np) target ADF (unsqueezed batch dim).
        comp_frac: (1, ns) composition fractions (unsqueezed batch dim).
        cutoff: Graph construction cutoff (default 5.0 A).
        num_steps: Number of ODE integration steps (default 30).
        method: "euler" or "midpoint" (default "midpoint").
        return_trajectory: If True, return all intermediate positions.

    Returns:
        Final positions (num_atoms, 3), or trajectory (num_steps+1, num_atoms, 3).
    """
    device = cell.device
    batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)

    # Initialize from uniform noise in the cell
    pos = sample_uniform_in_cell(num_atoms, cell)
    trajectory = [pos.clone()] if return_trajectory else None

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((num_atoms, 1), t_val, device=device)

        if method == "euler":
            edge_index, edge_attr = _build_graph(pos, cell, cutoff)
            v = velocity_model(
                z, edge_index, edge_attr, t,
                g2_target, adf_target, comp_frac, batch_idx,
            )
            pos = pos + dt * v
            pos = wrap_periodic(pos, cell)

        elif method == "midpoint":
            # Half step
            edge_index, edge_attr = _build_graph(pos, cell, cutoff)
            v1 = velocity_model(
                z, edge_index, edge_attr, t,
                g2_target, adf_target, comp_frac, batch_idx,
            )
            pos_mid = wrap_periodic(pos + 0.5 * dt * v1, cell)

            # Full step from midpoint
            t_mid = torch.full((num_atoms, 1), t_val + 0.5 * dt, device=device)
            edge_index_mid, edge_attr_mid = _build_graph(pos_mid, cell, cutoff)
            v2 = velocity_model(
                z, edge_index_mid, edge_attr_mid, t_mid,
                g2_target, adf_target, comp_frac, batch_idx,
            )
            pos = pos + dt * v2
            pos = wrap_periodic(pos, cell)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'euler' or 'midpoint'.")

        if return_trajectory:
            trajectory.append(pos.clone())

    if return_trajectory:
        return torch.stack(trajectory)
    return pos


def positions_to_atoms(
    pos: Tensor,
    cell: Tensor,
    species: Tensor,
) -> ase.Atoms:
    """Convert generated positions to an ASE Atoms object."""
    atoms = ase.Atoms(
        numbers=species.cpu().numpy(),
        positions=pos.detach().cpu().numpy(),
        cell=cell.cpu().numpy(),
        pbc=[True, True, True],
    )
    atoms.wrap()
    return atoms
