"""Guided ODE sampler for unconditional flow matching.

Combines the unconditional velocity model with inference-time gradient
guidance from differentiable PDF/ADF, following GLASS's approach but
with flow matching instead of diffusion.

At each ODE step:
  1. Predict velocity: v = v_theta(x_t, t)
  2. Estimate clean structure: x_hat_1 = x_t + (1 - t) * v
  3. Compute spectral loss: L = ||g2(x_hat_1) - g2_target||^2
  4. Guidance gradient: s_guide = -grad_x L
  5. Combined velocity: v_total = v + w * s_guide
  6. ODE step: x_{t+dt} = x_t + dt * v_total
"""

import torch
import ase
from torch import Tensor
from typing import Optional, Callable

from graphite.nn import periodic_radius_graph

from .flow_utils import wrap_periodic, sample_uniform_in_cell


def _build_graph(pos: Tensor, cell: Tensor, cutoff: float):
    edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    return edge_index, edge_attr


@torch.no_grad()
def generate_unconditional(
    cell: Tensor,
    num_atoms: int,
    z: Tensor,
    velocity_model,
    *,
    cutoff: float = 5.0,
    num_steps: int = 30,
    method: str = "midpoint",
    return_trajectory: bool = False,
) -> Tensor:
    """Generate a structure without guidance (unconditional).

    Useful for checking that the prior model produces physically
    plausible structures before adding guidance.
    """
    device = cell.device
    pos = sample_uniform_in_cell(num_atoms, cell)
    trajectory = [pos.clone()] if return_trajectory else None

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_atoms, 1), i * dt, device=device)

        if method == "euler":
            edge_index, edge_attr = _build_graph(pos, cell, cutoff)
            v = velocity_model(z, edge_index, edge_attr, t)
            pos = wrap_periodic(pos + dt * v, cell)

        elif method == "midpoint":
            edge_index, edge_attr = _build_graph(pos, cell, cutoff)
            v1 = velocity_model(z, edge_index, edge_attr, t)
            pos_mid = wrap_periodic(pos + 0.5 * dt * v1, cell)

            t_mid = torch.full((num_atoms, 1), i * dt + 0.5 * dt, device=device)
            ei_mid, ea_mid = _build_graph(pos_mid, cell, cutoff)
            v2 = velocity_model(z, ei_mid, ea_mid, t_mid)
            pos = wrap_periodic(pos + dt * v2, cell)

        if return_trajectory:
            trajectory.append(pos.clone())

    if return_trajectory:
        return torch.stack(trajectory)
    return pos


@torch.no_grad()
def generate_guided(
    cell: Tensor,
    num_atoms: int,
    z: Tensor,
    velocity_model,
    spectral_loss_fn: Callable,
    target_g2: Tensor,
    target_adf: Tensor,
    species: Tensor,
    *,
    cutoff: float = 5.0,
    w: float = 3000.0,
    num_steps: int = 50,
    return_trajectory: bool = False,
    verbose: bool = False,
) -> Tensor:
    """Generate a structure with inference-time spectral guidance.

    Like GLASS's conditional denoising, but using flow matching ODE
    instead of reverse SDE. At each step, the unconditional velocity
    is combined with a gradient from the differentiable spectral loss.

    Args:
        cell: (3, 3) periodic cell matrix.
        num_atoms: Number of atoms.
        z: (num_atoms, num_species) one-hot species.
        velocity_model: Trained unconditional velocity model (EMA).
        spectral_loss_fn: DifferentiableSpectralLoss from tricor.
        target_g2: Target PDF tensor.
        target_adf: Target ADF tensor.
        species: (num_atoms,) integer atomic numbers.
        cutoff: Graph cutoff (default 5.0 A).
        w: Guidance weight (default 3000).
        num_steps: ODE integration steps (default 50).
        return_trajectory: If True, return all intermediate positions.
        verbose: Print progress.

    Returns:
        Final positions (num_atoms, 3), or trajectory.
    """
    device = cell.device
    pos = sample_uniform_in_cell(num_atoms, cell)
    trajectory = [pos.clone()] if return_trajectory else None

    cell_spec = cell.double()
    target_g2 = target_g2.to(device)
    target_adf = target_adf.to(device)
    species_dev = species.to(device)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((num_atoms, 1), t_val, device=device)

        # 1. Unconditional velocity
        edge_index, edge_attr = _build_graph(pos, cell, cutoff)
        v_prior = velocity_model(z, edge_index, edge_attr, t)

        # 2. Estimate clean structure (flow matching Tweedie analogue)
        # x_hat_1 = x_t + (1 - t) * v_theta
        remaining = max(1.0 - t_val, 1e-4)
        x_hat_1 = pos + remaining * v_prior
        x_hat_1 = wrap_periodic(x_hat_1, cell)

        # 3-4. Spectral guidance gradient
        with torch.enable_grad():
            x_hat_grad = x_hat_1.detach().double().requires_grad_(True)
            loss, components = spectral_loss_fn(
                x_hat_grad, species_dev, cell_spec, target_g2, target_adf,
            )
            grad = torch.autograd.grad(loss, x_hat_grad)[0]
        s_guide = -grad.float()

        if verbose and i % max(1, num_steps // 10) == 0:
            pdf_l = components["pdf_loss"].item()
            adf_l = components["adf_loss"].item()
            print(
                f"  step {i:>3}/{num_steps} | t={t_val:.3f} | "
                f"pdf_loss={pdf_l:.4f} | adf_loss={adf_l:.4f} | "
                f"|v_prior|={v_prior.norm():.1f} | |s_guide|={s_guide.norm():.1f}"
            )

        # 5. Combined velocity
        v_total = v_prior + w * s_guide

        # 6. ODE step
        pos = wrap_periodic(pos + dt * v_total, cell)

        if return_trajectory:
            trajectory.append(pos.clone())

    if return_trajectory:
        return torch.stack(trajectory)
    return pos


def positions_to_atoms(pos: Tensor, cell: Tensor, species: Tensor) -> ase.Atoms:
    """Convert generated positions to an ASE Atoms object."""
    atoms = ase.Atoms(
        numbers=species.cpu().numpy(),
        positions=pos.detach().cpu().numpy(),
        cell=cell.cpu().numpy(),
        pbc=[True, True, True],
    )
    atoms.wrap()
    return atoms
