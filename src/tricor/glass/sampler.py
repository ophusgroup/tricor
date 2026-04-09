"""Denoising samplers for GLASS structure generation.

Implements the two-stage denoising procedure from Guo & Schwalbe-Koda
(arXiv:2603.23210):

  Stage 1 (unconditional): Reverse VE-SDE from t_max=1.0 to t_min=0.001
      with N=512 steps. Only the prior score drives the dynamics.

  Stage 2 (conditional): Reverse VE-SDE from t_max=0.6 to t_min=0.001
      with N=512 steps. At each step, a Tweedie clean-structure estimate
      is formed, the differentiable spectroscopic loss is evaluated, and
      the guidance gradient is combined with the prior score.

The reverse SDE update (GLASS Methods, p.11):

    x_{t-dt} = x_t + [f(t)*x_t - g(t)^2 * s_total] * dt + g(t)*sqrt(|dt|)*z

where:
    s_total = s_prior(x_t, t)                           [unconditional]
    s_total = s_prior(x_t, t) + w * s_guide(x_t, t)    [conditional]

and s_guide is derived from the Tweedie clean estimate:
    x_hat_0 = x_t + sigma(t)^2 * s_prior(x_t, t)
    s_guide = -grad_{x_t} L_spec(F(x_hat_0), y_target)
"""

import torch
import ase
from torch import Tensor
from typing import Optional, Callable

from graphite.nn import periodic_radius_graph
from graphite.diffusion import VarianceExplodingDiffuser


def _build_graph(pos: Tensor, cell: Tensor, cutoff: float):
    """Build periodic graph and return edge_index, edge_attr."""
    edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])
    return edge_index, edge_attr


def _prior_score_fn(
    pos: Tensor,
    cell: Tensor,
    t: Tensor,
    score_model,
    z: Tensor,
    cutoff: float,
    diffuser: VarianceExplodingDiffuser,
) -> Tensor:
    """Evaluate the prior score s_prior(x_t, t) / sigma, then return the
    un-normalized score s_prior(x_t, t) = model_output (already divided by sigma
    inside the model)."""
    edge_index, edge_attr = _build_graph(pos, cell, cutoff)
    sigma = diffuser.sigma(t)
    return score_model(z, edge_index, edge_attr, t, sigma)


def _sde_step(
    pos: Tensor,
    score: Tensor,
    t: Tensor,
    dt: Tensor,
    diffuser: VarianceExplodingDiffuser,
) -> Tensor:
    """Single reverse SDE step."""
    f_t = diffuser.f(t)
    g2_t = diffuser.g2(t)
    g_t = diffuser.g(t)
    eps = dt.abs().sqrt() * torch.randn_like(pos)
    disp = (f_t * pos - g2_t * score) * dt + g_t * eps
    return pos + disp


def _enforce_min_distance(
    pos: Tensor,
    pos_prev: Tensor,
    cell: Tensor,
    threshold: float,
) -> Tensor:
    """Minimum-distance veto: revert atoms that create short contacts.

    Following GLASS Sec. S2.7: after each SDE step, check for atom pairs
    below a threshold distance. Atoms involved in overlaps have their
    displacements suppressed (reverted to the previous position).
    """
    edge_index, edge_vec = periodic_radius_graph(pos, threshold, cell)
    if edge_index.size(1) == 0:
        return pos

    # Find atoms involved in short contacts
    overlapping = torch.zeros(pos.size(0), dtype=torch.bool, device=pos.device)
    overlapping[edge_index[0]] = True
    overlapping[edge_index[1]] = True

    # Revert overlapping atoms to previous positions
    pos = pos.clone()
    pos[overlapping] = pos_prev[overlapping]
    return pos


@torch.no_grad()
def denoise_unconditional(
    pos: Tensor,
    cell: Tensor,
    z: Tensor,
    score_model,
    *,
    diffuser: Optional[VarianceExplodingDiffuser] = None,
    cutoff: float = 5.0,
    t_max: float = 1.0,
    t_min: float = 0.001,
    num_steps: int = 512,
    min_dist_threshold: Optional[float] = None,
    return_trajectory: bool = False,
) -> Tensor:
    """Stage 1: Unconditional denoising with the prior score only.

    Args:
        pos: (N, 3) initial atom positions (e.g. random uniform in cell).
        cell: (3, 3) periodic cell matrix.
        z: (N, num_species) one-hot species encoding.
        score_model: Trained score network (EMA model). Should accept
            (z, edge_index, edge_attr, t, sigma) and return per-atom scores.
        diffuser: VE-SDE diffuser instance. If None, creates one with k=0.8.
        cutoff: Graph construction cutoff (default 5.0 A).
        t_max: Starting noise level (default 1.0).
        t_min: Final noise level (default 0.001).
        num_steps: Number of discretization steps (default 512).
        min_dist_threshold: If set, apply minimum-distance veto at each step.
        return_trajectory: If True, return all intermediate positions.

    Returns:
        Final denoised positions (N, 3), or trajectory (num_steps+1, N, 3).
    """
    if diffuser is None:
        diffuser = VarianceExplodingDiffuser(k=0.8)

    device = pos.device
    ts = torch.linspace(t_max, t_min, num_steps + 1, device=device).view(-1, 1)
    trajectory = [pos.clone()] if return_trajectory else None

    for i in range(num_steps):
        t = ts[i].expand(pos.size(0), 1)
        dt = ts[i + 1] - ts[i]

        score = _prior_score_fn(pos, cell, t, score_model, z, cutoff, diffuser)
        pos_prev = pos.clone()
        pos = _sde_step(pos, score, t, dt, diffuser)

        if min_dist_threshold is not None:
            pos = _enforce_min_distance(pos, pos_prev, cell, min_dist_threshold)

        if return_trajectory:
            trajectory.append(pos.clone())

    if return_trajectory:
        return torch.stack(trajectory)
    return pos


@torch.no_grad()
def denoise_conditional(
    pos: Tensor,
    cell: Tensor,
    z: Tensor,
    score_model,
    spectral_loss_fn: Callable,
    target_g2: Tensor,
    target_adf: Tensor,
    species: Tensor,
    *,
    diffuser: Optional[VarianceExplodingDiffuser] = None,
    cutoff: float = 5.0,
    w: float = 3000.0,
    t_max: float = 0.6,
    t_min: float = 0.001,
    num_steps: int = 512,
    min_dist_threshold: Optional[float] = None,
    return_trajectory: bool = False,
    verbose: bool = False,
) -> Tensor:
    """Stage 2: Conditional denoising with spectroscopic guidance.

    At each step:
      1. Compute prior score s_prior = score_model(x_t, t)
      2. Tweedie clean estimate: x_hat_0 = x_t + sigma(t)^2 * s_prior
      3. Compute spectroscopic loss: L = ||F(x_hat_0) - y_target||^2
      4. Guidance: s_guide = -grad_{x_t} L
      5. Combined score: s_total = s_prior + w * s_guide
      6. Reverse SDE step with s_total

    Args:
        pos: (N, 3) initial positions (typically output of Stage 1).
        cell: (3, 3) periodic cell matrix.
        z: (N, num_species) one-hot species encoding.
        score_model: Trained score network (EMA model).
        spectral_loss_fn: A DifferentiableSpectralLoss instance from tricor.
        target_g2: Target PDF tensor for guidance.
        target_adf: Target ADF tensor for guidance.
        species: (N,) integer atomic numbers for the spectral calculator.
        diffuser: VE-SDE diffuser. If None, creates with k=0.8.
        cutoff: Graph construction cutoff (default 5.0 A).
        w: Guidance weight (default 3000, GLASS Sec. S1.7).
        t_max: Starting noise level for conditional stage (default 0.6).
        t_min: Final noise level (default 0.001).
        num_steps: Discretization steps (default 512).
        min_dist_threshold: Minimum-distance veto threshold.
        return_trajectory: If True, return all intermediate positions.
        verbose: If True, print loss at each step.

    Returns:
        Final denoised positions (N, 3), or trajectory (num_steps+1, N, 3).
    """
    if diffuser is None:
        diffuser = VarianceExplodingDiffuser(k=0.8)

    device = pos.device
    ts = torch.linspace(t_max, t_min, num_steps + 1, device=device).view(-1, 1)
    trajectory = [pos.clone()] if return_trajectory else None

    cell_spec = cell.double()
    target_g2 = target_g2.to(device)
    target_adf = target_adf.to(device)
    species_dev = species.to(device)

    for i in range(num_steps):
        t = ts[i].expand(pos.size(0), 1)
        dt = ts[i + 1] - ts[i]
        sigma = diffuser.sigma(t)

        # 1. Prior score
        score_prior = _prior_score_fn(
            pos, cell, t, score_model, z, cutoff, diffuser
        )

        # 2. Tweedie clean estimate
        x_hat_0 = pos + sigma.pow(2) * score_prior

        # 3-4. Spectral guidance gradient
        # Must enable grad inside the no_grad context for autograd
        with torch.enable_grad():
            x_hat_0_grad = x_hat_0.detach().double().requires_grad_(True)
            loss, components = spectral_loss_fn(
                x_hat_0_grad, species_dev, cell_spec, target_g2, target_adf
            )
            grad = torch.autograd.grad(loss, x_hat_0_grad)[0]
        s_guide = -grad.float()

        if verbose and i % max(1, num_steps // 20) == 0:
            pdf_l = components["pdf_loss"].item()
            adf_l = components["adf_loss"].item()
            print(
                f"  step {i:>4}/{num_steps} | t={ts[i].item():.3f} | "
                f"pdf_loss={pdf_l:.4f} | adf_loss={adf_l:.4f} | "
                f"|s_prior|={score_prior.norm():.1f} | |s_guide|={s_guide.norm():.1f}"
            )

        # 5. Combined score
        score_total = score_prior + w * s_guide

        # 6. Reverse SDE step
        pos_prev = pos.clone()
        pos = _sde_step(pos, score_total, t, dt, diffuser)

        if min_dist_threshold is not None:
            pos = _enforce_min_distance(pos, pos_prev, cell, min_dist_threshold)

        if return_trajectory:
            trajectory.append(pos.clone())

    if return_trajectory:
        return torch.stack(trajectory)
    return pos


def generate(
    cell: Tensor,
    num_atoms: int,
    z: Tensor,
    score_model,
    spectral_loss_fn,
    target_g2: Tensor,
    target_adf: Tensor,
    species: Tensor,
    *,
    diffuser: Optional[VarianceExplodingDiffuser] = None,
    cutoff: float = 5.0,
    w: float = 3000.0,
    uncond_steps: int = 512,
    cond_steps: int = 512,
    uncond_t_max: float = 1.0,
    cond_t_max: float = 0.6,
    t_min: float = 0.001,
    min_dist_threshold: Optional[float] = None,
    verbose: bool = False,
) -> Tensor:
    """Full two-stage GLASS generation pipeline.

    Initializes from random positions, runs unconditional denoising (Stage 1),
    then conditional denoising with spectroscopic guidance (Stage 2).

    Args:
        cell: (3, 3) periodic cell matrix.
        num_atoms: Number of atoms to generate.
        z: (num_atoms, num_species) one-hot species encoding.
        score_model: Trained EMA score network.
        spectral_loss_fn: DifferentiableSpectralLoss from tricor.
        target_g2: Target PDF for guidance.
        target_adf: Target ADF for guidance.
        species: (num_atoms,) integer atomic numbers.
        Other args: see denoise_unconditional and denoise_conditional.

    Returns:
        Final denoised positions (num_atoms, 3).
    """
    if diffuser is None:
        diffuser = VarianceExplodingDiffuser(k=0.8)

    device = cell.device

    # Initialize from random uniform positions in the cell
    cell_diag = cell.diag()
    pos = torch.empty(num_atoms, 3, device=device).uniform_(0, 1) * cell_diag

    if verbose:
        print("Stage 1: Unconditional denoising")

    # Stage 1: Unconditional
    pos = denoise_unconditional(
        pos, cell, z, score_model,
        diffuser=diffuser,
        cutoff=cutoff,
        t_max=uncond_t_max,
        t_min=t_min,
        num_steps=uncond_steps,
        min_dist_threshold=min_dist_threshold,
    )

    if verbose:
        print("Stage 2: Conditional denoising with spectroscopic guidance")

    # Re-noise to cond_t_max for Stage 2
    t_restart = torch.full((pos.size(0), 1), cond_t_max, device=device)
    pos, _ = diffuser.forward_noise(pos, t_restart)

    # Stage 2: Conditional
    pos = denoise_conditional(
        pos, cell, z, score_model,
        spectral_loss_fn, target_g2, target_adf, species,
        diffuser=diffuser,
        cutoff=cutoff,
        w=w,
        t_max=cond_t_max,
        t_min=t_min,
        num_steps=cond_steps,
        min_dist_threshold=min_dist_threshold,
        verbose=verbose,
    )

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
