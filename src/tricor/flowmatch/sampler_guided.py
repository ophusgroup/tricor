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

from .flow_utils import (
    wrap_periodic,
    sample_uniform_in_cell,
    periodic_radius_graph_chunked,
    periodic_radius_graph_cell_list,
)

# Chunk size for the O(N^2) chunked fallback (triclinic or tiny cells).
_GRAPH_CHUNK = 1024


def _build_graph(pos: Tensor, cell: Tensor, cutoff: float):
    """Build a periodic radius graph for the flow-matching velocity model.

    Uses the O(N) cell list when the cell is orthogonal and each axis is
    ≥ 3*cutoff; falls back to the O(N^2) chunked neighbor search otherwise
    (triclinic cells or boxes too small for a 27-cell neighborhood).
    """
    try:
        edge_index, edge_vec = periodic_radius_graph_cell_list(pos, cutoff, cell)
    except ValueError:
        edge_index, edge_vec = periodic_radius_graph_chunked(
            pos, cutoff, cell, chunk=_GRAPH_CHUNK,
        )
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
    guide_norm_mode: str = "absolute",   # "absolute" | "relative"
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

        if guide_norm_mode == "relative":
            prior_norm = v_prior.norm()
            guide_norm = s_guide.norm().clamp_min(1e-12)
            s_guide = s_guide * (prior_norm / guide_norm)
        elif guide_norm_mode != "absolute":
            raise ValueError(f"Unknown guide_norm_mode: {guide_norm_mode}")

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


def _enforce_min_distance(
    pos: Tensor,
    pos_prev: Tensor,
    cell: Tensor,
    threshold: float,
) -> Tensor:
    """Revert atoms involved in sub-threshold pairs (GLASS Sec. S2.7)."""
    edge_index, _ = periodic_radius_graph_chunked(
        pos, threshold, cell, chunk=_GRAPH_CHUNK,
    )
    if edge_index.size(1) == 0:
        return pos
    overlap = torch.zeros(pos.size(0), dtype=torch.bool, device=pos.device)
    overlap[edge_index[0]] = True
    overlap[edge_index[1]] = True
    pos = pos.clone()
    pos[overlap] = pos_prev[overlap]
    return pos


@torch.no_grad()
def generate_glass_style(
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
    w: float = 1.0,
    uncond_steps: int = 300,
    cond_steps: int = 300,
    t_switch: float = 0.4,
    min_dist_threshold: Optional[float] = None,
    guide_norm_mode: str = "relative",
    noise_sigma: float = 0.1,            # 0 = pure ODE, >0 = SDE per-step noise
    noise_schedule: str = "linear",      # "linear" | "bridge" | "const" | "none"
    surrogate=None,                      # LitSurrogate; if provided, replaces
                                         # autograd through spectral_loss_fn
                                         # for the per-step guidance gradient.
    skip_branch_a: bool = False,         # production mode: skip the uncond
                                         # comparison branch entirely; returns
                                         # (None, pos_b). ~30% wall-time saving.
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """Two-stage FM sampler matching GLASS's generation procedure.

    Single shared trajectory from fresh noise up to ``t_switch``, then
    branches into two continuations:

      * Branch A (no guidance): continues uncond to t=1 -> ``pos_uncond``.
      * Branch B (guided):      continues with guidance to t=1 -> ``pos_final``.

    Both outputs come from the same noise seed and the same phase-1
    trajectory, so differences are attributable purely to the guidance
    applied over ``t in [t_switch, 1]``.

    ``t_switch=0.4`` is the FM analog of GLASS's ``cond_t_max=0.6``
    (denoising level at which guidance turns on).

    Returns
    -------
    (pos_uncond, pos_final)
        Fully-denoised positions from the uncond branch and the guided
        branch respectively.
    """
    device = cell.device
    pos = sample_uniform_in_cell(num_atoms, cell)

    cell_spec = cell.double()
    target_g2 = target_g2.to(device)
    target_adf = target_adf.to(device)
    species_dev = species.to(device)

    def _noise_scale(t_val: float) -> float:
        if noise_sigma <= 0.0 or noise_schedule == "none":
            return 0.0
        if noise_schedule == "const":
            return float(noise_sigma)
        if noise_schedule == "linear":
            return float(noise_sigma) * max(0.0, 1.0 - t_val)
        if noise_schedule == "bridge":
            return float(noise_sigma) * max(0.0, t_val * (1.0 - t_val)) ** 0.5
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}")

    # ── Phase 1 (shared): unconditional, t in [0, t_switch] ───────────
    if verbose:
        print(f"Phase 1 (shared uncond): t in [0, {t_switch}], {uncond_steps} steps")
    dt1 = t_switch / uncond_steps
    sqrt_dt1 = dt1 ** 0.5
    for i in range(uncond_steps):
        t_val = i * dt1
        t = torch.full((num_atoms, 1), t_val, device=device)
        edge_index, edge_attr = _build_graph(pos, cell, cutoff)
        v = velocity_model(z, edge_index, edge_attr, t)
        pos_prev = pos.clone()
        sigma = _noise_scale(t_val)
        eps = torch.randn_like(pos) if sigma > 0.0 else 0.0
        pos = wrap_periodic(pos + dt1 * v + sigma * sqrt_dt1 * eps, cell)
        if min_dist_threshold is not None:
            pos = _enforce_min_distance(pos, pos_prev, cell, min_dist_threshold)

    pos_branch_start = pos.clone()
    dt2 = (1.0 - t_switch) / cond_steps
    sqrt_dt2 = dt2 ** 0.5

    # ── Branch A: continue uncond t in [t_switch, 1] ──────────────────
    if skip_branch_a:
        if verbose:
            print("Branch A skipped (skip_branch_a=True)")
        pos_a = None
    else:
        if verbose:
            print(f"Branch A (uncond continuation): t in [{t_switch}, 1], {cond_steps} steps")
        pos_a = pos_branch_start.clone()
        for i in range(cond_steps):
            t_val = t_switch + i * dt2
            t = torch.full((num_atoms, 1), t_val, device=device)
            edge_index, edge_attr = _build_graph(pos_a, cell, cutoff)
            v = velocity_model(z, edge_index, edge_attr, t)
            pos_prev = pos_a.clone()
            sigma = _noise_scale(t_val)
            eps = torch.randn_like(pos_a) if sigma > 0.0 else 0.0
            pos_a = wrap_periodic(pos_a + dt2 * v + sigma * sqrt_dt2 * eps, cell)
            if min_dist_threshold is not None:
                pos_a = _enforce_min_distance(pos_a, pos_prev, cell, min_dist_threshold)

    # ── Branch B: guided, t in [t_switch, 1] ──────────────────────────
    if verbose:
        print(f"Branch B (guided, w={w}): t in [{t_switch}, 1], {cond_steps} steps")
    pos_b = pos_branch_start.clone()
    for i in range(cond_steps):
        t_val = t_switch + i * dt2
        t = torch.full((num_atoms, 1), t_val, device=device)

        edge_index, edge_attr = _build_graph(pos_b, cell, cutoff)
        v_prior = velocity_model(z, edge_index, edge_attr, t)

        remaining = max(1.0 - t_val, 1e-4)
        x_hat_1 = wrap_periodic(pos_b + remaining * v_prior, cell)

        if surrogate is None:
            # Exact gradient via autograd through differentiable PDF/ADF.
            with torch.enable_grad():
                x_hat_grad = x_hat_1.detach().double().requires_grad_(True)
                loss, components = spectral_loss_fn(
                    x_hat_grad, species_dev, cell_spec, target_g2, target_adf,
                )
                grad = torch.autograd.grad(loss, x_hat_grad)[0]
            s_guide = -grad.float()
        else:
            # Surrogate forward pass on x_hat_1's graph.
            ei_h, ea_h = _build_graph(x_hat_1, cell, cutoff)
            batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)
            grad_pred = surrogate.predict_gradient(
                z, ei_h, ea_h, t,
                target_g2.unsqueeze(0).float(),
                target_adf.unsqueeze(0).float(),
                batch_idx,
            )
            s_guide = -grad_pred.float()
            # Diagnostics: still compute spectral loss components for logging.
            if verbose:
                with torch.no_grad():
                    _, components = spectral_loss_fn(
                        x_hat_1.double(), species_dev, cell_spec,
                        target_g2, target_adf,
                    )

        if guide_norm_mode == "relative":
            prior_norm = v_prior.norm()
            guide_norm = s_guide.norm().clamp_min(1e-12)
            s_guide = s_guide * (prior_norm / guide_norm)
        elif guide_norm_mode != "absolute":
            raise ValueError(f"Unknown guide_norm_mode: {guide_norm_mode}")

        if verbose and i % max(1, cond_steps // 10) == 0:
            print(
                f"  cond step {i:>3}/{cond_steps} | t={t_val:.3f} | "
                f"pdf_loss={components['pdf_loss'].item():.4f} | "
                f"adf_loss={components['adf_loss'].item():.4f} | "
                f"|v_prior|={v_prior.norm():.1f} | |s_guide|={s_guide.norm():.1f}"
            )

        v_total = v_prior + w * s_guide
        pos_prev = pos_b.clone()
        sigma = _noise_scale(t_val)
        eps = torch.randn_like(pos_b) if sigma > 0.0 else 0.0
        pos_b = wrap_periodic(pos_b + dt2 * v_total + sigma * sqrt_dt2 * eps, cell)
        if min_dist_threshold is not None:
            pos_b = _enforce_min_distance(pos_b, pos_prev, cell, min_dist_threshold)

    return pos_a, pos_b


@torch.no_grad()
def generate_hybrid(
    cell: Tensor,
    num_atoms: int,
    z: Tensor,
    cond_velocity_model,
    uncond_velocity_model,
    spectral_loss_fn: Callable,
    target_g2: Tensor,
    species: Tensor,
    *,
    g2_target_cond: Optional[Tensor] = None,
    adf_target_cond: Optional[Tensor] = None,
    comp_frac: Optional[Tensor] = None,
    cutoff: float = 5.0,
    cond_steps: int = 30,
    cond_method: str = "midpoint",
    refine_steps: int = 10,
    refine_t_start: float = 0.1,
    w: float = 3000.0,
    g2_only: bool = True,
    verbose: bool = False,
) -> Tensor:
    """Hybrid generation: conditional flow matching → guided refinement.

    Stage 1: Run the conditional flow matching model (fast, no gradients)
        to produce an approximate structure matching the target.
    Stage 2: Re-noise slightly, then run a few guided ODE steps using
        the unconditional model + g2 gradient guidance to refine toward
        the exact target.

    Args:
        cell: (3, 3) periodic cell matrix.
        num_atoms: Number of atoms.
        z: (num_atoms, num_species) one-hot species.
        cond_velocity_model: Trained conditional velocity model (EMA).
        uncond_velocity_model: Trained unconditional velocity model (EMA).
        spectral_loss_fn: Loss function wrapping DifferentiablePDFADF_Fast.
            If g2_only=True, must have a calculator with compute_g2_only().
        target_g2: Target g2 tensor for guidance (raw, not normalized).
        species: (num_atoms,) integer atomic numbers.
        g2_target_cond: Target g2 for the conditional model (normalized
            by atom count, matching training convention). If None, uses
            target_g2 / num_atoms.
        adf_target_cond: Target ADF for conditional model. If None, uses
            zeros (ignored if the conditional model was trained with ADF).
        comp_frac: (1, num_species) composition fractions for conditional
            model. Required.
        cutoff: Graph cutoff (default 5.0 A).
        cond_steps: ODE steps for conditional stage (default 30).
        cond_method: "euler" or "midpoint" for conditional stage.
        refine_steps: Guided refinement steps (default 10).
        refine_t_start: Flow time to re-noise to before refinement.
            Smaller = less perturbation = fewer corrections needed.
            Default 0.1.
        w: Guidance weight (default 3000).
        g2_only: If True, compute only g2 during guidance (much faster).
        verbose: Print progress.

    Returns:
        Final refined positions (num_atoms, 3).
    """
    device = cell.device
    batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)

    # ── Stage 1: Conditional flow matching (fast) ─────────────────────

    if verbose:
        print("Stage 1: Conditional flow matching")

    pos = sample_uniform_in_cell(num_atoms, cell)
    dt = 1.0 / cond_steps

    for i in range(cond_steps):
        t = torch.full((num_atoms, 1), i * dt, device=device)

        if cond_method == "euler":
            edge_index, edge_attr = _build_graph(pos, cell, cutoff)
            v = cond_velocity_model(
                z, edge_index, edge_attr, t,
                g2_target_cond, adf_target_cond, comp_frac, batch_idx,
            )
            pos = wrap_periodic(pos + dt * v, cell)

        elif cond_method == "midpoint":
            edge_index, edge_attr = _build_graph(pos, cell, cutoff)
            v1 = cond_velocity_model(
                z, edge_index, edge_attr, t,
                g2_target_cond, adf_target_cond, comp_frac, batch_idx,
            )
            pos_mid = wrap_periodic(pos + 0.5 * dt * v1, cell)

            t_mid = torch.full((num_atoms, 1), i * dt + 0.5 * dt, device=device)
            ei_mid, ea_mid = _build_graph(pos_mid, cell, cutoff)
            v2 = cond_velocity_model(
                z, ei_mid, ea_mid, t_mid,
                g2_target_cond, adf_target_cond, comp_frac, batch_idx,
            )
            pos = wrap_periodic(pos + dt * v2, cell)

    if verbose:
        print(f"  Conditional stage complete ({cond_steps} steps)")

    if refine_steps == 0:
        return pos

    # ── Stage 2: Guided refinement ────────────────────────────────────

    if verbose:
        print(f"Stage 2: Guided refinement ({refine_steps} steps, w={w})")

    # Re-noise slightly: push back to t = refine_t_start
    noise = torch.randn_like(pos) * refine_t_start * 0.8  # scale noise
    pos = wrap_periodic(pos + noise, cell)

    cell_spec = cell.double()
    target_g2_dev = target_g2.to(device)
    # Create a dummy ADF target for the loss function
    target_adf_dev = torch.zeros(
        spectral_loss_fn.calc.num_triplets, spectral_loss_fn.calc.phi_num_bins,
        device=device, dtype=torch.float64,
    )
    species_dev = species.to(device)

    dt_refine = refine_t_start / refine_steps

    for i in range(refine_steps):
        t_val = refine_t_start - i * dt_refine
        t = torch.full((num_atoms, 1), t_val, device=device)

        # Unconditional velocity
        edge_index, edge_attr = _build_graph(pos, cell, cutoff)
        v_prior = uncond_velocity_model(z, edge_index, edge_attr, t)

        # Clean estimate
        remaining = max(1.0 - t_val, 1e-4)
        x_hat_1 = wrap_periodic(pos + remaining * v_prior, cell)

        # g2 guidance gradient
        with torch.enable_grad():
            x_hat_grad = x_hat_1.detach().double().requires_grad_(True)
            if g2_only and hasattr(spectral_loss_fn.calc, 'compute_g2_only'):
                g2_pred, _ = spectral_loss_fn.calc.compute_g2_only(
                    x_hat_grad, species_dev, cell_spec,
                )
                pdf_loss = torch.mean((g2_pred - target_g2_dev) ** 2)
                grad = torch.autograd.grad(pdf_loss, x_hat_grad)[0]
                adf_loss_val = 0.0
            else:
                loss, components = spectral_loss_fn(
                    x_hat_grad, species_dev, cell_spec, target_g2_dev, target_adf_dev,
                )
                grad = torch.autograd.grad(loss, x_hat_grad)[0]
                pdf_loss = components["pdf_loss"]
                adf_loss_val = components["adf_loss"].item()
        s_guide = -grad.float()

        if verbose:
            pdf_l = pdf_loss.item() if isinstance(pdf_loss, torch.Tensor) else pdf_loss
            print(
                f"  refine {i+1:>2}/{refine_steps} | t={t_val:.3f} | "
                f"pdf_loss={pdf_l:.6f} | "
                f"|v_prior|={v_prior.norm():.1f} | |s_guide|={s_guide.norm():.1f}"
            )

        v_total = v_prior + w * s_guide
        pos = wrap_periodic(pos + dt_refine * v_total, cell)

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
