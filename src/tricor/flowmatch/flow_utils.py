"""Flow matching utilities for periodic atomic structures.

Provides:
  - Optimal transport conditional flow matching (OT-CFM) interpolation
    with periodic boundary conditions
  - Per-species Hungarian assignment for straightening flow paths
  - Periodic wrapping helpers
"""

import torch
import numpy as np
from torch import Tensor
from typing import Optional

from scipy.optimize import linear_sum_assignment


def wrap_periodic(pos: Tensor, cell: Tensor) -> Tensor:
    """Wrap positions into the periodic cell [0, cell)."""
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T
    frac = frac - torch.floor(frac)
    return frac @ cell


def minimum_image_displacement(pos_i: Tensor, pos_j: Tensor, cell: Tensor) -> Tensor:
    """Minimum-image displacement vectors pos_j - pos_i."""
    inv_cell = torch.linalg.inv(cell)
    delta = pos_j - pos_i
    delta_frac = delta @ inv_cell.T
    delta_frac = delta_frac - torch.round(delta_frac)
    return delta_frac @ cell


def periodic_interpolation(
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    cell: Tensor,
) -> tuple[Tensor, Tensor]:
    """OT-CFM interpolation with periodic boundary conditions.

    Computes:
        x_t = x0 + t * delta     (where delta is the minimum-image displacement)
        target_velocity = delta   (the constant velocity along the straight path)

    Then wraps x_t back into the periodic cell.

    Args:
        x0: (N, 3) source positions (noise).
        x1: (N, 3) target positions (data).
        t: (N, 1) interpolation time in [0, 1].
        cell: (3, 3) periodic cell matrix.

    Returns:
        x_t: (N, 3) interpolated positions, wrapped into cell.
        target_velocity: (N, 3) target velocity field = x1 - x0 (minimum image).
    """
    delta = minimum_image_displacement(x0, x1, cell)
    x_t = x0 + t * delta
    x_t = wrap_periodic(x_t, cell)
    return x_t, delta


def per_species_ot_assignment(
    x0: Tensor,
    x1: Tensor,
    species: Tensor,
    cell: Tensor,
) -> Tensor:
    """Compute optimal transport assignment per species using Hungarian algorithm.

    For each species, solves a linear assignment problem to pair noise atoms
    with data atoms of the same species, minimizing total minimum-image
    displacement. This straightens flow paths and improves training.

    Args:
        x0: (N, 3) noise positions.
        x1: (N, 3) data positions.
        species: (N,) integer species labels (local indices, not atomic numbers).
        cell: (3, 3) periodic cell matrix.

    Returns:
        perm: (N,) permutation of x0 indices such that x0[perm] is optimally
              paired with x1.
    """
    N = x0.shape[0]
    perm = torch.arange(N, device=x0.device)

    unique_species = species.unique()
    for sp in unique_species:
        mask = species == sp
        idx = torch.where(mask)[0]

        x0_sp = x0[idx]  # (n_sp, 3)
        x1_sp = x1[idx]  # (n_sp, 3)
        n_sp = x0_sp.shape[0]

        if n_sp <= 1:
            continue

        # Pairwise minimum-image distances
        # (n_sp, 1, 3) - (1, n_sp, 3) -> (n_sp, n_sp, 3)
        delta = minimum_image_displacement(
            x0_sp.unsqueeze(1), x1_sp.unsqueeze(0), cell
        )
        cost = delta.pow(2).sum(dim=-1).cpu().numpy()  # (n_sp, n_sp)

        # Hungarian algorithm
        # row_ind[i], col_ind[i] means: noise atom row_ind[i] pairs with data atom col_ind[i]
        row_ind, col_ind = linear_sum_assignment(cost)

        # We want perm such that x0[perm][i] is close to x1[i].
        # row_ind[i] is the noise atom, col_ind[i] is the data atom it should pair with.
        # So noise atom row_ind[i] should end up at position col_ind[i].
        # Equivalently: at position col_ind[i], we want noise atom row_ind[i].
        # perm[col_ind[i]] = row_ind[i] → x0[perm][col_ind[i]] = x0[row_ind[i]]
        inv_col = torch.empty_like(idx)
        inv_col[torch.tensor(col_ind, device=idx.device)] = torch.tensor(row_ind, device=idx.device)
        perm[idx] = idx[inv_col]

    return perm


def sample_uniform_in_cell(num_atoms: int, cell: Tensor) -> Tensor:
    """Sample positions uniformly in the periodic cell."""
    frac = torch.rand(num_atoms, 3, device=cell.device, dtype=cell.dtype)
    return frac @ cell
