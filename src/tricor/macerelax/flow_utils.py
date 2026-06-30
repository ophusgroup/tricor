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


def periodic_radius_graph_chunked(
    x: Tensor,
    r: float,
    cell: Tensor,
    *,
    chunk: int = 1024,
    loop: bool = False,
) -> tuple[Tensor, Tensor]:
    """Drop-in chunked replacement for graphite.nn.periodic_radius_graph.

    Matches the v1 convention: edge_index[0]=i, edge_index[1]=j,
    edge_vec = min_image(x[j] - x[i]). Self-loops excluded unless
    loop=True. Works for any (triclinic) cell.

    Peak memory is O(chunk * N) instead of the brute-force O(N^2).
    At N=46k, chunk=1024 → ~540 MB peak vs ~25 GB for v1.
    """
    N = x.shape[0]
    inv_cell = torch.linalg.pinv(cell)
    r_sq = r * r

    i_parts, j_parts, v_parts = [], [], []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        B = end - start
        vec = x.unsqueeze(0) - x[start:end].unsqueeze(1)            # (B, N, 3)
        vec = vec - torch.round(vec @ inv_cell) @ cell
        dist_sq = (vec * vec).sum(dim=-1)                           # (B, N)
        if not loop:
            rows = torch.arange(B, device=x.device)
            cols = torch.arange(start, end, device=x.device)
            dist_sq[rows, cols] = float("inf")
        mask = dist_sq < r_sq
        i_local, j = torch.where(mask)
        i_parts.append(i_local + start)
        j_parts.append(j)
        v_parts.append(vec[i_local, j])

    if not i_parts or sum(p.numel() for p in i_parts) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=x.device),
            torch.zeros(0, x.shape[-1], device=x.device, dtype=x.dtype),
        )

    edge_index = torch.stack([torch.cat(i_parts), torch.cat(j_parts)], dim=0)
    edge_vec = torch.cat(v_parts)
    return edge_index, edge_vec


def periodic_radius_graph_cell_list(
    x: Tensor,
    r: float,
    cell: Tensor,
    *,
    loop: bool = False,
) -> tuple[Tensor, Tensor]:
    """O(N) periodic radius graph via a 3D cell list.

    Same interface and semantics as periodic_radius_graph_chunked:
    edge_index[0]=i, edge_index[1]=j, edge_vec = min_image(x[j] - x[i]);
    self-loops excluded unless ``loop=True``.

    Partitions space into cubic bins of side ≥ r. Each atom only checks
    atoms in its own bin + 26 neighbors (3×3×3 = 27-cell neighborhood),
    which is mathematically guaranteed to contain every true neighbor.
    Total work is O(N * atoms_per_bin * 27), independent of N for fixed
    density — vs O(N^2) for the chunked version.

    Precondition: the cell must be orthogonal (diagonal) and each axis
    length must be ≥ 3*r so the 27-cell neighborhood is well-defined
    under PBCs. A ``ValueError`` is raised otherwise; callers should
    fall back to :func:`periodic_radius_graph_chunked`.

    Args:
        x: (N, 3) positions.
        r: cutoff radius.
        cell: (3, 3) orthogonal cell matrix.
        loop: include self-edges if True.

    Returns:
        edge_index: (2, E) directed edges.
        edge_vec: (E, 3) minimum-image displacement vectors.
    """
    N = x.shape[0]
    device = x.device
    dtype = x.dtype

    box = torch.stack([cell[0, 0], cell[1, 1], cell[2, 2]])  # (3,)
    off_diag = cell - torch.diag(box)
    if off_diag.abs().max() > 1e-6 * box.abs().max():
        raise ValueError(
            "periodic_radius_graph_cell_list requires an orthogonal (diagonal) "
            "cell; use periodic_radius_graph_chunked for triclinic cells."
        )

    n_bins = torch.clamp((box / r).floor().long(), min=1)
    if (n_bins < 3).any():
        raise ValueError(
            f"Cell too small for cell-list search: n_bins={n_bins.tolist()} "
            f"at r={r}, box={box.tolist()}. Each axis must be ≥ 3*r."
        )
    nb0 = int(n_bins[0].item()); nb1 = int(n_bins[1].item()); nb2 = int(n_bins[2].item())
    total_bins = nb0 * nb1 * nb2
    inv_cell = torch.diag(1.0 / box)

    # Fractional coords → bin indices.
    frac = x / box
    frac = frac - torch.floor(frac)
    bin_xyz = (frac * n_bins.to(dtype)).long().clamp(max=(n_bins - 1))  # (N, 3)
    bin_id = bin_xyz[:, 0] * (nb1 * nb2) + bin_xyz[:, 1] * nb2 + bin_xyz[:, 2]

    # Build padded (total_bins, max_count) atom-index table.
    bin_counts = torch.bincount(bin_id, minlength=total_bins)
    max_count = int(bin_counts.max().item())
    sort_idx = torch.argsort(bin_id)
    bin_id_sorted = bin_id[sort_idx]
    bin_starts = torch.zeros(total_bins + 1, dtype=torch.long, device=device)
    bin_starts[1:] = bin_counts.cumsum(0)
    bin_rank_sorted = torch.arange(N, device=device) - bin_starts[bin_id_sorted]
    bins_padded = torch.full(
        (total_bins, max_count), -1, dtype=torch.long, device=device,
    )
    bins_padded[bin_id_sorted, bin_rank_sorted] = sort_idx

    # 27 neighbor bins per atom, wrapped under PBCs.
    offs = torch.arange(-1, 2, device=device)
    offsets = torch.stack(
        torch.meshgrid(offs, offs, offs, indexing="ij"), dim=-1,
    ).reshape(-1, 3)                                              # (27, 3)
    nb_xyz = bin_xyz.unsqueeze(1) + offsets.unsqueeze(0)          # (N, 27, 3)
    nb_xyz = torch.remainder(nb_xyz, n_bins.view(1, 1, 3))
    nb_bin = nb_xyz[..., 0] * (nb1 * nb2) + nb_xyz[..., 1] * nb2 + nb_xyz[..., 2]

    # Gather candidates, mask padding/self.
    cand = bins_padded[nb_bin].view(N, -1)                        # (N, 27*max_count)
    M = cand.shape[1]
    centers = torch.arange(N, device=device).unsqueeze(1).expand(-1, M)
    valid = (cand >= 0) if loop else ((cand >= 0) & (cand != centers))
    i_atoms = centers[valid]
    j_atoms = cand[valid]

    # MI displacement + cutoff filter.
    vec = x[j_atoms] - x[i_atoms]
    vec = vec - torch.round(vec @ inv_cell) @ cell
    dist_sq = (vec * vec).sum(dim=-1)
    keep = dist_sq < r * r

    if not keep.any():
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, 3, dtype=dtype, device=device),
        )
    edge_index = torch.stack([i_atoms[keep], j_atoms[keep]], dim=0)
    edge_vec = vec[keep]
    return edge_index, edge_vec


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
