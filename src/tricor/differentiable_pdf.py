"""
Differentiable pair distribution function (PDF) and angular distribution
function (ADF) for use as gradient-based guidance in score-based generative
models (GLASS-style conditional denoising).

Follows tricor's measure_g3 computational structure:
  - Per-center-atom loop with neighbor filtering within r_max
  - Species-grouped neighbor tables (vector_table, radius_sq_table)
  - Rooted triplet convention with unordered neighbors (neigh1 <= neigh2)
  - Same symmetry handling: diagonal exclusion for same-species pairs,
    double-counting for cross-species pairs

Key differences from tricor:
  - Gaussian kernel smoothing instead of hard histogram binning (for
    differentiability / autograd compatibility)
  - Outputs g2(r) and ADF(phi) separately instead of the full
    g3(r1, r2, phi)
  - Minimum-image convention instead of explicit cell tiling (requires
    cell dimensions > 2 * r_max in all directions)

Reference: Guo & Schwalbe-Koda, arXiv:2603.23210 (GLASS), Sec. S2.4

Usage:
    calc = DifferentiablePDFADF(r_max=8.0, r_step=0.05, phi_num_bins=90,
                                 sigma_r=0.15, sigma_phi=0.1, species=[14])
    g2, adf = calc.compute(positions, species, cell)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def minimum_image_displacement(
    pos_i: torch.Tensor,
    pos_j: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Compute displacement vectors pos_j - pos_i using minimum-image convention.

    Converts to fractional coordinates, wraps into [-0.5, 0.5), and maps
    back to Cartesian space.

    Args:
        pos_i: (..., 3) positions of atoms i
        pos_j: (..., 3) positions of atoms j (broadcastable with pos_i)
        cell: (3, 3) cell matrix where rows are lattice vectors

    Returns:
        Displacement vectors with minimum image applied.
    """
    inv_cell = torch.linalg.inv(cell)
    delta_cart = pos_j - pos_i
    delta_frac = delta_cart @ inv_cell.T
    delta_frac = delta_frac - torch.round(delta_frac)
    return delta_frac @ cell


class DifferentiablePDFADF(nn.Module):
    """Differentiable element-resolved PDF g2(r) and angular distribution ADF(phi).

    Follows tricor's measure_g3 computational structure:
      1. Set up radial and angular grids
      2. Build species index and rooted triplet types (g3_index, g3_lookup)
      3. Loop over center species, then over individual center atoms
      4. For each center: build vector_table of neighbors within r_max,
         grouped by species
      5. Accumulate g2 from neighbor distances (Gaussian-smoothed)
      6. For each triplet type belonging to this center species:
         compute angles among filtered neighbors from vector_table
      7. Accumulate ADF from angles (Gaussian-smoothed)
      8. Handle symmetry: diagonal exclusion for same-species neighbor
         pairs, double-counting for cross-species pairs

    Uses minimum-image convention instead of explicit cell tiling.
    Requires cell dimensions > 2 * r_max in all directions.
    """

    def __init__(
        self,
        r_max: float,
        r_step: float,
        phi_num_bins: int = 90,
        sigma_r: float = 0.15,
        sigma_phi: float = 0.1,
        species: Optional[list[int]] = None,
    ):
        """
        Args:
            r_max: Maximum radial distance in Angstroms (same cutoff for
                   both g2 and ADF neighbor finding).
            r_step: Radial bin width in Angstroms. r_max / r_step must be
                    an integer.
            phi_num_bins: Number of angular bins spanning 0 to pi.
            sigma_r: Gaussian bandwidth for radial smoothing in Angstroms.
            sigma_phi: Gaussian bandwidth for angular smoothing in radians.
            species: List of atomic numbers present in the system. If None,
                     treats all atoms as one type.
        """
        super().__init__()
        self.r_max = r_max
        self.r_step = r_step
        self.sigma_r = sigma_r
        self.sigma_phi = sigma_phi

        # --- Radial grid (same as tricor) ---
        num_r_float = r_max / r_step
        num_r = int(round(num_r_float))
        if abs(num_r_float - num_r) > 1e-6:
            raise ValueError("r_max must be divisible by r_step.")
        self.num_r = num_r
        r_grid = torch.arange(num_r, dtype=torch.float64) * r_step + 0.5 * r_step
        self.register_buffer("r_grid", r_grid)

        # --- Angular grid (same as tricor) ---
        self.phi_num_bins = phi_num_bins
        phi_edges = torch.linspace(0.0, math.pi, phi_num_bins + 1, dtype=torch.float64)
        phi_step = phi_edges[1] - phi_edges[0]
        phi_grid = phi_edges[:-1] + 0.5 * phi_step
        self.register_buffer("phi_grid", phi_grid)
        self.phi_step = float(phi_step)

        # --- Species setup ---
        if species is not None:
            species_sorted = sorted(species)
            self.register_buffer(
                "species_list", torch.tensor(species_sorted, dtype=torch.long)
            )
            self.num_species = len(species_sorted)
        else:
            self.species_list = None
            self.num_species = 1

        # --- Rooted triplet index with unordered neighbors (same as tricor) ---
        # [center, neigh_1, neigh_2] with neigh_1 <= neigh_2
        # For a binary system this gives:
        # [0,0,0], [0,0,1], [0,1,1], [1,0,0], [1,0,1], [1,1,1]
        g3_index = []
        for center_ind in range(self.num_species):
            for neigh1_ind in range(self.num_species):
                for neigh2_ind in range(neigh1_ind, self.num_species):
                    g3_index.append([center_ind, neigh1_ind, neigh2_ind])
        self.register_buffer(
            "g3_index", torch.tensor(g3_index, dtype=torch.long)
        )
        self.num_triplets = len(g3_index)

        # Fast lookup from (center, neigh1, neigh2) -> triplet channel (same as tricor).
        # Neighbor order is symmetrized so (n1, n2) and (n2, n1) map to the same channel.
        g3_lookup = -torch.ones(
            (self.num_species, self.num_species, self.num_species),
            dtype=torch.long,
        )
        for tri_idx, (c, n1, n2) in enumerate(g3_index):
            g3_lookup[c, n1, n2] = tri_idx
            g3_lookup[c, n2, n1] = tri_idx
        self.register_buffer("g3_lookup", g3_lookup)

        # Precompute which triplet types belong to each center species (same as tricor)
        self._triplets_by_center = [
            [i for i, (c, _, _) in enumerate(g3_index) if c == center_ind]
            for center_ind in range(self.num_species)
        ]

    def compute(
        self,
        positions: torch.Tensor,
        species: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute differentiable g2(r) and ADF(phi).

        Follows the same loop structure as tricor's measure_g3:
        loop over center species -> loop over center atoms -> build
        vector_table of filtered neighbors -> accumulate g2 -> loop
        over triplet types -> compute angles -> accumulate ADF.

        Args:
            positions: (N, 3) atomic positions (requires_grad=True for
                       gradient-based guidance).
            species: (N,) integer tensor of atomic numbers.
            cell: (3, 3) cell matrix (rows = lattice vectors).

        Returns:
            g2: (num_species, num_species, num_r) Gaussian-smoothed pair
                distance histogram.
            adf: (num_triplets, phi_num_bins) Gaussian-smoothed bond angle
                 histogram.
        """
        N = positions.shape[0]
        device = positions.device
        dtype = positions.dtype
        r_max_sq = self.r_max ** 2
        zero_tol = max(1e-12, (1e-9 * self.r_step) ** 2)
        eps = 1e-7

        # --- Map atomic numbers to local species indices ---
        # (like tricor's origin_species_index = np.searchsorted(self.species, numbers))
        if self.species_list is not None:
            sp_idx = torch.zeros(N, dtype=torch.long, device=device)
            for local_idx, Z in enumerate(self.species_list):
                sp_idx[species == Z] = local_idx
        else:
            sp_idx = torch.zeros(N, dtype=torch.long, device=device)

        # --- Group positions by species ---
        # origin indices by species (like tricor's origin_xyz_by_species)
        origin_indices_by_species = [
            torch.where(sp_idx == ind0)[0] for ind0 in range(self.num_species)
        ]
        # All positions by species (like tricor's xyz_all)
        # In tricor these are tiled supercell coordinates; here we use
        # minimum-image convention instead, so xyz_all == origin positions
        xyz_all_indices = origin_indices_by_species

        # --- Initialize accumulators ---
        # g2: (num_species, num_species, num_r) like tricor's g2count
        # adf: (num_triplets, phi_num_bins) — angular projection of tricor's g3count
        g2 = torch.zeros(
            self.num_species, self.num_species, self.num_r,
            device=device, dtype=dtype,
        )
        adf = torch.zeros(
            self.num_triplets, self.phi_num_bins,
            device=device, dtype=dtype,
        )

        r_grid = self.r_grid  # (num_r,)
        phi_grid = self.phi_grid  # (phi_num_bins,)
        r_norm = self.r_step / (math.sqrt(2 * math.pi) * self.sigma_r)
        phi_norm = self.phi_step / (math.sqrt(2 * math.pi) * self.sigma_phi)

        # --- Main loop: center species, then origin atoms ---
        # (follows tricor's measure_g3 structure exactly)
        for ind0 in range(self.num_species):
            for origin_idx in origin_indices_by_species[ind0]:
                xyz0 = positions[origin_idx]  # (3,) single center atom

                # Build neighbor tables by species (like tricor's vector_table)
                vector_table: list[torch.Tensor] = []
                radius_sq_table: list[torch.Tensor] = []

                for indn in range(self.num_species):
                    neighbor_positions = positions[xyz_all_indices[indn]]  # (M, 3)

                    # Displacement vectors from center to neighbors
                    # (like tricor's: vectors = xyz_all[indn] - xyz0)
                    vectors = minimum_image_displacement(
                        xyz0.unsqueeze(0), neighbor_positions, cell
                    )  # (M, 3)

                    radius_sq = torch.sum(vectors * vectors, dim=-1)  # (M,)

                    # Filter: keep within r_max, exclude self
                    # (like tricor's: keep = (radius_sq > zero_tol) & (radius_sq < r_max_sq))
                    keep = (radius_sq > zero_tol) & (radius_sq < r_max_sq)
                    vectors = vectors[keep]
                    radius_sq = radius_sq[keep]
                    radius = torch.sqrt(radius_sq)

                    vector_table.append(vectors)
                    radius_sq_table.append(radius_sq)

                    # Accumulate g2 with Gaussian kernel
                    # (like tricor's: self.g2count[ind0, indn] += bincount(...))
                    if radius.numel() > 0:
                        r_kernel = torch.exp(
                            -0.5 * ((radius.unsqueeze(-1) - r_grid.unsqueeze(0)) / self.sigma_r) ** 2
                        ) * r_norm  # (num_neighbors, num_r)
                        g2[ind0, indn] = g2[ind0, indn] + r_kernel.sum(dim=0)

                # Compute ADF for each triplet type with this center
                # (like tricor's: for ind in triplets_by_center[ind0])
                for ind in self._triplets_by_center[ind0]:
                    _, ind1, ind2 = self.g3_index[ind].tolist()

                    v01 = vector_table[ind1]
                    v02 = vector_table[ind2]
                    r01_sq = radius_sq_table[ind1]
                    r02_sq = radius_sq_table[ind2]

                    if v01.shape[0] == 0 or v02.shape[0] == 0:
                        continue

                    # Angle computation (same as tricor)
                    # dot = v01 @ v02.T
                    # denom = sqrt(r01_sq[:, None] * r02_sq[None, :])
                    # cos_phi = clip(dot / denom, -1, 1)
                    dot = v01 @ v02.T  # (k1, k2)
                    denom = torch.sqrt(r01_sq.unsqueeze(1) * r02_sq.unsqueeze(0))
                    cos_phi = torch.clamp(dot / denom, -1.0 + eps, 1.0 - eps)
                    phi = torch.acos(cos_phi)  # (k1, k2)

                    # Same-species neighbor pair: exclude diagonal
                    # (like tricor's: if ind1 == ind2: fill_diagonal(valid, False))
                    if ind1 == ind2:
                        k = phi.shape[0]
                        diag_mask = ~torch.eye(k, dtype=torch.bool, device=device)
                        phi_vals = phi[diag_mask]
                    else:
                        phi_vals = phi.reshape(-1)

                    if phi_vals.numel() == 0:
                        continue

                    # Gaussian kernel for angular bins
                    # (like tricor's: counts = bincount(phi_bin.ravel(), ...))
                    phi_kernel = torch.exp(
                        -0.5 * ((phi_vals.unsqueeze(-1) - phi_grid.unsqueeze(0)) / self.sigma_phi) ** 2
                    ) * phi_norm  # (num_angles, phi_num_bins)
                    adf[ind] = adf[ind] + phi_kernel.sum(dim=0)

                    # Cross-species symmetry: count both orderings
                    # (like tricor's: if ind1 != ind2: ... rr_index_sym ...)
                    # In tricor this swaps the radial bin indices. For the ADF
                    # (no radial binning), the angle is identical so we add
                    # the same contribution again.
                    if ind1 != ind2:
                        adf[ind] = adf[ind] + phi_kernel.sum(dim=0)

        return g2, adf

    @property
    def pair_labels(self) -> list[str]:
        """Human-readable labels for g2 channels: 'center-neighbor'.

        Follows tricor's g2_labels convention: all (center, neighbor) pairs
        including both orderings.
        """
        if self.species_list is not None:
            sp = self.species_list.tolist()
            return [
                f"{sp[i]}-{sp[j]}"
                for i in range(self.num_species)
                for j in range(self.num_species)
            ]
        return ["X-X"]

    @property
    def triplet_labels(self) -> list[str]:
        """Human-readable labels for ADF channels: 'neigh1-center-neigh2'.

        Follows tricor's pair_labels convention for rooted triplets.
        """
        if self.species_list is not None:
            sp = self.species_list.tolist()
            return [
                f"{sp[n1]}-{sp[c]}-{sp[n2]}"
                for c, n1, n2 in self.g3_index.tolist()
            ]
        return ["X-X-X"]


class DifferentiableSpectralLoss(nn.Module):
    """Combined PDF + ADF guidance loss for GLASS-style conditional denoising.

    Computes a weighted sum of L2 losses between predicted and target
    spectral observables. The gradient with respect to atomic positions
    provides the guidance score for conditional denoising:
        s_guide = -grad_{x} L_spec(F(x), y)
    """

    def __init__(
        self,
        calculator: DifferentiablePDFADF,
        pdf_weight: float = 1.0,
        adf_weight: float = 1.0,
    ):
        super().__init__()
        self.calc = calculator
        self.pdf_weight = pdf_weight
        self.adf_weight = adf_weight

    def forward(
        self,
        positions: torch.Tensor,
        species: torch.Tensor,
        cell: torch.Tensor,
        target_g2: torch.Tensor,
        target_adf: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute guidance loss.

        Args:
            positions: (N, 3) atomic positions (requires_grad=True).
            species: (N,) atomic numbers.
            cell: (3, 3) cell matrix.
            target_g2: (num_species, num_species, num_r) target PDF.
            target_adf: (num_triplets, phi_num_bins) target ADF.

        Returns:
            loss: Scalar total loss.
            components: Dict with 'pdf_loss' and 'adf_loss' for diagnostics.
        """
        if self.adf_weight == 0.0 and hasattr(self.calc, "compute_g2_only"):
            g2, _ = self.calc.compute_g2_only(positions, species, cell)
            adf_loss = torch.zeros((), device=g2.device, dtype=g2.dtype)
        else:
            g2, adf_pred = self.calc.compute(positions, species, cell)
            adf_loss = torch.mean((adf_pred - target_adf) ** 2)

        pdf_loss = torch.mean((g2 - target_g2) ** 2)
        loss = self.pdf_weight * pdf_loss + self.adf_weight * adf_loss

        return loss, {"pdf_loss": pdf_loss.detach(), "adf_loss": adf_loss.detach()}

    def compute_guidance(
        self,
        positions: torch.Tensor,
        species: torch.Tensor,
        cell: torch.Tensor,
        target_g2: torch.Tensor,
        target_adf: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute guidance score (negative gradient of loss w.r.t. positions).

        This is the s_guide term in the GLASS conditional denoising algorithm:
            s_guide = -grad_{x} L_spec(F(x), y)

        Args:
            positions: (N, 3) atomic positions (will be detached and
                       re-attached with grad).
            species: (N,) atomic numbers.
            cell: (3, 3) cell matrix.
            target_g2: (num_species, num_species, num_r) target PDF.
            target_adf: (num_triplets, phi_num_bins) target ADF.

        Returns:
            guidance: (N, 3) per-atom guidance vectors.
            components: Dict with loss components for diagnostics.
        """
        pos = positions.detach().requires_grad_(True)
        loss, components = self.forward(pos, species, cell, target_g2, target_adf)
        guidance = -torch.autograd.grad(loss, pos)[0]
        return guidance, components
