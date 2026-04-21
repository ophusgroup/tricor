"""First-shell coordination targets extracted from crystalline reference cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.atoms import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import neighbor_list

from .g3 import _EPS


def _cell_face_spacings(cell_matrix: np.ndarray) -> np.ndarray:
    """Return periodic face spacings for a cell spanned by row vectors."""
    inverse = np.linalg.inv(np.asarray(cell_matrix, dtype=np.float64))
    return 1.0 / np.maximum(np.linalg.norm(inverse, axis=0), _EPS)


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    """Return a normalized 1D Gaussian kernel."""
    sigma_bins = float(sigma_bins)
    if sigma_bins <= _EPS:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma_bins, _EPS)) ** 2)
    kernel /= max(np.sum(kernel), _EPS)
    return kernel


def _smooth_histogram(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Smooth a histogram using a small Gaussian kernel."""
    kernel = _gaussian_kernel(sigma_bins)
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def _first_local_maximum(values: np.ndarray) -> int:
    """Return the first prominent local maximum, falling back to the global one."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0
    if values.size == 1:
        return 0
    peak_threshold = 0.25 * float(np.max(values))
    for index in range(1, values.size - 1):
        if values[index] >= peak_threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            return int(index)
    return int(np.argmax(values))


def _infer_shell_window(
    distances: np.ndarray,
    *,
    hist_step: float,
    smooth_sigma_bins: float,
) -> tuple[float, float, float, float, float]:
    """Infer a first-shell window from reference pair distances."""
    distances = np.sort(np.asarray(distances, dtype=np.float64))
    if distances.size == 0:
        return 0.0, 0.0, 0.0, hist_step, 0.0

    upper = float(np.max(distances) + hist_step)
    edges = np.arange(0.0, upper + hist_step, hist_step, dtype=np.float64)
    if edges.size < 3:
        edges = np.array([0.0, hist_step, 2.0 * hist_step], dtype=np.float64)
    hist, _ = np.histogram(distances, bins=edges)
    occupied = np.flatnonzero(hist > 0)
    if occupied.size == 0:
        return 0.0, 0.0, 0.0, hist_step, 0.0

    first_occ = int(occupied[0])
    left_index = max(first_occ - 1, 0)
    right_index = hist.size - 1
    zero_run = 0
    zero_right_index: int | None = None
    for index in range(first_occ + 1, hist.size):
        if hist[index] == 0:
            zero_run += 1
            if zero_run >= 2:
                zero_right_index = index - zero_run + 1
                break
        else:
            zero_run = 0
    if zero_right_index is not None:
        right_index = min(hist.size - 1, zero_right_index)
    else:
        smooth = _smooth_histogram(hist, sigma_bins=smooth_sigma_bins)
        peak_index = _first_local_maximum(smooth[first_occ:]) + first_occ
        peak_value = float(max(smooth[peak_index], _EPS))
        cutoff_value = 0.12 * peak_value
        right_index = peak_index
        while right_index < smooth.size - 1 and smooth[right_index] > cutoff_value:
            right_index += 1

    r_inner = float(edges[max(left_index, 0)])
    r_outer = float(edges[min(right_index + 1, edges.size - 1)])
    in_shell = distances[(distances >= r_inner) & (distances <= r_outer)]
    if in_shell.size == 0:
        in_shell = distances

    r_peak = float(np.mean(in_shell))
    sigma_r = float(max(np.std(in_shell), hist_step))
    hard_min = float(max(0.0, r_inner - 1.5 * sigma_r))
    return hard_min, r_inner, r_peak, sigma_r, r_outer


@dataclass(frozen=True)
class CoordinationShellTarget:
    """Species-aware first-shell coordination targets extracted from a crystal."""

    atoms: Atoms
    label: str
    species: np.ndarray
    species_labels: tuple[str, ...]
    phi_num_bins: int
    phi_edges: np.ndarray
    phi: np.ndarray
    phi_deg: np.ndarray
    angle_index: np.ndarray
    angle_lookup: np.ndarray
    pair_hard_min: np.ndarray
    pair_inner: np.ndarray
    pair_peak: np.ndarray
    pair_sigma: np.ndarray
    pair_outer: np.ndarray
    pair_mask: np.ndarray
    coordination_target: np.ndarray
    coordination_std: np.ndarray
    angle_target: np.ndarray
    angle_pair_mass_target: np.ndarray
    angle_mode_deg: np.ndarray
    # Per-triplet mask that controls whether ``shell_relax`` installs an
    # angle spring for that triplet type.  Bond-distance springs are
    # unaffected.  Useful for multi-modal shells (SrO\u2081\u2082
    # cuboctahedron, Sr-O-Sr triplets in SrTiO\u2083) where forcing a
    # single ``angle_mode_deg`` would strain pairs at the other modes.
    angle_enabled_mask: np.ndarray
    motif_center_species: np.ndarray
    motif_neighbor_species: tuple[np.ndarray, ...]
    motif_neighbor_vectors: tuple[np.ndarray, ...]
    max_pair_outer: float
    max_pair_outer_by_center: np.ndarray
    summary: dict[str, object]

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        *,
        phi_num_bins: int = 72,
        shell_hist_step: float = 0.05,
        shell_smooth_sigma_bins: float = 1.2,
        extract_cutoff: float | None = None,
        label: str | None = None,
    ) -> "CoordinationShellTarget":
        """Extract first-shell count, radius, and angle targets from reference atoms."""
        atoms = atoms.copy()
        species = np.unique(np.asarray(atoms.numbers, dtype=np.int64))
        num_species = int(species.size)
        species_index = np.searchsorted(species, np.asarray(atoms.numbers, dtype=np.int64))
        species_labels = tuple(chemical_symbols[int(spec)] for spec in species)

        phi_num_bins = int(phi_num_bins)
        if phi_num_bins <= 0:
            raise ValueError("phi_num_bins must be positive.")
        phi_edges = np.linspace(0.0, np.pi, phi_num_bins + 1, dtype=np.float64)
        phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        phi_deg = np.rad2deg(phi)

        angle_index = []
        angle_lookup = -np.ones((num_species, num_species, num_species), dtype=np.intp)
        for center_index in range(num_species):
            for neigh1_index in range(num_species):
                for neigh2_index in range(neigh1_index, num_species):
                    triplet_index = len(angle_index)
                    angle_index.append((center_index, neigh1_index, neigh2_index))
                    angle_lookup[center_index, neigh1_index, neigh2_index] = triplet_index
                    angle_lookup[center_index, neigh2_index, neigh1_index] = triplet_index
        angle_index = np.asarray(angle_index, dtype=np.intp)

        cell_matrix = np.asarray(atoms.cell.array, dtype=np.float64)
        cell_face_spacings = _cell_face_spacings(cell_matrix)
        cell_lengths = np.linalg.norm(cell_matrix, axis=1)
        default_probe_cutoff = max(8.0, 1.5 * float(np.max(cell_lengths)), 1.2 * float(np.max(cell_face_spacings)))
        if extract_cutoff is None:
            probe_cutoff = default_probe_cutoff
        else:
            probe_cutoff = float(extract_cutoff)
        if probe_cutoff <= 0:
            raise ValueError("extract_cutoff must be positive when provided.")

        i, j, d, D = neighbor_list(
            "ijdD",
            atoms,
            probe_cutoff,
            self_interaction=False,
        )
        if d.size == 0:
            raise ValueError("Could not detect any periodic neighbors in the reference cell.")

        nearest = np.full(len(atoms), np.inf, dtype=np.float64)
        np.minimum.at(nearest, i.astype(np.intp), d.astype(np.float64))
        finite_nearest = nearest[np.isfinite(nearest)]
        if finite_nearest.size == 0:
            raise ValueError("Failed to infer nearest-neighbor distances from the reference cell.")
        nearest_reference = float(np.median(finite_nearest))
        if extract_cutoff is None:
            probe_cutoff = min(default_probe_cutoff, max(3.8 * nearest_reference, nearest_reference + 2.0))
            i, j, d, D = neighbor_list(
                "ijdD",
                atoms,
                probe_cutoff,
                self_interaction=False,
            )

        center_species = species_index[np.asarray(i, dtype=np.intp)]
        neighbor_species = species_index[np.asarray(j, dtype=np.intp)]
        pair_hard_min = np.zeros((num_species, num_species), dtype=np.float64)
        pair_inner = np.zeros((num_species, num_species), dtype=np.float64)
        pair_peak = np.zeros((num_species, num_species), dtype=np.float64)
        pair_sigma = np.full((num_species, num_species), float(shell_hist_step), dtype=np.float64)
        pair_outer = np.zeros((num_species, num_species), dtype=np.float64)
        pair_mask = np.zeros((num_species, num_species), dtype=bool)

        for species_a in range(num_species):
            for species_b in range(species_a, num_species):
                mask = (
                    ((center_species == species_a) & (neighbor_species == species_b))
                    | ((center_species == species_b) & (neighbor_species == species_a))
                )
                distances = np.asarray(d[mask], dtype=np.float64)
                if distances.size == 0:
                    continue
                hard_min, r_inner, r_peak, sigma_r, r_outer = _infer_shell_window(
                    distances,
                    hist_step=float(shell_hist_step),
                    smooth_sigma_bins=float(shell_smooth_sigma_bins),
                )
                pair_hard_min[species_a, species_b] = hard_min
                pair_hard_min[species_b, species_a] = hard_min
                pair_inner[species_a, species_b] = r_inner
                pair_inner[species_b, species_a] = r_inner
                pair_peak[species_a, species_b] = r_peak
                pair_peak[species_b, species_a] = r_peak
                pair_sigma[species_a, species_b] = sigma_r
                pair_sigma[species_b, species_a] = sigma_r
                pair_outer[species_a, species_b] = r_outer
                pair_outer[species_b, species_a] = r_outer
                pair_mask[species_a, species_b] = True
                pair_mask[species_b, species_a] = True

        coordination_target = np.zeros((num_species, num_species), dtype=np.float64)
        coordination_std = np.zeros((num_species, num_species), dtype=np.float64)
        for center_ind in range(num_species):
            centers = np.flatnonzero(species_index == center_ind)
            if centers.size == 0:
                continue
            for neigh_ind in range(num_species):
                if not pair_mask[center_ind, neigh_ind]:
                    continue
                r_inner = pair_inner[center_ind, neigh_ind]
                r_outer = pair_outer[center_ind, neigh_ind]
                mask = (
                    (center_species == center_ind)
                    & (neighbor_species == neigh_ind)
                    & (d >= r_inner)
                    & (d <= r_outer)
                )
                counts = np.bincount(np.asarray(i[mask], dtype=np.intp), minlength=len(atoms))
                centered_counts = counts[centers]
                coordination_target[center_ind, neigh_ind] = float(np.mean(centered_counts))
                coordination_std[center_ind, neigh_ind] = float(np.std(centered_counts))

        angle_target = np.zeros((angle_index.shape[0], phi_num_bins), dtype=np.float64)
        angle_pair_mass_target = np.zeros(angle_index.shape[0], dtype=np.float64)
        angle_mode_deg = np.zeros(angle_index.shape[0], dtype=np.float64)
        # Default: angle springs enabled for every triplet type.  Use
        # ``with_angle_triplets`` / ``without_angle_triplets`` to mask
        # specific triplets for multi-modal shells.
        angle_enabled_mask = np.ones(angle_index.shape[0], dtype=bool)
        motif_center_species: list[int] = []
        motif_neighbor_species: list[np.ndarray] = []
        motif_neighbor_vectors: list[np.ndarray] = []

        neighbors_by_center: list[dict[str, np.ndarray]] = []
        for atom_index in range(len(atoms)):
            mask = np.asarray(i, dtype=np.intp) == int(atom_index)
            neighbors_by_center.append(
                {
                    "neighbor_index": np.asarray(j[mask], dtype=np.intp),
                    "neighbor_species": neighbor_species[mask].astype(np.intp, copy=False),
                    "vectors": np.asarray(D[mask], dtype=np.float64),
                    "distance": np.asarray(d[mask], dtype=np.float64),
                }
            )

        for center_atom in range(len(atoms)):
            center_species_index = int(species_index[center_atom])
            local = neighbors_by_center[center_atom]
            if local["neighbor_index"].size == 0:
                motif_center_species.append(center_species_index)
                motif_neighbor_species.append(np.empty(0, dtype=np.intp))
                motif_neighbor_vectors.append(np.empty((0, 3), dtype=np.float64))
                continue

            keep = np.zeros(local["neighbor_index"].shape[0], dtype=bool)
            for neighbor_ind, neighbor_species_index in enumerate(local["neighbor_species"]):
                if not pair_mask[center_species_index, int(neighbor_species_index)]:
                    continue
                radius = float(local["distance"][neighbor_ind])
                keep[neighbor_ind] = (
                    radius >= float(pair_inner[center_species_index, int(neighbor_species_index)])
                    and radius <= float(pair_outer[center_species_index, int(neighbor_species_index)])
                )
            local_species = local["neighbor_species"][keep].astype(np.intp, copy=False)
            local_vectors = local["vectors"][keep].astype(np.float64, copy=False)
            if local_species.size:
                radius = np.linalg.norm(local_vectors, axis=1)
                order = np.lexsort(
                    (
                        local_vectors[:, 2],
                        local_vectors[:, 1],
                        local_vectors[:, 0],
                        radius,
                        local_species,
                    )
                )
                local_species = local_species[order]
                local_vectors = local_vectors[order]
            motif_center_species.append(center_species_index)
            motif_neighbor_species.append(np.array(local_species, dtype=np.intp, copy=True))
            motif_neighbor_vectors.append(np.array(local_vectors, dtype=np.float64, copy=True))

        for center_atom in range(len(atoms)):
            center_species_index = int(species_index[center_atom])
            local = neighbors_by_center[center_atom]
            if local["neighbor_index"].size == 0:
                continue
            for triplet_index, (_, species_1, species_2) in enumerate(angle_index):
                if angle_index[triplet_index, 0] != center_species_index:
                    continue
                inner_1 = pair_inner[center_species_index, species_1]
                outer_1 = pair_outer[center_species_index, species_1]
                inner_2 = pair_inner[center_species_index, species_2]
                outer_2 = pair_outer[center_species_index, species_2]

                mask_1 = (
                    (local["neighbor_species"] == species_1)
                    & (local["distance"] >= inner_1)
                    & (local["distance"] <= outer_1)
                )
                mask_2 = (
                    (local["neighbor_species"] == species_2)
                    & (local["distance"] >= inner_2)
                    & (local["distance"] <= outer_2)
                )
                if not np.any(mask_1) or not np.any(mask_2):
                    continue

                v1 = local["vectors"][mask_1]
                v2 = local["vectors"][mask_2]
                r1_sq = np.einsum("ij,ij->i", v1, v1)
                r2_sq = np.einsum("ij,ij->i", v2, v2)
                if species_1 == species_2:
                    if v1.shape[0] < 2:
                        continue
                    dot = v1 @ v2.T
                    denom = np.sqrt(np.maximum(r1_sq[:, None] * r2_sq[None, :], _EPS))
                    cos_phi = np.clip(dot / denom, -1.0, 1.0)
                    phi_bin = np.floor(np.arccos(cos_phi) / (phi_edges[1] - phi_edges[0])).astype(np.intp)
                    np.clip(phi_bin, 0, phi_num_bins - 1, out=phi_bin)
                    upper = np.triu_indices(phi_bin.shape[0], k=1)
                    bins = phi_bin[upper]
                else:
                    dot = v1 @ v2.T
                    denom = np.sqrt(np.maximum(r1_sq[:, None] * r2_sq[None, :], _EPS))
                    cos_phi = np.clip(dot / denom, -1.0, 1.0)
                    phi_bin = np.floor(np.arccos(cos_phi) / (phi_edges[1] - phi_edges[0])).astype(np.intp)
                    np.clip(phi_bin, 0, phi_num_bins - 1, out=phi_bin)
                    bins = phi_bin.ravel()
                if bins.size == 0:
                    continue
                angle_target[triplet_index] += np.bincount(bins, minlength=phi_num_bins)
                angle_pair_mass_target[triplet_index] += float(bins.size)

        centers_per_species = np.bincount(species_index, minlength=num_species).astype(np.float64)
        for triplet_index, (center_ind, _, _) in enumerate(angle_index):
            mass = float(angle_pair_mass_target[triplet_index])
            if mass > 0.0:
                angle_target[triplet_index] /= mass
            angle_pair_mass_target[triplet_index] = mass / max(float(centers_per_species[center_ind]), 1.0)
            angle_mode_deg[triplet_index] = float(phi_deg[int(np.argmax(angle_target[triplet_index]))])

        max_pair_outer = float(np.max(pair_outer[pair_mask])) if np.any(pair_mask) else 0.0
        max_pair_outer_by_center = np.zeros(num_species, dtype=np.float64)
        for center_ind in range(num_species):
            row = pair_outer[center_ind][pair_mask[center_ind]]
            max_pair_outer_by_center[center_ind] = float(np.max(row)) if row.size else 0.0

        summary = {
            "num_atoms": len(atoms),
            "num_species": num_species,
            "phi_num_bins": phi_num_bins,
            "extract_cutoff": float(probe_cutoff),
            "nearest_reference": nearest_reference,
            "max_pair_outer": max_pair_outer,
            "num_triplets": int(angle_index.shape[0]),
            "num_motifs": int(len(motif_center_species)),
        }

        return cls(
            atoms=atoms,
            label=label or "coordination-shell-target",
            species=species.astype(np.int64, copy=False),
            species_labels=species_labels,
            phi_num_bins=phi_num_bins,
            phi_edges=phi_edges,
            phi=phi,
            phi_deg=phi_deg,
            angle_index=angle_index,
            angle_lookup=angle_lookup,
            pair_hard_min=pair_hard_min,
            pair_inner=pair_inner,
            pair_peak=pair_peak,
            pair_sigma=pair_sigma,
            pair_outer=pair_outer,
            pair_mask=pair_mask,
            coordination_target=coordination_target,
            coordination_std=coordination_std,
            angle_target=angle_target,
            angle_pair_mass_target=angle_pair_mass_target,
            angle_mode_deg=angle_mode_deg,
            angle_enabled_mask=angle_enabled_mask,
            motif_center_species=np.asarray(motif_center_species, dtype=np.intp),
            motif_neighbor_species=tuple(motif_neighbor_species),
            motif_neighbor_vectors=tuple(motif_neighbor_vectors),
            max_pair_outer=max_pair_outer,
            max_pair_outer_by_center=max_pair_outer_by_center,
            summary=summary,
        )

    @classmethod
    def from_targets(
        cls,
        targets: "dict[str, CoordinationShellTarget]",
        *,
        cross_pair_peak: "dict[tuple[str, str], float] | None" = None,
        cross_pair_outer_scale: float = 1.15,
        label: str | None = None,
    ) -> "CoordinationShellTarget":
        """Stack multiple shell targets into one with a widened species axis.

        Used for blended materials where atoms share an atomic number but
        want different local coordination (e.g. graphite sp\u00b2 + diamond
        sp\u00b3 carbon).  Each input target contributes a *virtual species
        slot* per element of its ``species`` array; the composite target's
        ``species`` is the concatenation of all inputs, with
        ``species_labels`` rewritten as ``f"{key}_{element}"``.

        Cross-target pairs default to:

        - ``coordination_target = 0`` (no bonds form across virtual
          species boundaries; the repulsion term still keeps them apart),
        - ``pair_peak = mean(peak_a, peak_b)`` unless overridden by
          ``cross_pair_peak``,
        - ``pair_outer = max(outer_a, outer_b) * cross_pair_outer_scale``,
        - ``pair_hard_min`` / ``pair_inner`` pro-rated from the two
          source values.

        Cross-target triplets (any of the three species drawn from a
        different source than the other two) get
        ``coordination_target = 0`` and zero ``angle_mode_deg`` — the
        relaxer will never enumerate these triplets because no such
        bonds form.

        Parameters
        ----------
        targets
            Mapping ``{key: CoordinationShellTarget}``.  Insertion order
            defines the virtual-species order.
        cross_pair_peak
            Optional overrides for ``pair_peak`` between elements drawn
            from different source targets.  Keys are ``(key_a, key_b)``
            with symbol lookup done via ``atomic_numbers`` when pair
            contains non-tuple element labels.
        cross_pair_outer_scale
            Multiplier applied to the larger of the two source
            ``pair_outer`` values when populating cross-target entries.
        label
            Optional label; defaults to ``"composite(" + keys + ")"``.
        """
        from dataclasses import replace as _dc_replace  # noqa: F401

        if len(targets) == 0:
            raise ValueError("from_targets requires at least one target.")
        keys = list(targets.keys())
        first = targets[keys[0]]
        phi_num_bins = int(first.phi_num_bins)
        for key in keys[1:]:
            t = targets[key]
            if int(t.phi_num_bins) != phi_num_bins:
                raise ValueError(
                    f"All targets must share phi_num_bins; got "
                    f"{phi_num_bins} (from {keys[0]!r}) and "
                    f"{int(t.phi_num_bins)} (from {key!r})."
                )

        # Per-source species counts and global offsets.
        src_species_counts = [int(np.asarray(targets[k].species).size) for k in keys]
        offsets = np.zeros(len(keys) + 1, dtype=np.intp)
        offsets[1:] = np.cumsum(src_species_counts)
        num_species = int(offsets[-1])

        # Track which source (key index) each global species belongs to.
        species_source = np.zeros(num_species, dtype=np.intp)
        for ki, count in enumerate(src_species_counts):
            species_source[offsets[ki] : offsets[ki + 1]] = ki

        # --- concat species + labels ---
        species = np.zeros(num_species, dtype=np.int64)
        species_labels: list[str] = []
        for ki, key in enumerate(keys):
            t = targets[key]
            species[offsets[ki] : offsets[ki + 1]] = np.asarray(t.species, dtype=np.int64)
            for sym in t.species_labels:
                species_labels.append(f"{key}_{sym}")

        # --- block-diagonal pair-array scaffold ---
        pair_hard_min = np.zeros((num_species, num_species), dtype=np.float64)
        pair_inner = np.zeros_like(pair_hard_min)
        pair_peak = np.zeros_like(pair_hard_min)
        pair_sigma = np.zeros_like(pair_hard_min)
        pair_outer = np.zeros_like(pair_hard_min)
        pair_mask = np.zeros_like(pair_hard_min, dtype=bool)
        coordination_target = np.zeros_like(pair_hard_min)
        coordination_std = np.zeros_like(pair_hard_min)
        for ki, key in enumerate(keys):
            t = targets[key]
            a, b = int(offsets[ki]), int(offsets[ki + 1])
            pair_hard_min[a:b, a:b] = np.asarray(t.pair_hard_min, dtype=np.float64)
            pair_inner[a:b, a:b] = np.asarray(t.pair_inner, dtype=np.float64)
            pair_peak[a:b, a:b] = np.asarray(t.pair_peak, dtype=np.float64)
            pair_sigma[a:b, a:b] = np.asarray(t.pair_sigma, dtype=np.float64)
            pair_outer[a:b, a:b] = np.asarray(t.pair_outer, dtype=np.float64)
            pair_mask[a:b, a:b] = np.asarray(t.pair_mask, dtype=bool)
            coordination_target[a:b, a:b] = np.asarray(t.coordination_target, dtype=np.float64)
            coordination_std[a:b, a:b] = np.asarray(t.coordination_std, dtype=np.float64)

        # --- cross-target pair entries (repulsion-only by default) ---
        cross_peak_lookup: dict[tuple[str, str], float] = {}
        if cross_pair_peak is not None:
            for (ka, kb), v in cross_pair_peak.items():
                cross_peak_lookup[(ka, kb)] = float(v)
                cross_peak_lookup[(kb, ka)] = float(v)

        for i in range(num_species):
            for j in range(num_species):
                if species_source[i] == species_source[j]:
                    continue
                # Use the source's own self-pair as a proxy for the
                # same-element repulsion wall on each side.
                ki, kj = int(species_source[i]), int(species_source[j])
                key_a, key_b = keys[ki], keys[kj]
                peak_a = float(pair_peak[i, i])
                peak_b = float(pair_peak[j, j])
                inner_a = float(pair_inner[i, i])
                inner_b = float(pair_inner[j, j])
                outer_a = float(pair_outer[i, i])
                outer_b = float(pair_outer[j, j])
                hmin_a = float(pair_hard_min[i, i])
                hmin_b = float(pair_hard_min[j, j])
                sig_a = float(pair_sigma[i, i])
                sig_b = float(pair_sigma[j, j])
                if (key_a, key_b) in cross_peak_lookup:
                    peak_ij = cross_peak_lookup[(key_a, key_b)]
                elif peak_a > 0 and peak_b > 0:
                    peak_ij = 0.5 * (peak_a + peak_b)
                else:
                    peak_ij = max(peak_a, peak_b)
                pair_peak[i, j] = peak_ij
                pair_inner[i, j] = 0.5 * (inner_a + inner_b)
                pair_outer[i, j] = max(outer_a, outer_b) * float(cross_pair_outer_scale)
                pair_hard_min[i, j] = max(hmin_a, hmin_b)
                pair_sigma[i, j] = max(sig_a, sig_b) if (sig_a + sig_b) > 0 else 0.05
                pair_mask[i, j] = True
                # coordination_target[i, j] stays 0 — no cross bonds.

        # --- rebuild triplet index over the widened species set ---
        angle_index_list: list[tuple[int, int, int]] = []
        angle_lookup_new = -np.ones((num_species, num_species, num_species), dtype=np.intp)
        for c in range(num_species):
            for n1 in range(num_species):
                for n2 in range(n1, num_species):
                    idx = len(angle_index_list)
                    angle_index_list.append((c, n1, n2))
                    angle_lookup_new[c, n1, n2] = idx
                    angle_lookup_new[c, n2, n1] = idx
        angle_index_new = np.asarray(angle_index_list, dtype=np.intp)

        angle_target = np.zeros((angle_index_new.shape[0], phi_num_bins), dtype=np.float64)
        angle_pair_mass_target = np.zeros(angle_index_new.shape[0], dtype=np.float64)
        angle_mode_deg = np.zeros(angle_index_new.shape[0], dtype=np.float64)
        # Default: True everywhere; cross-source triplets will never fire
        # anyway because their coordination_target is zero.
        angle_enabled_mask = np.ones(angle_index_new.shape[0], dtype=bool)

        # Copy triplets where all three species come from the same source.
        for ki, key in enumerate(keys):
            t = targets[key]
            a = int(offsets[ki])
            src_angle_index = np.asarray(t.angle_index, dtype=np.intp)
            src_angle_lookup = np.asarray(t.angle_lookup, dtype=np.intp)
            src_angle_target = np.asarray(t.angle_target, dtype=np.float64)
            src_angle_mass = np.asarray(t.angle_pair_mass_target, dtype=np.float64)
            src_angle_mode = np.asarray(t.angle_mode_deg, dtype=np.float64)
            src_angle_mask = np.asarray(
                getattr(t, "angle_enabled_mask",
                        np.ones(src_angle_index.shape[0], dtype=bool)),
                dtype=bool,
            )
            for local_t, (lc, ln1, ln2) in enumerate(src_angle_index):
                gc = int(a + lc)
                gn1 = int(a + ln1)
                gn2 = int(a + ln2)
                new_t = int(angle_lookup_new[gc, gn1, gn2])
                angle_target[new_t] = src_angle_target[local_t]
                angle_pair_mass_target[new_t] = float(src_angle_mass[local_t])
                angle_mode_deg[new_t] = float(src_angle_mode[local_t])
                angle_enabled_mask[new_t] = bool(src_angle_mask[local_t])
            del src_angle_lookup  # silence unused

        # --- concatenate motif arrays with species remapping ---
        motif_center_species_list: list[int] = []
        motif_neighbor_species_list: list[np.ndarray] = []
        motif_neighbor_vectors_list: list[np.ndarray] = []
        for ki, key in enumerate(keys):
            t = targets[key]
            a = int(offsets[ki])
            src_center = np.asarray(t.motif_center_species, dtype=np.intp)
            for local_i, cs in enumerate(src_center):
                motif_center_species_list.append(int(a + int(cs)))
                ns = np.asarray(t.motif_neighbor_species[local_i], dtype=np.intp) + a
                vs = np.asarray(t.motif_neighbor_vectors[local_i], dtype=np.float64)
                motif_neighbor_species_list.append(ns)
                motif_neighbor_vectors_list.append(vs)

        max_pair_outer = float(np.max(pair_outer[pair_mask])) if np.any(pair_mask) else 0.0
        max_pair_outer_by_center = np.zeros(num_species, dtype=np.float64)
        for center_ind in range(num_species):
            row = pair_outer[center_ind][pair_mask[center_ind]]
            max_pair_outer_by_center[center_ind] = float(np.max(row)) if row.size else 0.0

        phi_edges = np.asarray(first.phi_edges, dtype=np.float64)
        phi = np.asarray(first.phi, dtype=np.float64)
        phi_deg = np.asarray(first.phi_deg, dtype=np.float64)

        summary = {
            "composite": True,
            "keys": tuple(keys),
            "source_species_counts": tuple(int(c) for c in src_species_counts),
            "phi_num_bins": phi_num_bins,
            "num_species": num_species,
            "num_triplets": int(angle_index_new.shape[0]),
            "max_pair_outer": float(max_pair_outer),
        }

        composite_label = label or f"composite({'+'.join(keys)})"

        return cls(
            atoms=first.atoms,  # placeholder; not used by shell_relax
            label=composite_label,
            species=species,
            species_labels=tuple(species_labels),
            phi_num_bins=phi_num_bins,
            phi_edges=phi_edges,
            phi=phi,
            phi_deg=phi_deg,
            angle_index=angle_index_new,
            angle_lookup=angle_lookup_new,
            pair_hard_min=pair_hard_min,
            pair_inner=pair_inner,
            pair_peak=pair_peak,
            pair_sigma=pair_sigma,
            pair_outer=pair_outer,
            pair_mask=pair_mask,
            coordination_target=coordination_target,
            coordination_std=coordination_std,
            angle_target=angle_target,
            angle_pair_mass_target=angle_pair_mass_target,
            angle_mode_deg=angle_mode_deg,
            angle_enabled_mask=angle_enabled_mask,
            motif_center_species=np.asarray(motif_center_species_list, dtype=np.intp),
            motif_neighbor_species=tuple(motif_neighbor_species_list),
            motif_neighbor_vectors=tuple(motif_neighbor_vectors_list),
            max_pair_outer=max_pair_outer,
            max_pair_outer_by_center=max_pair_outer_by_center,
            summary=summary,
        )

    def with_bonded_species_pairs(
        self,
        pairs: "list[tuple[str, str]]",
    ) -> "CoordinationShellTarget":
        """Return a copy whose ``coordination_target`` is zero everywhere
        *except* the given symmetric species pairs.

        Useful for materials with spectator ions: perovskites like
        SrTiO\u2083 want only Ti-O bonds considered by
        :meth:`Supercell.shell_relax`, since Sr-O, Sr-Ti, O-O, Ti-Ti, etc.
        would either install spurious angle springs (``angle_mode_deg``
        is a geometric artefact for non-bond triplets) or pin atoms via
        bond springs to distances that are really second-shell
        separations, not chemical bonds.

        Examples
        --------

        .. code-block:: python

            # SrTiO3: preserve only TiO6 octahedra
            st.with_bonded_species_pairs([('Ti', 'O')])

            # SiO2: equivalent to ``with_cross_species_bonds_only`` for a
            # binary, but explicit about what a bond is
            st.with_bonded_species_pairs([('Si', 'O')])
        """
        from dataclasses import replace as _dc_replace
        from ase.data import atomic_numbers as _an

        ct = np.zeros_like(np.asarray(self.coordination_target, dtype=np.float64))
        sp = np.asarray(self.species, dtype=np.int64)
        orig = np.asarray(self.coordination_target, dtype=np.float64)
        for sa, sb in pairs:
            za = int(_an[sa])
            zb = int(_an[sb])
            ia_arr = np.where(sp == za)[0]
            ib_arr = np.where(sp == zb)[0]
            if ia_arr.size == 0 or ib_arr.size == 0:
                continue
            ia, ib = int(ia_arr[0]), int(ib_arr[0])
            ct[ia, ib] = orig[ia, ib]
            ct[ib, ia] = orig[ib, ia]
        return _dc_replace(self, coordination_target=ct)

    def with_cross_species_bonds_only(self) -> "CoordinationShellTarget":
        """Return a copy where same-species ``coordination_target`` entries
        are zeroed.

        Useful for network-former compounds such as SiO\u2082 where only
        cross-species pairs (Si-O) are real chemical bonds; the same-species
        "shell" peaks (Si-Si, O-O) come from the second coordination shell
        through the bridging atom and should not be treated as bonds by
        :meth:`Supercell.shell_relax` (which would otherwise install
        spurious angle springs on triplets like Si-Si-Si or O-O-O whose
        ``angle_mode_deg`` is just a geometric artefact of the reference
        sampling, not a physical target).
        """
        from dataclasses import replace as _dc_replace

        ct = np.asarray(self.coordination_target, dtype=np.float64).copy()
        for i in range(ct.shape[0]):
            ct[i, i] = 0.0
        return _dc_replace(self, coordination_target=ct)

    def with_angle_triplets(
        self,
        triplets: "list[tuple[str, str, str]]",
    ) -> "CoordinationShellTarget":
        """Return a copy whose angle-spring mask is enabled *only* for
        the listed triplet types; all other angle springs are disabled.

        Each triplet is ``(centre_symbol, neighbour_1_symbol,
        neighbour_2_symbol)``; both (n1, n2) and (n2, n1) are enabled
        automatically.  Bond-distance springs are untouched — only the
        angle springs installed during ``shell_relax`` are filtered.

        Useful for multi-modal shells where the extracted
        ``angle_mode_deg`` picks one peak of a bimodal / quadrimodal
        distribution; enforcing it would strain the other modes.
        SrTiO\u2083's SrO\u2081\u2082 cuboctahedron (O-Sr-O angles at
        60°/90°/120°/180°) is the canonical example.

        .. code-block:: python

            # Keep Ti-centered 90° and linear O-Ti-Ti 180° angle
            # springs; silence every Sr-centered or Sr-in-triplet
            # angle spring.
            st.with_angle_triplets([
                ('Ti', 'O', 'O'),
                ('O',  'Ti', 'Ti'),
            ])
        """
        from dataclasses import replace as _dc_replace
        from ase.data import atomic_numbers as _an

        sp = np.asarray(self.species, dtype=np.int64)
        ai = np.asarray(self.angle_index, dtype=np.intp)
        mask = np.zeros(ai.shape[0], dtype=bool)

        def _species_slots(sym: str) -> np.ndarray:
            return np.where(sp == int(_an[sym]))[0]

        for centre_sym, n1_sym, n2_sym in triplets:
            c_idx = _species_slots(centre_sym)
            n1_idx = _species_slots(n1_sym)
            n2_idx = _species_slots(n2_sym)
            if c_idx.size == 0 or n1_idx.size == 0 or n2_idx.size == 0:
                continue
            for ci in c_idx:
                for a_i in n1_idx:
                    for b_i in n2_idx:
                        # Canonical order: neigh_1 <= neigh_2.
                        lo, hi = (int(a_i), int(b_i)) if a_i <= b_i else (int(b_i), int(a_i))
                        t = int(self.angle_lookup[int(ci), lo, hi])
                        if t >= 0:
                            mask[t] = True
        return _dc_replace(self, angle_enabled_mask=mask)

    def without_angle_triplets(
        self,
        triplets: "list[tuple[str, str, str]]",
    ) -> "CoordinationShellTarget":
        """Return a copy with the angle mask disabled for the listed
        triplets (inverse of :meth:`with_angle_triplets`).
        """
        from dataclasses import replace as _dc_replace
        from ase.data import atomic_numbers as _an

        sp = np.asarray(self.species, dtype=np.int64)
        mask = np.asarray(self.angle_enabled_mask, dtype=bool).copy()

        def _species_slots(sym: str) -> np.ndarray:
            return np.where(sp == int(_an[sym]))[0]

        for centre_sym, n1_sym, n2_sym in triplets:
            c_idx = _species_slots(centre_sym)
            n1_idx = _species_slots(n1_sym)
            n2_idx = _species_slots(n2_sym)
            for ci in c_idx:
                for a_i in n1_idx:
                    for b_i in n2_idx:
                        lo, hi = (int(a_i), int(b_i)) if a_i <= b_i else (int(b_i), int(a_i))
                        t = int(self.angle_lookup[int(ci), lo, hi])
                        if t >= 0:
                            mask[t] = False
        return _dc_replace(self, angle_enabled_mask=mask)

    @property
    def pair_labels(self) -> list[str]:
        """Return human-readable pair labels."""
        labels = []
        for center_ind, center_label in enumerate(self.species_labels):
            for neigh_ind, neigh_label in enumerate(self.species_labels):
                if self.pair_mask[center_ind, neigh_ind]:
                    labels.append(f"{center_label}-{neigh_label}")
        return labels

    @property
    def angle_labels(self) -> list[str]:
        """Return human-readable rooted angle labels."""
        labels = []
        for center_ind, neigh1_ind, neigh2_ind in self.angle_index:
            labels.append(
                f"{self.species_labels[neigh1_ind]}-{self.species_labels[center_ind]}-{self.species_labels[neigh2_ind]}"
            )
        return labels
