"""Shell-target → flat-array conversion for relaxml conditioning.

The trained surrogate needs to know which target distances and angles
``shell_relax`` is pulling toward — that's what distinguishes one phase
from another at the same composition (e.g. α-quartz vs β-cristobalite
SiO₂).  This module turns a :class:`tricor.shells.CoordinationShellTarget`
into a small set of flat NumPy arrays suitable for storage in .npz files
and consumption by a deep-set encoder in the model.

Schema produced by :func:`extract_shell_target_arrays`:

    shell_pair_species    (P, 2) int32    atomic numbers (Z_a, Z_b)
    shell_pair_features   (P, 4) float32  [target_r, sigma, n_ab, n_ba]
    shell_triplet_species (T, 3) int32    atomic numbers (Z_a, Z_b, Z_c)
    shell_triplet_features(T, 2) float32  [angle_mode_rad, mass_weight]

NOTE on feature scaling: pair features mix Å distances (~1-3),
unitless sigmas (~0.05-0.3) and raw coordination numbers (~2-12) on
heterogeneous scales.  Triplet features mix radians (~0-π) and a
[0, ~10] mass weight.  We feed these raw and rely on the encoder's
LayerNorm to absorb the scale mismatch.  If training shows the larger-
magnitude features (coordination numbers, mass weights) dominating the
encoded representation, normalize per feature here — divide n_ab / n_ba
by 12 (max sensible coord) and mass_weight by some reasonable scale.

Only entries that ``shell_relax`` actually installs springs for are
emitted:

  * Pairs where ``coordination_target > 0`` in at least one direction —
    this filters out geometric-only pairs (e.g. Si-Si in SiO₂ at
    second-shell distance through a bridging O) that ``pair_mask``
    would still mark True but that aren't chemical bonds.
  * Triplets where ``angle_enabled_mask=True`` *and* both legs
    correspond to real bonds (so the angle is actually installed by
    ``shell_relax``).
"""

from __future__ import annotations

import numpy as np

from tricor.shells import CoordinationShellTarget


def extract_shell_target_arrays(
    shell_target: CoordinationShellTarget,
) -> dict[str, np.ndarray]:
    """Flatten a CoordinationShellTarget into the .npz schema above.

    Returns a dict with exactly the four keys listed in the module
    docstring.  P and T are post-filter counts (mask=True only).
    """
    species = np.asarray(shell_target.species, dtype=np.int32)         # (S,)
    pair_peak = np.asarray(shell_target.pair_peak, dtype=np.float32)
    pair_sigma = np.asarray(shell_target.pair_sigma, dtype=np.float32)
    coord = np.asarray(shell_target.coordination_target, dtype=np.float32)

    # "Real bond" filter: at least one direction has a non-zero target
    # neighbor count.  shell_relax only installs a bond spring when this
    # condition holds — geometric-only pairs (pair_mask=True but
    # coordination_target=0, e.g. Si-Si in SiO₂ at second shell) are
    # excluded.
    real_bond = (coord > 0.0) | (coord.T > 0.0)                        # (S, S)

    # Pairs: keep only i <= j to avoid duplicating (a,b) and (b,a) — they
    # share the same target r/sigma anyway.  We emit n_ab and n_ba
    # separately so the encoder sees both directional coordination
    # numbers (relevant when the structure isn't 1:1 stoichiometric).
    pair_species_list: list[tuple[int, int]] = []
    pair_features_list: list[tuple[float, float, float, float]] = []
    S = species.shape[0]
    for i in range(S):
        for j in range(i, S):
            if not real_bond[i, j]:
                continue
            pair_species_list.append((int(species[i]), int(species[j])))
            pair_features_list.append((
                float(pair_peak[i, j]),
                float(pair_sigma[i, j]),
                float(coord[i, j]),
                float(coord[j, i]),
            ))

    if pair_species_list:
        shell_pair_species = np.asarray(pair_species_list, dtype=np.int32)
        shell_pair_features = np.asarray(pair_features_list, dtype=np.float32)
    else:
        shell_pair_species = np.zeros((0, 2), dtype=np.int32)
        shell_pair_features = np.zeros((0, 4), dtype=np.float32)

    # Triplets: keep only those that are (a) enabled AND (b) involve real
    # bonds on both legs (center→n1 and center→n2).  An angle spring on a
    # non-bond leg would refer to a geometric artefact, not a chemical
    # bond, so it'd be misleading conditioning for the model.
    angle_index = np.asarray(shell_target.angle_index, dtype=np.intp)
    angle_enabled = np.asarray(shell_target.angle_enabled_mask, dtype=bool)
    angle_mode_deg = np.asarray(shell_target.angle_mode_deg, dtype=np.float32)
    angle_mass = np.asarray(shell_target.angle_pair_mass_target, dtype=np.float32)

    centers = angle_index[:, 0]
    leg1 = angle_index[:, 1]
    leg2 = angle_index[:, 2]
    real_legs = real_bond[centers, leg1] & real_bond[centers, leg2]
    keep_t = np.flatnonzero(angle_enabled & real_legs)
    if keep_t.size:
        kept_idx = angle_index[keep_t]                                 # (T, 3)
        shell_triplet_species = np.empty((keep_t.size, 3), dtype=np.int32)
        for col in range(3):
            shell_triplet_species[:, col] = species[kept_idx[:, col]]
        shell_triplet_features = np.stack([
            np.deg2rad(angle_mode_deg[keep_t]),
            angle_mass[keep_t],
        ], axis=1).astype(np.float32)
    else:
        shell_triplet_species = np.zeros((0, 3), dtype=np.int32)
        shell_triplet_features = np.zeros((0, 2), dtype=np.float32)

    return {
        "shell_pair_species":     shell_pair_species,
        "shell_pair_features":    shell_pair_features,
        "shell_triplet_species":  shell_triplet_species,
        "shell_triplet_features": shell_triplet_features,
    }


# Number of feature columns per pair / triplet, so callers (model + data
# loader) can sanity-check shapes without re-computing.
NUM_PAIR_FEATURES: int = 4
NUM_TRIPLET_FEATURES: int = 2
