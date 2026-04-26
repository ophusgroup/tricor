from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.neighborlist import neighbor_list

from .g3 import _EPS, _TextProgressBar

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell


class _ShellRelaxMixin:
    # ------------------------------------------------------------------
    # shell_relax: vectorized spring-network relaxation
    # ------------------------------------------------------------------

    def shell_relax(
        self: "Supercell",
        shell_target: CoordinationShellTarget,
        num_steps: int = 200,
        *,
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        k_restraint: float = 0.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        step_size: float = 0.1,
        step_decay: float = 0.995,
        neighbor_update_interval: int = 10,
        neighbor_cutoff_scale: float = 1.5,
        max_force_clip: float = 2.0,
        capture_trajectory: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Relax random positions to match first-shell targets using spring forces.

        Moves **all atoms simultaneously** each step via three vectorized force
        terms: bond springs toward the target nearest-neighbor distance, angle
        springs toward the target bond angle, and soft repulsion to eliminate
        overlaps and close-packed background.  Bond topology (K-nearest
        assignment) is rebuilt periodically using ASE's ``neighbor_list``.

        Parameters
        ----------
        shell_target
            First-shell coordination targets extracted from the reference
            crystal via :meth:`CoordinationShellTarget.from_atoms`.
        num_steps
            Number of relaxation sweeps.
        bond_weight
            Strength of the harmonic spring pulling bonded neighbors toward
            ``pair_peak`` distance.
        angle_weight
            Strength of the angular spring pushing bond angles toward
            ``angle_mode_deg``.
        repulsion_weight
            Strength of the short-range repulsive force below ``pair_hard_min``.
        k_restraint
            Spring constant (eV / Å²) for a global position-restraint
            energy ``½ k_restraint Σ ‖r_i - r_initial_i‖²`` that tethers
            every atom to its starting position.  ``0.0`` (default)
            disables the term and reproduces unrestrained relaxation.
            Small values (~0.1 - 1.0) preserve regime character (grain
            layout, amorphous topology) while still permitting local
            relaxation; large values (≫ 10) hold the structure rigid.
        hard_core_scale
            Multiplier for the hard-core repulsion radius.  1.0 uses
            ``max(pair_hard_min, pair_inner)`` as the wall.  Values
            below 1.0 allow shorter bonds (softer wall for liquid).
            Values above 1.0 enforce a larger exclusion zone.
        nonbond_push_scale
            Multiplier for the non-bonded shell clearance distance.
            1.0 pushes non-bonded atoms to ``1.5 * pair_peak``.
            Values below 1.0 allow non-bonded atoms closer (broader
            2nd shell for liquid).  0.0 disables non-bonded push.
        step_size
            Initial maximum displacement per step (Angstrom).
        step_decay
            Multiplicative decay applied to *step_size* each iteration.
        neighbor_update_interval
            Rebuild the bond topology every this many steps.
        neighbor_cutoff_scale
            Neighbor search cutoff as a multiple of ``max_pair_outer``.
        max_force_clip
            Per-atom force magnitude is clipped to this value before
            integration to keep the dynamics stable.
        show_progress
            Display a text progress bar.

        Returns
        -------
        dict[str, Any]
            Summary with parameters and final/initial loss values.
        """
        num_atoms = len(self.atoms)
        # Prefer the composite-target virtual species mapping if the
        # caller set one (e.g. sp²/sp³ carbon blends where atomic
        # number alone can't distinguish the two chemistries);
        # otherwise fall back to the atomic-number mapping.
        species_idx = (
            self._atom_shell_species_index
            if getattr(self, "_atom_shell_species_index", None) is not None
            else self._atom_species_index
        )  # (num_atoms,) int
        cell_inv = self._cell_inverse
        cell_mat = self._cell_matrix

        # --- extract targets from shell_target ---
        coord_target = np.asarray(shell_target.coordination_target, dtype=np.float64)
        pair_peak = np.asarray(shell_target.pair_peak, dtype=np.float64)
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        angle_mode_rad = np.deg2rad(
            np.asarray(shell_target.angle_mode_deg, dtype=np.float64)
        )
        angle_lookup = np.asarray(shell_target.angle_lookup, dtype=np.intp)
        # Per-triplet angle-spring mask.  Defaults to all-True for
        # shell targets produced by older builds that predate the
        # field, so back-compat is preserved.
        _default_mask = np.ones(angle_mode_rad.size, dtype=bool)
        angle_enabled_mask = np.asarray(
            getattr(shell_target, "angle_enabled_mask", _default_mask),
            dtype=bool,
        )
        cutoff = float(shell_target.max_pair_outer * neighbor_cutoff_scale)

        # K nearest neighbors per atom, both total and per-species-pair
        k_per_species = np.zeros(shell_target.species.size, dtype=np.intp)
        for s in range(shell_target.species.size):
            k_per_species[s] = int(np.round(coord_target[s].sum()))

        # Per-species-pair coordination targets (rounded to int)
        num_sp = shell_target.species.size
        coord_target_int = np.round(coord_target).astype(np.intp)

        # Repulsion radii: hard core (overlap prevention) and non-bonded
        # shell clearance (eliminates close-packed background).
        pair_outer = np.asarray(shell_target.pair_outer, dtype=np.float64)
        # Hard core: use max of pair_hard_min and pair_inner to
        # prevent any bonds shorter than the shell inner boundary.
        pair_inner = np.asarray(shell_target.pair_inner, dtype=np.float64)
        hard_core = np.maximum(pair_hard_min, pair_inner) * float(hard_core_scale)
        mask_zero = hard_core < _EPS
        hard_core[mask_zero] = 0.4 * pair_peak[mask_zero]
        global_floor = float(np.min(pair_peak[pair_peak > _EPS])) * 0.4 if np.any(pair_peak > _EPS) else 1.0
        hard_core[hard_core < _EPS] = global_floor
        # Non-bonded atoms are pushed beyond this radius to create a
        # clean gap between the first and second coordination shells.
        # For Si, 2nd shell is at ~3.84Å (sqrt(8/3) * pair_peak).
        # Push non-bonded atoms to at least 1.5x pair_peak to
        # eliminate close-packed triplets from nearby non-bonded pairs.
        nonbond_push = pair_peak * 1.5 * float(nonbond_push_scale)
        nonbond_push[nonbond_push < _EPS] = float(np.max(pair_peak)) * 1.5

        # --- grain-aware force scaling ---
        # When _grain_ids is set, interior atoms are frozen to preserve
        # crystalline order; boundary atoms get full relaxation forces.
        grain_ids = self._grain_ids
        grain_seeds = self._grain_seeds
        if (
            grain_ids is not None
            and grain_seeds is not None
            and len(grain_ids) == num_atoms
        ):
            is_boundary = np.ones(num_atoms, dtype=bool)  # start all boundary
            _grain_boundary_detected = [False]

            def _detect_boundary_atoms() -> None:
                """Mark atoms as boundary using distance to grain boundary.

                For each atom, boundary_depth = half the gap between the
                distance to the nearest foreign seed and the distance to
                its own seed.  Atoms deep inside a grain (boundary_depth
                > threshold) are interior; the rest are boundary.
                """
                if _grain_boundary_detected[0]:
                    return
                _grain_boundary_detected[0] = True

                pos = self.atoms.positions
                n_seeds_local = len(grain_seeds)
                boundary_width = float(np.max(pair_peak)) * 0.5

                is_boundary[:] = True  # default boundary

                # Process in chunks to bound memory
                _bchunk = max(1, 25_000_000 // max(n_seeds_local, 1))
                for start in range(0, num_atoms, _bchunk):
                    end = min(start + _bchunk, num_atoms)
                    delta = pos[start:end, None, :] - grain_seeds[None, :, :]
                    frac_d = delta @ cell_inv
                    frac_d -= np.rint(frac_d)
                    cart_d = frac_d @ cell_mat
                    dist_chunk = np.sqrt(np.sum(cart_d ** 2, axis=2))

                    for ia_local in range(end - start):
                        ia = start + ia_local
                        gid = grain_ids[ia]
                        if gid < 0:
                            continue
                        dist_own = dist_chunk[ia_local, gid]
                        dists_row = dist_chunk[ia_local].copy()
                        dists_row[gid] = np.inf
                        dist_other = float(np.min(dists_row))
                        if (dist_other - dist_own) * 0.5 > boundary_width:
                            is_boundary[ia] = False

            # Interior atoms: forces zeroed completely (frozen)
            interior_force_scale = 0.0
        else:
            grain_ids = None
            is_boundary = None
            _grain_boundary_detected = None

            def _detect_boundary_atoms() -> None:
                pass

            interior_force_scale = 1.0

        # --- vectorized minimum-image helper for paired arrays ---
        def min_image(delta: np.ndarray) -> np.ndarray:
            frac = delta @ cell_inv
            frac -= np.rint(frac)
            return frac @ cell_mat

        # --- neighbor rebuild ---
        bond_i = np.empty(0, dtype=np.intp)
        bond_j = np.empty(0, dtype=np.intp)
        bond_r_target = np.empty(0, dtype=np.float64)
        tri_center = np.empty(0, dtype=np.intp)
        tri_a = np.empty(0, dtype=np.intp)
        tri_b = np.empty(0, dtype=np.intp)
        tri_phi_target = np.empty(0, dtype=np.float64)
        bonded_set: set[tuple[int, int]] = set()

        def rebuild_topology() -> None:
            nonlocal bond_i, bond_j, bond_r_target
            nonlocal tri_center, tri_a, tri_b, tri_phi_target
            nonlocal bonded_set

            nl_i, nl_j, nl_d = neighbor_list("ijd", self.atoms, cutoff)

            # Symmetric bond matching with angular + species awareness:
            # greedily build a bond graph respecting per-species-pair
            # coordination targets.  Candidates sorted by distance;
            # each accepted only if angle and species constraints pass.
            bond_count = np.zeros(num_atoms, dtype=np.intp)
            # Per-atom, per-neighbor-species bond counts
            bond_count_pair = np.zeros((num_atoms, num_sp), dtype=np.intp)
            k_atom = np.array(
                [int(k_per_species[species_idx[a]]) for a in range(num_atoms)],
                dtype=np.intp,
            )

            # Pre-compute displacement vectors for all neighbor pairs
            nl_vecs = min_image(
                self.atoms.positions[nl_j] - self.atoms.positions[nl_i]
            )
            nl_hats = nl_vecs / np.maximum(nl_d, _EPS)[:, None]

            # Sort candidates by distance (nearest first)
            dist_order = np.argsort(nl_d)

            _bond_i_list: list[int] = []
            _bond_j_list: list[int] = []
            _bond_rt_list: list[float] = []
            bonded_set = set()
            bonded_neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
            # Store unit vectors of existing bonds per atom for angle check
            bond_hats_per_atom: list[list[np.ndarray]] = [[] for _ in range(num_atoms)]

            min_accept_angle = np.deg2rad(60.0)  # reject bonds with < 60deg to existing

            def _species_pair_ok(ai: int, aj: int) -> bool:
                """Check per-species-pair coordination limits."""
                si, sj = species_idx[ai], species_idx[aj]
                if bond_count_pair[ai, sj] >= coord_target_int[si, sj]:
                    return False
                if bond_count_pair[aj, si] >= coord_target_int[sj, si]:
                    return False
                return True

            def _accept_bond(ai: int, aj: int) -> None:
                """Record a new bond between atoms ai and aj."""
                si, sj = species_idx[ai], species_idx[aj]
                _bond_i_list.append(ai)
                _bond_j_list.append(aj)
                _bond_rt_list.append(float(pair_peak[si, sj]))
                bonded_set.add((ai, aj))
                bonded_set.add((aj, ai))
                bonded_neighbors[ai].append(aj)
                bonded_neighbors[aj].append(ai)
                bond_count[ai] += 1
                bond_count[aj] += 1
                bond_count_pair[ai, sj] += 1
                bond_count_pair[aj, si] += 1

            for idx in dist_order:
                ai = int(nl_i[idx])
                aj = int(nl_j[idx])
                if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
                    continue
                if (ai, aj) in bonded_set:
                    continue
                if not _species_pair_ok(ai, aj):
                    continue

                hat_ij = nl_hats[idx]
                hat_ji = -hat_ij

                # Check angular compatibility with existing bonds at ai
                accept = True
                for existing_hat in bond_hats_per_atom[ai]:
                    cos_a = np.dot(hat_ij, existing_hat)
                    if cos_a > np.cos(min_accept_angle):
                        accept = False
                        break
                if not accept:
                    continue

                # Check angular compatibility at aj
                for existing_hat in bond_hats_per_atom[aj]:
                    cos_a = np.dot(hat_ji, existing_hat)
                    if cos_a > np.cos(min_accept_angle):
                        accept = False
                        break
                if not accept:
                    continue

                bond_hats_per_atom[ai].append(hat_ij.copy())
                bond_hats_per_atom[aj].append(hat_ji.copy())
                _accept_bond(ai, aj)

            # Second pass: fill remaining unsatisfied atoms with
            # distance-only matching (relaxing angle constraint but
            # still respecting species-pair limits)
            for idx in dist_order:
                ai = int(nl_i[idx])
                aj = int(nl_j[idx])
                if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
                    continue
                if (ai, aj) in bonded_set:
                    continue
                if not _species_pair_ok(ai, aj):
                    continue
                _accept_bond(ai, aj)

            bond_i = np.array(_bond_i_list, dtype=np.intp)
            bond_j = np.array(_bond_j_list, dtype=np.intp)
            bond_r_target = np.array(_bond_rt_list, dtype=np.float64)

            # Build triplet arrays from bonded neighbors
            _tc: list[int] = []
            _ta: list[int] = []
            _tb: list[int] = []
            _tp: list[float] = []
            for atom in range(num_atoms):
                bn = bonded_neighbors[atom]
                if len(bn) < 2:
                    continue
                s_center = species_idx[atom]
                for ia in range(len(bn)):
                    for ib in range(ia + 1, len(bn)):
                        s_a = species_idx[bn[ia]]
                        s_b = species_idx[bn[ib]]
                        # Ensure canonical order for angle lookup
                        if s_a <= s_b:
                            triplet_idx = int(angle_lookup[s_center, s_a, s_b])
                        else:
                            triplet_idx = int(angle_lookup[s_center, s_b, s_a])
                        # Skip triplets whose angle spring is masked
                        # off (multi-modal shells; see
                        # ``CoordinationShellTarget.with_angle_triplets``).
                        if not angle_enabled_mask[triplet_idx]:
                            continue
                        phi_t = float(angle_mode_rad[triplet_idx])
                        _tc.append(atom)
                        _ta.append(int(bn[ia]))
                        _tb.append(int(bn[ib]))
                        _tp.append(phi_t)

            tri_center = np.array(_tc, dtype=np.intp)
            tri_a = np.array(_ta, dtype=np.intp)
            tri_b = np.array(_tb, dtype=np.intp)
            tri_phi_target = np.array(_tp, dtype=np.float64)

        # --- history arrays ---
        loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        best_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        bond_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        angle_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        repulsion_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        restraint_loss_history = np.zeros(num_steps + 1, dtype=np.float64)

        if capture_trajectory:
            trajectory = np.zeros(
                (num_steps + 1, num_atoms, 3), dtype=np.float32,
            )
            atom_cost_history = np.zeros(
                (num_steps + 1, num_atoms), dtype=np.float32,
            )
        else:
            trajectory = None
            atom_cost_history = None

        current_step = float(step_size)
        velocity = np.zeros((num_atoms, 3), dtype=np.float64)
        momentum = 0.8  # momentum damping factor
        best_positions = self.atoms.positions.copy()
        best_loss = np.inf

        # Position-restraint reference: a frozen snapshot of the
        # starting positions.  Force contribution is
        # ``F_i = -k_restraint · (r_i - r_initial_i)`` (min-image-
        # corrected); ``k_restraint == 0`` disables the term.  This
        # lets the FIRE relaxation be globally differentiable yet still
        # tethered to the input regime structure.
        r_initial = self.atoms.positions.copy()
        k_restraint_f = float(k_restraint)

        if show_progress:
            progress = _TextProgressBar(num_steps, label="Shell relax", width=28)
        else:
            progress = None

        # --- main loop ---
        for step in range(num_steps + 1):
            pos = self.atoms.positions  # (num_atoms, 3) -- live reference

            # Rebuild bond topology periodically
            if step % neighbor_update_interval == 0:
                rebuild_topology()
                if step == 0:
                    _detect_boundary_atoms()

            # ---------- compute forces ----------
            force = np.zeros((num_atoms, 3), dtype=np.float64)
            if atom_cost_history is not None:
                atom_cost = np.zeros(num_atoms, dtype=np.float64)
            else:
                atom_cost = None

            # 1) Bond springs
            bond_loss = 0.0
            if bond_i.size > 0:
                bond_vec = min_image(pos[bond_j] - pos[bond_i])
                bond_r = np.linalg.norm(bond_vec, axis=1)
                bond_r_safe = np.maximum(bond_r, _EPS)
                bond_hat = bond_vec / bond_r_safe[:, None]
                delta_r = bond_r - bond_r_target
                bond_loss = float(np.mean(delta_r ** 2))
                f_bond = (bond_weight * delta_r)[:, None] * bond_hat
                np.add.at(force, bond_i, f_bond)
                np.add.at(force, bond_j, -f_bond)
                if atom_cost is not None:
                    # Spring-energy contribution: 0.5 * k * delta_r^2
                    # with k = bond_weight.  Before this the cost
                    # stored just delta_r^2/2 (unscaled), so weak-
                    # relax liquids with large residual delta_r
                    # reported spuriously enormous per-atom costs in
                    # the trajectory viewer (e.g. Cu liquid with
                    # bond_weight=0.05 showed cost_max=100 vs Si's
                    # bond_weight=0.4 showing cost_max=4).
                    half_bond_cost = 0.5 * float(bond_weight) * delta_r ** 2
                    np.add.at(atom_cost, bond_i, half_bond_cost)
                    np.add.at(atom_cost, bond_j, half_bond_cost)

            # 2) Angle springs
            angle_loss = 0.0
            if tri_center.size > 0:
                vec_a = min_image(pos[tri_a] - pos[tri_center])
                vec_b = min_image(pos[tri_b] - pos[tri_center])
                r_a = np.linalg.norm(vec_a, axis=1)
                r_b = np.linalg.norm(vec_b, axis=1)
                r_a_safe = np.maximum(r_a, _EPS)
                r_b_safe = np.maximum(r_b, _EPS)
                hat_a = vec_a / r_a_safe[:, None]
                hat_b = vec_b / r_b_safe[:, None]

                cos_phi = np.sum(hat_a * hat_b, axis=1)
                cos_phi = np.clip(cos_phi, -1.0 + 1e-7, 1.0 - 1e-7)
                phi = np.arccos(cos_phi)
                sin_phi = np.sqrt(1.0 - cos_phi ** 2)
                sin_phi_safe = np.maximum(sin_phi, 1e-7)

                delta_phi = phi - tri_phi_target
                angle_loss = float(np.mean(delta_phi ** 2))
                if atom_cost is not None:
                    # 0.5 * angle_weight * delta_phi^2 split 1/3 to
                    # each of the three triplet atoms.  Same scaling
                    # rationale as the bond cost above.
                    third_angle_cost = (
                        0.5 * float(angle_weight) * delta_phi ** 2 / 3.0
                    )
                    np.add.at(atom_cost, tri_center, third_angle_cost)
                    np.add.at(atom_cost, tri_a, third_angle_cost)
                    np.add.at(atom_cost, tri_b, third_angle_cost)

                perp_a = (hat_b - cos_phi[:, None] * hat_a) / sin_phi_safe[:, None]
                perp_b = (hat_a - cos_phi[:, None] * hat_b) / sin_phi_safe[:, None]

                f_angle_a = (angle_weight * delta_phi / r_a_safe)[:, None] * perp_a
                f_angle_b = (angle_weight * delta_phi / r_b_safe)[:, None] * perp_b

                np.add.at(force, tri_a, f_angle_a)
                np.add.at(force, tri_b, f_angle_b)
                np.add.at(force, tri_center, -(f_angle_a + f_angle_b))

            # 3) Repulsion: hard core + non-bonded shell clearance
            repulsion_loss = 0.0
            rep_cutoff = float(np.max(nonbond_push)) * 1.2
            rep_i_all, rep_j_all, rep_d_all, rep_D_all = neighbor_list(
                "ijdD", self.atoms, rep_cutoff,
            )
            if rep_i_all.size > 0:
                s_i = species_idx[rep_i_all]
                s_j = species_idx[rep_j_all]
                r_safe = np.maximum(rep_d_all, _EPS)
                rep_hat = rep_D_all / r_safe[:, None]

                # a) Hard core overlap prevention
                r_hard = hard_core[s_i, s_j]
                hard_ratio = r_hard / r_safe
                hard_mask = hard_ratio > 1.0
                hard_mag = np.zeros_like(r_safe)
                # Strong hard core: linear + quadratic for stiff wall
                hr = hard_ratio[hard_mask] - 1.0
                hard_mag[hard_mask] = repulsion_weight * 4.0 * (hr + hr ** 2)

                # b) Non-bonded clearance
                _pair_keys = rep_i_all.astype(np.int64) * num_atoms + rep_j_all.astype(np.int64)
                _bonded_keys = set(
                    int(a) * num_atoms + int(b) for a, b in bonded_set
                )
                is_bonded = np.array(
                    [int(k) in _bonded_keys for k in _pair_keys], dtype=bool,
                )
                r_push = nonbond_push[s_i, s_j]
                push_ratio = r_push / r_safe
                nonbond_mask = (~is_bonded) & (push_ratio > 1.0)
                nonbond_mag = np.zeros_like(r_safe)
                # Linear + quadratic: strong near boundary, stronger close in
                pr = push_ratio[nonbond_mask] - 1.0
                nonbond_mag[nonbond_mask] = repulsion_weight * (pr + pr ** 2)

                total_rep_mag = hard_mag + nonbond_mag
                active = total_rep_mag > 0.0
                repulsion_loss = float(np.sum(hard_mask)) + 0.1 * float(np.sum(nonbond_mask))

                if np.any(active):
                    f_rep = total_rep_mag[:, None] * rep_hat
                    np.add.at(force, rep_i_all, -f_rep)
                    np.add.at(force, rep_j_all, f_rep)

                if atom_cost is not None and np.any(active):
                    # Cost = hard-mask indicator + 0.1 * nonbond indicator, split 0.5 / 0.5
                    per_pair_cost = 0.5 * (
                        hard_mask.astype(np.float64)
                        + 0.1 * nonbond_mask.astype(np.float64)
                    )
                    np.add.at(atom_cost, rep_i_all, per_pair_cost)
                    np.add.at(atom_cost, rep_j_all, per_pair_cost)

            # 4) Position restraint (global tether to starting config)
            restraint_loss = 0.0
            if k_restraint_f > 0.0:
                d_restr = pos - r_initial
                # Min-image so PBC wrapping doesn't cause a spurious
                # tug across the cell boundary.
                d_restr = min_image(d_restr)
                # Force = -dE/dr = -k_restraint · (r - r_initial)
                f_restr = -k_restraint_f * d_restr
                force += f_restr
                # Diagnostic loss: mean squared displacement (Å²),
                # parallel to the bond / angle reporting style.
                restraint_loss = float(np.mean(np.sum(d_restr * d_restr, axis=1)))
                if atom_cost is not None:
                    # 0.5 k Σ‖Δ‖² split per atom (each atom owns its
                    # own restraint term, no double-counting).
                    np.add.at(
                        atom_cost,
                        np.arange(num_atoms),
                        0.5 * k_restraint_f * np.sum(d_restr * d_restr, axis=1),
                    )

            # ---------- record loss ----------
            total_loss = (
                bond_loss
                + angle_loss
                + repulsion_loss / max(num_atoms, 1)
                + restraint_loss * k_restraint_f
            )
            loss_history[step] = total_loss
            bond_loss_history[step] = bond_loss
            angle_loss_history[step] = angle_loss
            repulsion_loss_history[step] = repulsion_loss
            restraint_loss_history[step] = restraint_loss
            if trajectory is not None:
                trajectory[step] = pos.astype(np.float32)
            if atom_cost_history is not None and atom_cost is not None:
                atom_cost_history[step] = atom_cost.astype(np.float32)
            if total_loss < best_loss:
                best_loss = total_loss
                best_positions = pos.copy()
            best_loss_history[step] = best_loss

            # ---------- integrate (skip on last step) ----------
            if step < num_steps:
                # Freeze interior grain atoms: zero force and velocity
                if is_boundary is not None:
                    interior_mask = ~is_boundary
                    if np.any(interior_mask):
                        force[interior_mask] = 0.0
                        velocity[interior_mask] = 0.0

                force_mag = np.linalg.norm(force, axis=1)
                clip_mask = force_mag > max_force_clip
                if np.any(clip_mask):
                    force[clip_mask] *= (max_force_clip / force_mag[clip_mask])[:, None]

                # FIRE-inspired: reset velocity on direction reversal
                vf_dot = np.sum(velocity * force)
                if vf_dot < 0:
                    velocity[:] = 0.0
                else:
                    velocity = momentum * velocity + current_step * force

                new_pos = pos + velocity

                frac = new_pos @ cell_inv
                frac %= 1.0
                self.atoms.positions = frac @ cell_mat

                current_step *= step_decay

            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.update(num_steps)

        # Restore best positions
        frac_best = best_positions @ cell_inv
        frac_best %= 1.0
        self.atoms.positions = frac_best @ cell_mat

        # Store history
        step_arr = np.arange(num_steps + 1, dtype=np.int32)
        self.shell_relax_history = {
            "step": step_arr,
            "loss": loss_history,
            "best_loss": best_loss_history,
            "bond_loss": bond_loss_history,
            "angle_loss": angle_loss_history,
            "repulsion_loss": repulsion_loss_history,
            "restraint_loss": restraint_loss_history,
        }
        if trajectory is not None:
            self.shell_relax_history["trajectory"] = trajectory
        if atom_cost_history is not None:
            self.shell_relax_history["atom_cost"] = atom_cost_history

        # Invalidate caches
        self.current_distribution = None
        self.current_cost = None
        self.mc_history = None
        self.last_temperature = None
        self._rebuild_spatial_index()

        summary: dict[str, Any] = {
            "num_steps": int(num_steps),
            "bond_weight": float(bond_weight),
            "angle_weight": float(angle_weight),
            "repulsion_weight": float(repulsion_weight),
            "k_restraint": float(k_restraint),
            "step_size": float(step_size),
            "step_decay": float(step_decay),
            "neighbor_update_interval": int(neighbor_update_interval),
            "neighbor_cutoff_scale": float(neighbor_cutoff_scale),
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(best_loss),
            "num_atoms": num_atoms,
        }
        return summary

    # ------------------------------------------------------------------
    # thermal_relax: Metropolis Monte-Carlo with temperature schedule
    # ------------------------------------------------------------------

    def thermal_relax(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        *,
        num_sweeps: int = 1000,
        T_schedule="anneal",
        T_start: float = 0.05,
        T_end: float = 0.001,
        hold_sweeps: int = 200,
        step_sigma: float = 0.05,
        smart_dt: float = 0.02,
        adapt_step: bool = True,
        target_accept: float = 0.4,
        move_probs: "dict | None" = None,
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        k_restraint: float = 0.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        neighbor_update_interval: int = 100,
        rep_neighbor_update_interval: int = 20,
        capture_stride: int = 10,
        capture_trajectory: bool = True,
        restore_best: bool = True,
        freeze_interior: bool | None = None,
        freeze_mask: np.ndarray | None = None,
        grain_moves: bool | None = None,
        grain_move_interval: int = 1,
        grain_sigma_rot: float = 0.01,
        grain_sigma_trans: float = 0.01,
        show_progress: bool = True,
    ) -> dict:
        """Temperature-dependent Metropolis Monte-Carlo relaxation.

        Sits alongside :meth:`shell_relax` - same spring-network
        energy (bond + angle + repulsion from ``shell_target``), but
        moves atoms by Metropolis accept/reject with a temperature
        schedule.  Good for escaping the local minima that gradient
        descent settles into and for producing annealed amorphous
        configurations.

        A typical workflow is: run :meth:`shell_relax` or
        :meth:`generate` once to bring the initial random / tile
        configuration into a low-energy basin, then call
        :meth:`thermal_relax` with a moderate-to-cold anneal schedule
        to explore the basin and settle into the global minimum of
        that basin.

        Parameters
        ----------
        shell_target
            First-shell coordination targets, same object used by
            :meth:`shell_relax`.
        num_sweeps
            Number of sweeps.  Each sweep performs ``len(atoms)``
            trial moves.
        T_schedule
            ``"hold"`` (constant ``T_start``), ``"anneal"`` (hold at
            ``T_start`` for ``hold_sweeps`` then linearly drop to
            ``T_end``), or a ``callable(sweep_index) -> T``.
        T_start, T_end, hold_sweeps
            Parameters for the ``"anneal"`` schedule.
        step_sigma
            Initial Gaussian trial-move amplitude in Å.
        adapt_step
            If ``True``, adapt ``step_sigma`` every 20 sweeps to
            target ``target_accept`` acceptance rate.
        move_probs
            Dictionary with optional keys ``"displace"``, ``"swap"``,
            ``"smart"``.  Weights are normalised; default
            ``{"displace": 1.0}``.  ``"swap"`` exchanges virtual
            species between two atoms of different species; useful
            for multi-element + composite-shell-target systems.
            ``"smart"`` is a force-biased Langevin proposal with
            Metropolis correction (reserved for v2; currently treated
            as no-op).
        bond_weight, angle_weight, repulsion_weight,
        hard_core_scale, nonbond_push_scale
            Same as :meth:`shell_relax`.
        k_restraint
            Spring constant (eV / Å²) for a global position-restraint
            energy ``½ k_restraint Σ ‖r_i - r_initial_i‖²`` that tethers
            every atom to its starting position.  ``0.0`` (default)
            disables the term and reproduces unrestrained MC.  Small
            values (~0.1 - 1.0) preserve the input regime character
            (grain layout, amorphous topology) while still permitting
            local relaxation; large values (≫ 10) hold the structure
            essentially rigid.  Differentiable + globally defined, so
            unlike a hard ``freeze_interior`` the cost surface stays
            smooth and the relaxation can find consistent low-strain
            configurations across grain boundaries.
        neighbor_update_interval
            Rebuild the bond topology every this many sweeps.
        capture_stride
            Store a history frame every this many sweeps.  Smaller =
            finer trajectory movie, larger = less memory.
        capture_trajectory
            Whether to store per-sweep positions.  Disable for very
            long runs where only cost / T / accept_rate matter.
        freeze_interior
            Legacy hard-freeze of crystalline grain interiors.  Only
            takes effect when explicitly set to ``True`` *and* the cell
            has populated ``_grain_ids`` / ``_grain_seeds`` (i.e. came
            from :meth:`generate` with ``grain_size``).  Default
            ``None`` leaves all atoms free; prefer ``k_restraint > 0``
            for a smooth, differentiable, regime-preserving alternative.
        freeze_mask
            Explicit per-atom freeze mask of shape ``(num_atoms,)``.
            ``True`` entries are pinned for the entire run.  Overrides
            ``freeze_interior``.  Useful when the cell isn't grain-tiled
            but you still want to hold specific atoms (e.g. an
            interface) fixed.
        grain_moves
            If ``True`` (or ``None`` and the cell has ≥2 grains),
            propose rigid rotation + translation of each grain every
            ``grain_move_interval`` sweeps.
        grain_move_interval, grain_sigma_rot, grain_sigma_trans
            Frequency and amplitude of the grain rigid-body proposals.

        Returns
        -------
        dict
            History dictionary assigned to
            ``self.thermal_relax_history``.  Layout matches
            :meth:`shell_relax_history` where possible, so
            :meth:`export_trajectory_html` can visualise the
            trajectory unchanged.
        """
        from ._thermal_mc import thermal_relax_impl

        # Cache a deep-copy of the starting state so
        # ``plot_thermal_before_after`` can show initial vs final.
        from ase.atoms import Atoms
        self._thermal_start_snapshot = Atoms(
            numbers=self.atoms.numbers.copy(),
            positions=self.atoms.positions.copy(),
            cell=self.atoms.cell.array.copy(),
            pbc=self.atoms.pbc,
        )
        if getattr(self, "_atom_shell_species_index", None) is not None:
            self._thermal_start_shell_species = self._atom_shell_species_index.copy()
        else:
            self._thermal_start_shell_species = None

        species_idx = (
            self._atom_shell_species_index
            if getattr(self, "_atom_shell_species_index", None) is not None
            else self._atom_species_index
        ).astype(np.intp, copy=True)

        # --- resolve freeze_mask ---
        # Priority: explicit mask > explicit freeze_interior=True.
        # ``freeze_interior=None`` (default) leaves all atoms free; the
        # modern way to preserve regime structure is the differentiable
        # ``k_restraint`` term (see kwarg above), which gives a smooth
        # tether instead of a hard freeze.  ``freeze_interior=True`` is
        # retained for back-compat with workflows that explicitly opted
        # into the legacy hard-freeze behaviour.
        if freeze_mask is not None:
            _freeze_mask_arr = np.asarray(freeze_mask, dtype=bool)
        elif (
            freeze_interior is True
            and getattr(self, "_grain_ids", None) is not None
            and getattr(self, "_grain_seeds", None) is not None
        ):
            from ._thermal_mc import detect_grain_boundary_atoms
            pair_peak_max = float(np.max(np.asarray(shell_target.pair_peak)))
            is_boundary = detect_grain_boundary_atoms(
                self.atoms, self._grain_ids, self._grain_seeds,
                pair_peak_max=pair_peak_max,
            )
            # Freeze interior = everyone who is NOT on a grain boundary.
            _freeze_mask_arr = ~is_boundary
            if show_progress:
                nfroz = int(np.sum(_freeze_mask_arr))
                print(
                    f"thermal_relax: freezing {nfroz}/{len(self.atoms)} "
                    f"interior atoms; {len(self.atoms)-nfroz} grain-boundary "
                    f"atoms will move.  (Consider ``k_restraint`` as a "
                    f"smoother alternative — it tethers all atoms to "
                    f"their starting positions with a spring instead of "
                    f"hard-freezing the interior.)"
                )
        else:
            _freeze_mask_arr = None

        # Grain rigid-moves: default-on when cell has >=2 grains.
        _grain_ids_for_impl = None
        _use_grain_moves = (
            (grain_moves is True or grain_moves is None)
            and getattr(self, "_grain_ids", None) is not None
        )
        if _use_grain_moves:
            gids = np.asarray(self._grain_ids, dtype=np.intp)
            if int(np.unique(gids[gids >= 0]).size) >= 2:
                _grain_ids_for_impl = gids
                if show_progress:
                    nonneg_gids = gids[gids >= 0]
                    n_gr = int(np.unique(nonneg_gids).size)
                    print(
                        f"thermal_relax: will propose rigid rotations + "
                        f"translations of {n_gr} grains every "
                        f"{grain_move_interval} sweep(s) "
                        f"(σ_rot={grain_sigma_rot} rad, "
                        f"σ_trans={grain_sigma_trans} Å)."
                    )

        _kwargs_for_impl = dict(
            restore_best=bool(restore_best),
            freeze_mask=_freeze_mask_arr,
            grain_ids=_grain_ids_for_impl,
            grain_move_interval=int(grain_move_interval),
            grain_sigma_rot=float(grain_sigma_rot),
            grain_sigma_trans=float(grain_sigma_trans),
        )
        history = thermal_relax_impl(
            self.atoms,
            species_idx,
            shell_target,
            num_sweeps=int(num_sweeps),
            T_schedule=T_schedule,
            T_start=float(T_start),
            T_end=float(T_end),
            hold_sweeps=int(hold_sweeps),
            step_sigma=float(step_sigma),
            smart_dt=float(smart_dt),
            adapt_step=bool(adapt_step),
            target_accept=float(target_accept),
            move_probs=move_probs,
            bond_weight=float(bond_weight),
            angle_weight=float(angle_weight),
            repulsion_weight=float(repulsion_weight),
            k_restraint=float(k_restraint),
            hard_core_scale=float(hard_core_scale),
            nonbond_push_scale=float(nonbond_push_scale),
            neighbor_update_interval=int(neighbor_update_interval),
            rep_neighbor_update_interval=int(rep_neighbor_update_interval),
            capture_stride=int(capture_stride),
            capture_trajectory=bool(capture_trajectory),
            rng_seed=int(self.rng.integers(0, 2**31 - 1)) if hasattr(self, "rng") else None,
            show_progress=bool(show_progress),
            **_kwargs_for_impl,
        )

        # Write the final species assignments back to the Supercell.
        if getattr(self, "_atom_shell_species_index", None) is not None:
            self._atom_shell_species_index = history["final_species_idx"]
        # MC invalidates the cached g3 measurement.
        self.current_distribution = None
        self._rebuild_spatial_index()

        self.thermal_relax_history = history
        return history

    def plot_shell_relax(
        self: "Supercell",
        *,
        log_y: bool = False,
    ):
        """Plot the recorded shell-relax loss history using Matplotlib."""
        if self.shell_relax_history is None:
            raise ValueError("Run shell_relax() before plotting the history.")

        import matplotlib.pyplot as plt

        hist = self.shell_relax_history
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(hist["step"], hist["loss"], lw=1.8, label="total loss")
        ax.plot(hist["step"], hist["best_loss"], lw=1.4, ls="--", label="best")
        ax.plot(hist["step"], hist["bond_loss"], lw=1.0, alpha=0.7, label="bond")
        ax.plot(hist["step"], hist["angle_loss"], lw=1.0, alpha=0.7, label="angle")
        ax.plot(
            hist["step"],
            hist["repulsion_loss"] / max(len(self.atoms), 1),
            lw=1.0, alpha=0.7, label="repulsion (per atom)",
        )
        # Position-restraint loss is only surfaced when k_restraint > 0
        # was used; older histories (or runs at k=0) don't show this
        # curve so the plot stays clean for unrestrained relaxations.
        # Use a small absolute tolerance so floating-point noise from
        # the ``total - bond - angle - rep`` decomposition doesn't
        # accidentally trigger the curve at k=0.
        restraint = hist.get("restraint_loss")
        if restraint is not None and float(np.max(np.asarray(restraint))) > 1e-6:
            ax.plot(hist["step"], restraint, lw=1.0, alpha=0.7,
                    label="restraint", color="#9467bd")
        if log_y:
            positive = hist["loss"][hist["loss"] > 0.0]
            if positive.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Shell relax history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def plot_thermal_relax(
        self: "Supercell",
        *,
        log_y: bool = False,
        log_x: bool = False,
    ):
        """Plot thermal-MC history: cost + T on shared x-axis.

        Two stacked panels with ``sharex=True`` so the hold-plateau /
        anneal-ramp shape in temperature lines up visually with the
        cost trajectory.

        Parameters
        ----------
        log_y
            Log-scale the cost axis.  Recommended for convergence
            checking - a converged run shows a clear plateau when
            the cost curve flattens on a log scale.
        log_x
            Log-scale the sweep axis.  Off by default because a
            typical hold+anneal schedule is linear in sweep index, so
            log-x distorts the temperature ramp visually.  Useful if
            your schedule spans many orders of magnitude in sweep
            number or you want to emphasize early-sweep dynamics.
        """
        hist = getattr(self, "thermal_relax_history", None)
        if hist is None:
            raise ValueError("Run thermal_relax() before plotting.")

        import matplotlib.pyplot as plt

        fig, (ax_cost, ax_T) = plt.subplots(
            2, 1, figsize=(7.0, 5.0), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.08},
        )
        # Normalise to per-atom so the axis is comparable across
        # regime sizes and across systems of different N.
        n = float(max(1, len(self.atoms)))
        ax_cost.plot(hist["sweep"], np.asarray(hist["cost"]) / n, lw=1.8, label="total")
        ax_cost.plot(hist["sweep"], np.asarray(hist["best_cost"]) / n, lw=1.2, ls="--", label="best")
        ax_cost.plot(hist["sweep"], np.asarray(hist["cost_bond"]) / n, lw=0.9, alpha=0.7, label="bond")
        ax_cost.plot(hist["sweep"], np.asarray(hist["cost_angle"]) / n, lw=0.9, alpha=0.7, label="angle")
        ax_cost.plot(hist["sweep"], np.asarray(hist["cost_rep"]) / n, lw=0.9, alpha=0.7, label="repulsion")
        # Position-restraint cost is only surfaced when k_restraint > 0
        # was used; older histories (or runs at k=0) don't surface this
        # curve so the legend stays compact for unrestrained runs.
        # Tolerance compares to the ``cost`` magnitude so the curve
        # only shows when the restraint is a meaningful fraction of
        # total energy (filters FP noise from the
        # ``total - bond - angle - rep`` decomposition at k=0).
        cost_restraint = hist.get("cost_restraint")
        if cost_restraint is not None:
            restraint_arr = np.asarray(cost_restraint)
            cost_arr = np.asarray(hist["cost"])
            cost_scale = float(np.max(np.abs(cost_arr))) + 1e-12
            if float(np.max(np.abs(restraint_arr))) > 1e-6 * cost_scale:
                ax_cost.plot(hist["sweep"], restraint_arr / n,
                             lw=0.9, alpha=0.7, label="restraint",
                             color="#9467bd")
        if log_y:
            ax_cost.set_yscale("log")
        if log_x:
            # Shared x-axis: setting it on one panel sets both.
            ax_cost.set_xscale("log")
        ax_cost.set_ylabel("cost / atom")
        ax_cost.set_title("Thermal MC history")
        ax_cost.legend(ncols=6, fontsize=9, loc="upper right")
        ax_cost.grid(alpha=0.25)

        ax_T.plot(hist["sweep"], hist["T"], color="#c2454c", lw=1.6)
        ax_T.set_ylabel("temperature")
        ax_T.set_xlabel("sweep")
        ax_T.grid(alpha=0.25)

        # Secondary: acceptance rate vs sweep on a twin axis in the T panel.
        ax_acc = ax_T.twinx()
        ax_acc.plot(hist["sweep"], hist["accept_rate"], color="#2a6e4e",
                    lw=1.0, alpha=0.7, label="accept rate")
        ax_acc.set_ylabel("accept rate", color="#2a6e4e")
        ax_acc.tick_params(axis="y", labelcolor="#2a6e4e")
        ax_acc.set_ylim(0.0, 1.0)

        fig.tight_layout()
        return fig, (ax_cost, ax_T)

    def plot_thermal_before_after(
        self: "Supercell",
        *,
        r_max: float = 8.0,
        title: str | None = None,
    ):
        """Compare g(r) before and after thermal_relax.

        Returns the IPython HTML object from
        :func:`tricor.plot_g2_compare`.  The "before" snapshot is the
        cell state cached at the start of :meth:`thermal_relax`.
        """
        snap = getattr(self, "_thermal_start_snapshot", None)
        if snap is None:
            raise ValueError(
                "No pre-thermal snapshot; call thermal_relax() first."
            )
        hist = getattr(self, "thermal_relax_history", None)
        if hist is None:
            raise ValueError("Run thermal_relax() before plotting.")

        # Build a shadow Supercell-like object for the 'before' entry
        # so plot_g2_compare can measure g(r) from it.  Re-use the
        # current cell's shell-target + grid by copy.
        from copy import copy
        before_cell = copy(self)
        before_cell.atoms = snap
        before_cell.current_distribution = None
        if self._thermal_start_shell_species is not None:
            before_cell._atom_shell_species_index = (
                self._thermal_start_shell_species.copy()
            )
        # Rebuild spatial index (if the helper exists).
        if hasattr(before_cell, "_rebuild_spatial_index"):
            before_cell._rebuild_spatial_index()

        from ._plotting import plot_g2_compare as _plot_g2_compare
        cells = {
            "initial (static relax)": before_cell,
            "final (thermal MC)": self,
        }
        return _plot_g2_compare(cells, r_max=r_max, title=title or "Thermal MC: before vs after")
