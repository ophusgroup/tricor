from __future__ import annotations

from pathlib import Path as _Path
from typing import TYPE_CHECKING

import numpy as np
from ase.neighborlist import neighbor_list

from .g3 import _EPS, _TextProgressBar

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell


_STATIC_DIR = _Path(__file__).parent / "static"
_TRAJECTORY_HTML_TEMPLATE = (_STATIC_DIR / "trajectory_viewer.html").read_text()
_G3_HTML_TEMPLATE = (_STATIC_DIR / "g3_viewer.html").read_text()
_OVERVIEW_HTML_TEMPLATE = (_STATIC_DIR / "overview_viewer.html").read_text()


def export_overview_html(
    output_path: str,
    cells_and_labels,
    *,
    grid_cols: int = 3,
    atom_scale: float = 0.18,
    bond_radius: float = 0.07,
    bond_color=(0.95, 0.1, 0.1),
    background_color: str = "#f7f8f5",
    title: str = "",
    subtitle: str = "",
    bond_cutoff_scale: float = 1.2,
    max_bonds_per_atom: int = 4,
    bond_length_tol: float = 0.10,
    ideal_angle_deg: float = 109.47,
    bond_angle_tol_deg: float = 18.0,
) -> str:
    """Export a grid of static 3D structures as a self-contained HTML file.

    Each panel renders the final atoms of one :class:`Supercell` using ASE
    element colours, black outlines, and red bonds.  All panels share a
    camera that auto-rotates; dragging any panel pauses the rotation and
    orbits manually.

    Parameters
    ----------
    output_path
        Path to write the HTML file.
    cells_and_labels
        Iterable of ``(supercell, label)`` pairs.
    grid_cols
        Number of columns in the grid (rows are inferred).
    atom_scale, bond_radius
        Same meaning as in :meth:`export_trajectory_html`.
    bond_color
        RGB tuple in [0,1] for the bond colour (default red).
    background_color, title, subtitle
        Cosmetic.
    bond_cutoff_scale
        Bond cutoff = ``shell_target.pair_peak.max() * bond_cutoff_scale``
        (fallback ``3.0`` Å if no ``shell_target`` is attached).
    """
    import json
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors

    structures = []
    for cell, label in cells_and_labels:
        atoms = cell.atoms

        shell_target = getattr(cell, "_shell_target", None)
        if shell_target is not None:
            pair_peak = float(np.max(
                np.asarray(shell_target.pair_peak, dtype=np.float64),
            ))
        else:
            pair_peak = 2.35
        cutoff_lo = pair_peak * (1.0 - bond_length_tol)
        cutoff_hi = pair_peak * (1.0 + bond_length_tol)
        search_cutoff = pair_peak * bond_cutoff_scale

        bi_all, bj_all, bd_all, bD_all = neighbor_list(
            "ijdD", atoms, float(search_cutoff),
        )

        # Radial filter - keep only near-ideal bond lengths.
        length_ok = (bd_all >= cutoff_lo) & (bd_all <= cutoff_hi)
        bi_all = bi_all[length_ok]
        bj_all = bj_all[length_ok]
        bd_all = bd_all[length_ok]
        bD_all = bD_all[length_ok]

        # Take the ``max_bonds_per_atom`` shortest in-band neighbours per atom.
        order = np.lexsort((bd_all, bi_all))
        bi_s = bi_all[order]
        bj_s = bj_all[order]
        bD_s = bD_all[order]
        keep_mask = np.zeros(bi_s.size, dtype=bool)
        if bi_s.size:
            unique_i, start_idx = np.unique(bi_s, return_index=True)
            end_idx = np.concatenate([start_idx[1:], [bi_s.size]])
            for s, e in zip(start_idx, end_idx):
                keep_mask[s : min(s + max_bonds_per_atom, e)] = True
        bi_top = bi_s[keep_mask]
        bj_top = bj_s[keep_mask]
        bD_top = bD_s[keep_mask]

        # Angular filter: only keep bonds from atoms whose selected neighbours
        # form a tetrahedron whose 6 pair-wise angles are all within
        # ``bond_angle_tol_deg`` of ``ideal_angle_deg``.  Atoms that don't
        # satisfy this test contribute no bonds at all.
        ideal_angle = float(np.deg2rad(ideal_angle_deg))
        angle_tol = float(np.deg2rad(bond_angle_tol_deg))
        needed = int(max_bonds_per_atom)

        n_atoms = len(atoms)
        per_atom_js: list[list] = [[] for _ in range(n_atoms)]
        per_atom_vs: list[list] = [[] for _ in range(n_atoms)]
        for i_, j_, v_ in zip(bi_top.tolist(), bj_top.tolist(), bD_top):
            if len(per_atom_js[i_]) < needed:
                per_atom_js[i_].append(int(j_))
                per_atom_vs[i_].append(v_)

        good_pairs: set[tuple[int, int]] = set()
        for i_ in range(n_atoms):
            js = per_atom_js[i_]
            if len(js) < needed:
                continue
            vecs = np.asarray(per_atom_vs[i_], dtype=np.float64)
            norms = np.linalg.norm(vecs, axis=1)
            if np.any(norms <= 1e-9):
                continue
            unit = vecs / norms[:, None]
            # Pairwise cosines (needed×needed); off-diagonal entries are the
            # 6 angles we care about for a 4-neighbour atom.
            cos = np.clip(unit @ unit.T, -1.0, 1.0)
            angles = np.arccos(cos)
            triu = np.triu_indices(needed, k=1)
            dev = np.abs(angles[triu] - ideal_angle)
            if np.max(dev) <= angle_tol:
                for j_ in js:
                    good_pairs.add((min(i_, j_), max(i_, j_)))

        if good_pairs:
            pair_arr = np.asarray(sorted(good_pairs), dtype=np.int32)
            bi = pair_arr[:, 0]
            bj = pair_arr[:, 1]
        else:
            bi = np.zeros(0, dtype=np.int32)
            bj = np.zeros(0, dtype=np.int32)

        numbers = atoms.numbers
        colors = np.array([jmol_colors[z] for z in numbers], dtype=np.float32)
        radii = np.array([covalent_radii[z] for z in numbers], dtype=np.float32)
        cell_mat = np.asarray(atoms.cell.array, dtype=np.float32)

        centre = 0.5 * np.sum(cell_mat, axis=0)
        pos = (atoms.positions - centre).astype(np.float32)
        # Round to 3 decimals (Å) to keep the JSON footprint small.
        pos = np.round(pos, 3)

        structures.append({
            "label": label,
            "num_atoms": int(len(atoms)),
            "num_bonds": int(len(bi)),
            "positions": pos.ravel().tolist(),
            "atom_colors": colors.ravel().tolist(),
            "atom_radii": np.round(radii, 3).tolist(),
            "bond_i": bi.tolist(),
            "bond_j": bj.tolist(),
            "cell_matrix": np.round(cell_mat, 3).ravel().tolist(),
        })

    data = {
        "structures": structures,
        "grid_cols": int(grid_cols),
        "grid_rows": int(-(-len(structures) // grid_cols)),
        "atom_scale": float(atom_scale),
        "bond_radius": float(bond_radius),
        "bond_color": list(bond_color),
        "background_color": background_color,
        "title": title,
        "subtitle": subtitle,
    }

    html = _OVERVIEW_HTML_TEMPLATE.replace(
        "__TRICOR_DATA_PLACEHOLDER__",
        json.dumps(data),
    )
    output_path = str(output_path)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def _detect_shell_mask(triplet_data: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Auto-detect the first NN shell over r for one triplet of g3."""
    # Pair profile: collapse both angular-partner and phi dimensions.
    profile = triplet_data.sum(axis=(1, 2)) + triplet_data.sum(axis=(0, 2))
    profile = profile / np.maximum(r * r, _EPS)
    finite = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.zeros_like(r, dtype=bool)
    positive = np.flatnonzero(finite > 0)
    if positive.size == 0:
        return mask
    start = int(positive[0])
    peak_bin = None
    for idx in range(max(start, 1), finite.size - 1):
        if finite[idx] >= finite[idx - 1] and finite[idx] > finite[idx + 1]:
            peak_bin = idx
            break
    if peak_bin is None:
        peak_bin = int(start + int(np.argmax(finite[start:])))
    threshold = max(1e-12, 0.01 * float(finite[peak_bin]))
    leading = np.flatnonzero(finite[: peak_bin + 1] > threshold)
    left_bin = int(leading[0]) if leading.size else start
    right_bin = min(finite.size - 1, peak_bin + 1)
    for idx in range(peak_bin + 1, finite.size - 1):
        if finite[idx] <= finite[idx - 1] and finite[idx] <= finite[idx + 1]:
            right_bin = idx
            break
    mask[left_bin : right_bin + 1] = True
    return mask


def _g3_pair_profile(
    dist: "Any",
    triplet_idx: int,
    r: np.ndarray,
) -> np.ndarray:
    """Return the per-triplet pair profile g(r), normalised so tail -> 1.0."""
    g2 = getattr(dist, "g2", None)
    g3_index = getattr(dist, "g3_index", None)
    if g2 is not None and g3_index is not None:
        center_ind, neigh1_ind, neigh2_ind = g3_index[triplet_idx]
        profile = np.asarray(g2[center_ind, neigh1_ind], dtype=np.float64).copy()
        profile += np.asarray(g2[center_ind, neigh2_ind], dtype=np.float64)
    else:
        triplet_data = np.asarray(dist.g3[triplet_idx], dtype=np.float64)
        profile = triplet_data.sum(axis=(1, 2)) + triplet_data.sum(axis=(0, 2))

    profile = profile / np.maximum(r * r, _EPS)
    # Scale so the tail converges to 1.0
    tail_start = 0.7 * float(r[-1])
    tail_mask = r >= tail_start
    if not np.any(tail_mask):
        tail_mask = np.zeros_like(r, dtype=bool)
        tail_mask[-max(1, r.size // 4) :] = True
    tail = profile[tail_mask]
    finite = tail[np.isfinite(tail)]
    scale = float(np.mean(finite)) if finite.size else 1.0
    if scale <= _EPS:
        scale = 1.0
    return (profile / scale).astype(np.float32)


def _g3_slice_image(
    triplet_data: np.ndarray,
    shell_mask: np.ndarray,
    r: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """Compute the (num_phi, num_r) reduced-density slice for one triplet.

    Normalised so that the uniform far-field tends to 1.0.
    """
    image = triplet_data[shell_mask, :, :].sum(axis=0)
    image += triplet_data[:, shell_mask, :].sum(axis=1)
    image = image.T  # (num_phi, num_r)

    phi_rad = np.deg2rad(phi_deg)
    phi_factor = np.maximum(np.sin(phi_rad), 1e-3)[:, None]
    radial_factor = np.maximum(r * r, _EPS)[None, :]
    image = image / (phi_factor * radial_factor)

    tail_start = 0.7 * float(r[-1])
    tail_mask = r >= tail_start
    if not np.any(tail_mask):
        tail_mask = np.zeros_like(r, dtype=bool)
        tail_mask[-max(1, r.size // 4) :] = True
    tail = image[:, tail_mask]
    finite = tail[np.isfinite(tail)]
    scale = float(np.mean(finite)) if finite.size else 1.0
    if scale <= _EPS:
        scale = 1.0
    return (image / scale).astype(np.float32)


def _nice_round_up(v: float) -> float:
    """Round *v* up to a ``nice`` number whose half is also clean.

    Picks the smallest value in [1, 2, 4, 5, 10] x 10**k that is >= v.
    E.g. 3.85 -> 4.0, 0.87 -> 1.0, 12.5 -> 20.0, 2.1 -> 4.0.
    """
    import math

    if v <= 0 or not math.isfinite(v):
        return 1.0
    exp = math.floor(math.log10(v))
    magnitude = 10.0 ** exp
    mantissa = v / magnitude
    for m in (1.0, 2.0, 4.0, 5.0, 10.0):
        if mantissa <= m + 1e-9:
            return m * magnitude
    return 10.0 * magnitude


class _PlottingMixin:
    """Plotting methods extracted from Supercell."""

    def view_structure(
        self: "Supercell",
        shell_target: "CoordinationShellTarget | None" = None,
        **kwargs,
    ):
        """Return an interactive 3D structure viewer widget.

        Renders atoms as spheres (coloured by element) and bonds as
        cylinders inside the periodic cell outline.  Uses Three.js
        WebGL with OrbitControls for drag-to-rotate / scroll-to-zoom.

        Parameters
        ----------
        shell_target
            Sets the default bond cutoff from
            ``shell_target.max_pair_outer``.  If ``None``, uses the
            shell_target stored from the last :meth:`generate` call.
        **kwargs
            Forwarded to :class:`StructureWidget` (e.g. ``atom_scale``,
            ``bond_cutoff``, ``show_bonds``, ``slab_x``, etc.).

        Returns
        -------
        StructureWidget
            An anywidget instance for display in Jupyter.
        """
        from .structure_widget import StructureWidget

        if shell_target is None:
            shell_target = getattr(self, "_shell_target", None)

        return StructureWidget(
            self.atoms,
            shell_target=shell_target,
            grain_ids=self._grain_ids,
            **kwargs,
        )

    def plot_g3(
        self: "Supercell",
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive explorer for the supercell's measured g3.

        Requires :meth:`measure_g3` to have been called first.

        Parameters
        ----------
        pair
            Triplet index or label (e.g. ``0`` or ``"Si-Si-Si"``).
        normalize
            If ``True``, display the reduced (density-normalised) g3.
        """
        if self.current_distribution is None:
            raise ValueError("Call measure_g3() before plot_g3().")
        dist = self.current_distribution
        return dist.plot_g3(pair=pair, normalize=normalize)

    def plot_g3_compare(
        self: "Supercell",
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive comparison between the current supercell and target."""
        current = self.measure_g3()
        pair_index = self.target_distribution._resolve_pair_index(pair)

        from .g3_compare_widget import G3CompareWidget

        return G3CompareWidget(
            current_distribution=current,
            target_distribution=self.target_distribution,
            triplet_index=pair_index,
            normalize=normalize,
            supercell_title=f"{self.label} g3 slice",
            target_title=f"{self.target_distribution.label} g3 slice",
            status_prefix=(
                f"density {self.relative_density:.3f} | "
                "cell_dim_angstroms "
                f"{self.cell_dim_angstroms[0]:.2f}x"
                f"{self.cell_dim_angstroms[1]:.2f}x"
                f"{self.cell_dim_angstroms[2]:.2f} A"
            ),
        )

    def _display_compare_widget(self: "Supercell") -> None:
        """Display the comparison widget immediately when running in IPython."""
        try:
            from IPython.display import display
        except Exception:
            return
        display(self.plot_g3_compare())

    def plot_monte_carlo(
        self: "Supercell",
        *,
        log_y: bool = False,
        show_run_boundaries: bool = True,
    ):
        """Plot the recorded Monte Carlo cost history using Matplotlib."""
        if self.mc_history is None:
            raise ValueError("Run monte_carlo() before plotting the history.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.mc_history["step"], self.mc_history["cost"], lw=1.8, label="cost")
        ax.plot(self.mc_history["step"], self.mc_history["best_cost"], lw=1.4, label="best")
        if show_run_boundaries and "run_index" in self.mc_history:
            run_index = np.asarray(self.mc_history["run_index"], dtype=np.int32)
            step = np.asarray(self.mc_history["step"], dtype=np.int32)
            change_points = np.flatnonzero(np.diff(run_index) > 0)
            for point_index in change_points:
                ax.axvline(
                    float(step[point_index + 1]),
                    color="0.65",
                    lw=0.9,
                    ls="--",
                    alpha=0.7,
                )
        if log_y:
            positive_cost = np.asarray(self.mc_history["cost"], dtype=np.float64)
            positive_cost = positive_cost[positive_cost > 0.0]
            if positive_cost.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        ax.set_title("Monte Carlo cost history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def export_trajectory_html(
        self: "Supercell",
        output_path: str,
        *,
        bond_cutoff: float | None = None,
        atom_scale: float = 0.32,
        bond_radius: float = 0.06,
        background_color: str = "#f7f8f5",
        title: str = "",
    ) -> str:
        """Export an interactive 3D trajectory viewer as a self-contained HTML file.

        Requires :meth:`shell_relax` or :meth:`generate` to have been run
        with ``capture_trajectory=True``.  The resulting HTML embeds the
        full position trajectory, uses Three.js (from CDN) for rendering,
        and provides play/pause/slider controls.

        Parameters
        ----------
        output_path
            Path to write the HTML file.
        bond_cutoff
            Maximum bond length in Angstrom.  If ``None`` and the
            supercell was generated with a shell_target, uses
            ``shell_target.max_pair_outer * 1.2``.  Otherwise 3.0.
        atom_scale
            Radius scale for atom spheres (multiplied by covalent radii).
        bond_radius
            Radius of bond cylinders in Angstrom.
        background_color
            CSS colour for the viewer background.
        title
            Optional title displayed above the viewer.

        Returns
        -------
        str
            The output path.
        """
        import json
        from ase.data import covalent_radii
        from ase.data.colors import jmol_colors

        history = self.shell_relax_history
        if history is None or "trajectory" not in history:
            raise ValueError(
                "No trajectory data available.  Run shell_relax() or "
                "generate() with capture_trajectory=True first."
            )
        trajectory = np.asarray(history["trajectory"], dtype=np.float32)
        n_frames, n_atoms, _ = trajectory.shape

        atom_cost = history.get("atom_cost")
        if atom_cost is not None:
            atom_cost = np.asarray(atom_cost, dtype=np.float32)
            # Global colour range (constant across frames).  Use 99th percentile
            # of non-zero values to avoid single hot-spot saturation, then round
            # up to a "nice" value so the 0 / mid / max ticks are clean.
            flat = atom_cost.ravel()
            cost_min = 0.0
            positive = flat[flat > 0.0]
            if positive.size:
                raw_max = float(np.percentile(positive, 99.0))
                if raw_max <= 0.0:
                    raw_max = float(positive.max())
            else:
                raw_max = 1.0
            cost_max = _nice_round_up(raw_max)

        # Bond cutoff
        shell_target = getattr(self, "_shell_target", None)
        if bond_cutoff is None:
            if shell_target is not None:
                pair_peak_max = float(np.max(
                    np.asarray(shell_target.pair_peak, dtype=np.float64),
                ))
                bond_cutoff = pair_peak_max * 1.2
            else:
                bond_cutoff = 3.0

        # Bond topology from final frame
        bi_all, bj_all, bd_all = neighbor_list(
            "ijd", self.atoms, float(bond_cutoff),
        )
        mask = bi_all < bj_all
        bi = bi_all[mask].astype(np.int32)
        bj = bj_all[mask].astype(np.int32)

        # Atom metadata
        numbers = self.atoms.numbers
        colors = np.array([jmol_colors[z] for z in numbers], dtype=np.float32)
        radii = np.array([covalent_radii[z] for z in numbers], dtype=np.float32)

        # Cell matrix (for min-image bond wrapping in JS)
        cell_mat = np.asarray(self.atoms.cell.array, dtype=np.float32)

        # Centre all frames at the cell centre
        centre = 0.5 * np.sum(cell_mat, axis=0)
        trajectory_centered = trajectory - centre

        # Pack data
        data = {
            "num_frames": int(n_frames),
            "num_atoms": int(n_atoms),
            "num_bonds": int(len(bi)),
            "atom_colors": colors.ravel().tolist(),
            "atom_radii": radii.tolist(),
            "atom_scale": float(atom_scale),
            "bond_radius": float(bond_radius),
            "bond_i": bi.tolist(),
            "bond_j": bj.tolist(),
            "cell_matrix": cell_mat.ravel().tolist(),
            "positions": trajectory_centered.ravel().tolist(),
            "background_color": background_color,
            "title": title,
        }
        if atom_cost is not None:
            data["atom_cost"] = atom_cost.ravel().tolist()
            data["cost_min"] = float(cost_min)
            data["cost_max"] = float(cost_max)
            data["cost_label"] = "per-atom cost"

        html = _TRAJECTORY_HTML_TEMPLATE.replace(
            "__TRICOR_DATA_PLACEHOLDER__",
            json.dumps(data),
        )

        output_path = str(output_path)
        with open(output_path, "w") as f:
            f.write(html)
        return output_path

    def export_g3_html(
        self: "Supercell",
        output_path: str,
        *,
        r_max: float = 10.0,
        r_step: float = 0.1,
        phi_num_bins: int = 45,
        background_color: str = "#f7f8f5",
        title: str = "",
        show_progress: bool = False,
    ) -> str:
        """Export a static 2D g3 viewer as a self-contained HTML file.

        Renders one heatmap per species-triplet of the reduced three-body
        distribution (density / uniform, where ``1.0`` = white).  The viewer
        uses a diverging RdBu_r colormap centred at ``1.0`` and lets the user
        pick the triplet and adjust the upper colour limit.

        A fresh :class:`G3Distribution` is measured from the current atoms on
        the coarse export grid (by default 50 x 45 bins per triplet) so the
        embedded JSON stays small (~500 KB) without affecting anything the
        supercell itself has cached.

        Parameters
        ----------
        output_path
            Path to write the HTML file.
        r_max, r_step, phi_num_bins
            Measurement grid for the exported distribution.
        background_color, title
            Cosmetic.
        show_progress
            Forwarded to the g3 measurement call.
        """
        import json

        from .g3 import G3Distribution

        dist = G3Distribution(self.atoms, label=f"{self.label}-export-g3")
        dist.measure_g3(
            r_max=r_max,
            r_step=r_step,
            phi_num_bins=phi_num_bins,
            show_progress=show_progress,
        )
        if dist.g3 is None:
            raise ValueError("Measured distribution has no g3 array.")

        r = np.asarray(dist.r, dtype=np.float64)
        r_edges = np.asarray(dist.bin_edges, dtype=np.float64)
        phi_deg = np.asarray(dist.phi_deg, dtype=np.float64)
        phi_edges = np.asarray(dist.phi_edges, dtype=np.float64)
        phi_edges_deg = np.rad2deg(phi_edges)
        labels = list(dist.pair_labels)
        num_triplets = int(dist.g3.shape[0])

        slices = []
        pair_profiles = []
        shell_ranges = []
        for ti in range(num_triplets):
            triplet_data = np.asarray(dist.g3[ti], dtype=np.float64)
            shell_mask = _detect_shell_mask(triplet_data, r)
            if not np.any(shell_mask):
                shell_mask = np.zeros_like(r, dtype=bool)
                shell_mask[: max(1, r.size // 4)] = True
            img = _g3_slice_image(triplet_data, shell_mask, r, phi_deg)
            slices.append(img.ravel().astype(np.float32).tolist())

            profile = _g3_pair_profile(dist, ti, r)
            pair_profiles.append(profile.tolist())

            idx = np.flatnonzero(shell_mask)
            shell_min = float(r_edges[int(idx[0])])
            shell_max = float(r_edges[int(idx[-1]) + 1])
            shell_ranges.append([shell_min, shell_max])

        data = {
            "num_r": int(r.size),
            "num_phi": int(phi_deg.size),
            "num_triplets": num_triplets,
            "r_centers": r.tolist(),
            "r_edges": r_edges.tolist(),
            "phi_centers_deg": phi_deg.tolist(),
            "phi_edges_deg": phi_edges_deg.tolist(),
            "triplet_labels": labels,
            "slices": slices,
            "pair_profiles": pair_profiles,
            "shell_ranges": shell_ranges,
            "background_color": background_color,
            "title": title,
        }

        html = _G3_HTML_TEMPLATE.replace(
            "__TRICOR_DATA_PLACEHOLDER__",
            json.dumps(data),
        )
        output_path = str(output_path)
        with open(output_path, "w") as f:
            f.write(html)
        return output_path

    def plot_structure(
        self: "Supercell",
        shell_target: "CoordinationShellTarget | None" = None,
        *,
        output: str | None = None,
        width: int = 1024,
        height: int = 1024,
        fps: int = 60,
        duration: float = 6.0,
        elevation: float = 15.0,
        atom_size: float = 10.0,
        bond_cutoff: float | None = None,
        show_cell: bool = True,
        show_atoms: bool = True,
        background: str = "white",
        colormap: str = "Reds",
        tetrahedral_thresh: float = 0.4,
        show_progress: bool = True,
    ):
        """Render a bond-centric rotating 3D view of the atomic structure.

        Bonds are the primary visual: crystalline (tetrahedral) bonds
        are drawn thick and coloured by depth; boundary / amorphous
        bonds are drawn faint.  Atoms are optional small dots.  The
        animation performs a full periodic 360-degree rotation.

        Classification follows the MATLAB ``plotAtoms02`` convention:
        an atom is *crystalline* if it has exactly K nearest neighbours
        within *bond_cutoff* **and** the mean displacement of those
        neighbours is less than *tetrahedral_thresh* (i.e. the local
        coordination is symmetric / tetrahedral).

        Parameters
        ----------
        shell_target
            First-shell targets.  Used to set *bond_cutoff* and the
            coordination number K automatically.
        output
            File path for a ``.mp4`` (recommended) or ``.gif``.
            ``None`` shows a static figure.
        width, height
            Frame size in pixels.
        fps
            Frames per second (GIF only).
        duration
            Total GIF length in seconds.  Rotation is always exactly
            360 degrees so the loop is seamless.
        elevation
            Camera elevation in degrees.
        atom_size
            Matplotlib scatter marker size.  Set to 0 to hide atoms.
        bond_cutoff
            NN bond length cutoff in Angstrom.
        show_cell
            Draw the periodic cell outline.
        show_atoms
            Draw atom dots.
        background
            Figure background colour.
        colormap
            Matplotlib colormap for crystalline bonds (coloured by
            depth / y-coordinate after rotation, like MATLAB ``bone``).
        tetrahedral_thresh
            Maximum norm of mean NN displacement vector for an atom to
            be classified as crystalline.  Smaller = stricter.
        show_progress
            Print frame counter during GIF rendering.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pos = self.atoms.positions.copy()
        cell_mat = np.asarray(self.atoms.cell.array, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_mat)
        num_atoms = len(self.atoms)

        # --- bond cutoff ---
        if bond_cutoff is None:
            if shell_target is not None:
                # Use a generous cutoff (pair_peak + 3*sigma or 1.2x)
                pair_peak_max = float(np.max(
                    np.asarray(shell_target.pair_peak, dtype=np.float64),
                ))
                bond_cutoff = pair_peak_max * 1.2
            else:
                bond_cutoff = 3.0

        if shell_target is not None:
            coord_target = np.asarray(shell_target.coordination_target, dtype=np.float64)
            species_idx = self._atom_species_index
            k_per_atom = np.array([
                int(np.round(coord_target[species_idx[a]].sum()))
                for a in range(num_atoms)
            ], dtype=np.intp)
        else:
            k_per_atom = np.full(num_atoms, 4, dtype=np.intp)

        # --- find bonds ---
        bi_all, bj_all, bd_all = neighbor_list("ijd", self.atoms, bond_cutoff)

        def _min_image(delta: np.ndarray) -> np.ndarray:
            frac = delta @ cell_inv
            frac -= np.rint(frac)
            return frac @ cell_mat

        # --- classify atoms as crystalline ---
        # Two criteria (either makes an atom crystalline):
        # 1) Tetrahedral check (MATLAB style): K NN within cutoff and
        #    symmetric coordination (mean displacement < thresh).
        #    Allow K-1 to K+1 neighbors for tolerance.
        # 2) Grain interior: deep inside a crystalline Voronoi cell.
        is_crystalline_atom = np.zeros(num_atoms, dtype=bool)

        # Criterion 1: tetrahedral / symmetric coordination
        for a in range(num_atoms):
            mask = bi_all == a
            nn_count = int(np.sum(mask))
            k_target = int(k_per_atom[a])
            if nn_count < max(k_target - 1, 1) or nn_count > k_target + 1:
                continue
            # Use K nearest for the displacement check
            dists_a = bd_all[mask]
            js_a = bj_all[mask]
            order = np.argsort(dists_a)[:k_target]
            dxyz = _min_image(pos[js_a[order]] - pos[a])
            mean_disp = np.linalg.norm(np.mean(dxyz, axis=0))
            if mean_disp < tetrahedral_thresh:
                is_crystalline_atom[a] = True

        # Criterion 2: grain interior atoms (always crystalline)
        grain_ids = self._grain_ids
        grain_seeds = self._grain_seeds
        if grain_ids is not None and grain_seeds is not None:
            pp_max = float(np.max(
                np.asarray(shell_target.pair_peak, dtype=np.float64),
            )) if shell_target is not None else 2.5
            bw = pp_max * 0.5
            delta_seeds = pos[:, None, :] - grain_seeds[None, :, :]
            frac_ds = delta_seeds @ cell_inv
            frac_ds -= np.rint(frac_ds)
            cart_ds = frac_ds @ cell_mat
            dist_to_seeds = np.sqrt(np.sum(cart_ds ** 2, axis=2))
            for ia in range(num_atoms):
                gid = grain_ids[ia]
                if gid < 0:
                    continue
                d_own = dist_to_seeds[ia, gid]
                dists_copy = dist_to_seeds[ia].copy()
                dists_copy[gid] = np.inf
                d_other = float(np.min(dists_copy))
                if (d_other - d_own) * 0.5 > bw:
                    is_crystalline_atom[ia] = True

        # Keep i < j for unique bonds
        mask_ij = bi_all < bj_all
        bi, bj = bi_all[mask_ij], bj_all[mask_ij]

        # Bond is crystalline if BOTH endpoints are crystalline
        bond_is_cryst = is_crystalline_atom[bi] & is_crystalline_atom[bj]

        # Bond segment endpoints (minimum-image)
        bond_vecs = _min_image(pos[bj] - pos[bi])
        bond_starts = pos[bi]
        bond_ends = bond_starts + bond_vecs

        # --- center everything ---
        a_vec, b_vec, c_vec = cell_mat[0], cell_mat[1], cell_mat[2]
        center = 0.5 * (a_vec + b_vec + c_vec)
        pos_c = pos - center
        bstart_c = bond_starts - center
        bend_c = bond_ends - center

        # Cell outline edges
        o = -center
        cell_corners = [
            o, o + a_vec, o + b_vec, o + c_vec,
            o + a_vec + b_vec, o + a_vec + c_vec, o + b_vec + c_vec,
            o + a_vec + b_vec + c_vec,
        ]
        cell_edge_pairs = [
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        ]
        cell_segs = [(cell_corners[i], cell_corners[j]) for i, j in cell_edge_pairs]

        # --- colormap for crystalline bonds (depth-coloured) ---
        cmap = plt.get_cmap(colormap)

        cryst_mask = bond_is_cryst
        bnd_mask = ~bond_is_cryst
        extent = float(np.max(np.abs(pos_c))) * 1.15

        dpi = 100
        figsize = (width / dpi, height / dpi)

        def _rotate_2d(pts: np.ndarray, theta: float) -> np.ndarray:
            """Rotate x-y columns by theta radians (in-place friendly)."""
            c, s = np.cos(theta), np.sin(theta)
            x_new = pts[:, 0] * c - pts[:, 1] * s
            y_new = pts[:, 0] * s + pts[:, 1] * c
            out = pts.copy()
            out[:, 0] = x_new
            out[:, 1] = y_new
            return out

        def _draw_frame(theta_rad: float) -> "plt.Figure":
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(background)
            fig.patch.set_facecolor(background)

            # Perspective projection (like MATLAB camproj('perspective'))
            try:
                ax.set_proj_type("persp", focal_length=0.25)
            except (TypeError, AttributeError):
                pass  # older matplotlib

            # Rotate bond endpoints in x-y plane (like MATLAB)
            bs_r = _rotate_2d(bstart_c, theta_rad)
            be_r = _rotate_2d(bend_c, theta_rad)

            # --- boundary bonds: very faint ---
            if np.any(bnd_mask):
                segs_b = list(zip(bs_r[bnd_mask], be_r[bnd_mask]))
                lc_b = Line3DCollection(
                    segs_b, linewidths=0.3,
                    colors=(0.0, 0.0, 0.0, 0.05),
                )
                ax.add_collection3d(lc_b)

            # --- crystalline bonds: depth-coloured + depth-width ---
            # Camera at +x (azim=0): larger rotated-x = closer.
            # Linear depth with floor so far bonds remain visible.
            if np.any(cryst_mask):
                segs_cr = list(zip(bs_r[cryst_mask], be_r[cryst_mask]))
                mid_x_rot = 0.5 * (bs_r[cryst_mask, 0] + be_r[cryst_mask, 0])
                norm_depth = (mid_x_rot + extent) / max(2.0 * extent, _EPS)
                norm_depth = np.clip(norm_depth, 0, 1)
                # Colormap: 0.15 at back (faint but visible), 0.95 at front
                cryst_colors = cmap(0.15 + 0.8 * norm_depth)
                # Linewidth: 0.4 at back, 2.0 at front
                cryst_lw = 0.4 + 1.6 * norm_depth
                lc_c = Line3DCollection(
                    segs_cr, linewidths=cryst_lw, colors=cryst_colors,
                )
                ax.add_collection3d(lc_c)

            # --- cell outline ---
            if show_cell:
                cell_segs_r = []
                for s, e in cell_segs:
                    s_r = _rotate_2d(s.reshape(1, 3), theta_rad)[0]
                    e_r = _rotate_2d(e.reshape(1, 3), theta_rad)[0]
                    cell_segs_r.append((s_r, e_r))
                lc_cell = Line3DCollection(
                    cell_segs_r, linewidths=1.5, colors="k", alpha=0.5,
                )
                ax.add_collection3d(lc_cell)

            # --- atoms (tiny dots) ---
            if show_atoms and atom_size > 0:
                pos_r = _rotate_2d(pos_c, theta_rad)
                ax.scatter(
                    pos_r[:, 0], pos_r[:, 1], pos_r[:, 2],
                    s=atom_size, c="k", alpha=0.15,
                    edgecolors="none", depthshade=False,
                )

            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_zlim(-extent, extent)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elevation, azim=0)  # azim fixed; we rotate data
            ax.axis("off")
            fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)
            return fig

        # --- static display or animation ---
        if output is None:
            return _draw_frame(theta_rad=np.deg2rad(45.0))

        # Full 360-degree periodic rotation
        n_frames = int(fps * duration)
        thetas = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

        if show_progress:
            progress = _TextProgressBar(n_frames, label="Rendering", width=28)
        else:
            progress = None

        is_mp4 = str(output).lower().endswith(".mp4")

        if is_mp4:
            # MP4 via ffmpeg subprocess — true 60fps
            import subprocess
            import io

            # Frame dimensions from figure size (no bbox_inches="tight"
            # for the raw pipe — keeps pixel count deterministic).
            fw, fh = width, height

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{fw}x{fh}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "18",
                str(output),
            ]
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            for i, th in enumerate(thetas):
                fig_i = _draw_frame(th)
                buf_i = io.BytesIO()
                fig_i.savefig(buf_i, format="raw", dpi=dpi,
                              facecolor=background)
                plt.close(fig_i)
                proc.stdin.write(buf_i.getvalue())
                if progress is not None:
                    progress.update(i + 1)

            proc.stdin.close()
            proc.wait()
        else:
            # GIF fallback via Pillow
            from PIL import Image
            import io

            frames: list[Image.Image] = []
            for i, th in enumerate(thetas):
                fig = _draw_frame(th)
                buf = io.BytesIO()
                fig.savefig(
                    buf, format="png", dpi=dpi,
                    bbox_inches="tight", pad_inches=0,
                    facecolor=background,
                )
                plt.close(fig)
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                if progress is not None:
                    progress.update(i + 1)

            frame_duration_ms = int(1000 / fps)
            frames[0].save(
                output,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
                loop=0,
                optimize=True,
            )

        if progress is not None:
            progress.update(n_frames)
        return output


