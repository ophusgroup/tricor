"""Interactive 3D structure viewer widget using Three.js via anywidget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import traitlets
from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.neighborlist import neighbor_list

import anywidget

_STATIC_DIR = Path(__file__).parent / "static"


class StructureWidget(anywidget.AnyWidget):
    """Interactive 3D viewer for atomic structures in Jupyter.

    Renders atoms as spheres (coloured by element) and nearest-neighbour
    bonds as cylinders inside the periodic cell outline.  Uses Three.js
    WebGL with OrbitControls for smooth drag-to-rotate / scroll-to-zoom.

    Parameters
    ----------
    atoms
        ASE Atoms object to display.
    shell_target
        If provided, ``bond_cutoff`` defaults to
        ``shell_target.max_pair_outer``.
    atom_scale
        Radius multiplier for atom spheres (in Angstrom).
    bond_cutoff
        Maximum bond length in Angstrom.  If ``None``, derived from
        *shell_target* or defaults to 3.0.
    bond_radius
        Cylinder radius for bonds in Angstrom.
    show_bonds
        Initial bond visibility.
    show_cell
        Initial cell outline visibility.
    slab_x, slab_y, slab_z
        Fractional coordinate ranges ``(min, max)`` for slab clipping.
    grain_ids
        Per-atom grain assignment (``>= 0`` = grain, ``-1`` = amorphous).
        Currently stored for future use.
    """

    _esm = _STATIC_DIR / "structure_viewer.js"
    _css = _STATIC_DIR / "structure_viewer.css"

    # --- atom data ---
    atom_positions = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    atom_colors = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    atom_radii = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    atom_visible = traitlets.List(trait=traitlets.Bool()).tag(sync=True)
    atom_scale = traitlets.Float(0.4).tag(sync=True)
    num_atoms = traitlets.Int(0).tag(sync=True)

    # --- bond data ---
    bond_starts = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    bond_ends = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    bond_colors = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    bond_visible = traitlets.List(trait=traitlets.Bool()).tag(sync=True)
    bond_pair_indices = traitlets.List(trait=traitlets.Int()).tag(sync=True)
    bond_radius = traitlets.Float(0.08).tag(sync=True)
    num_bonds = traitlets.Int(0).tag(sync=True)

    # --- cell outline ---
    cell_vertices = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    cell_edges = traitlets.List(trait=traitlets.Int()).tag(sync=True)

    # --- species / bond-pair info ---
    species_labels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    bond_pair_labels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    bond_pair_visible = traitlets.List(trait=traitlets.Bool()).tag(sync=True)

    # --- controls ---
    bond_cutoff = traitlets.Float(3.0).tag(sync=True)
    bond_cutoff_max = traitlets.Float(6.0).tag(sync=True)
    show_bonds = traitlets.Bool(True).tag(sync=True)
    show_cell = traitlets.Bool(True).tag(sync=True)

    # --- polyhedra (tetrahedra / octahedra / cuboctahedra) ---
    show_polyhedra = traitlets.Bool(False).tag(sync=True)
    polyhedra_kind = traitlets.Unicode("tetrahedra").tag(sync=True)
    polyhedra_center_symbol = traitlets.Unicode("Si").tag(sync=True)
    polyhedra_vertex_symbol = traitlets.Unicode("O").tag(sync=True)
    polyhedra_bond_length = traitlets.Float(1.61).tag(sync=True)
    polyhedra_bond_length_tol = traitlets.Float(0.15).tag(sync=True)
    polyhedra_angle_tol_deg = traitlets.Float(25.0).tag(sync=True)
    polyhedra_scale = traitlets.Float(1.0).tag(sync=True)
    polyhedra_color = traitlets.List(trait=traitlets.Float(),
        default_value=[0.28, 0.62, 0.95]).tag(sync=True)
    polyhedra_opacity = traitlets.Float(0.2).tag(sync=True)
    # Synced polyhedra topology (per-polyhedron faces/edges for cuboctahedra,
    # shared tables for tets/octa)
    polyhedra_n_vertices = traitlets.Int(4).tag(sync=True)
    polyhedra_faces = traitlets.List(
        trait=traitlets.List(trait=traitlets.Int())).tag(sync=True)
    polyhedra_edges = traitlets.List(
        trait=traitlets.List(trait=traitlets.Int())).tag(sync=True)
    polyhedra_per_poly_topology = traitlets.Bool(False).tag(sync=True)
    polyhedra_faces_per_poly = traitlets.List(
        trait=traitlets.List(trait=traitlets.List(trait=traitlets.Int()))
    ).tag(sync=True)
    polyhedra_edges_per_poly = traitlets.List(
        trait=traitlets.List(trait=traitlets.List(trait=traitlets.Int()))
    ).tag(sync=True)
    # Per-polyhedron flattened vertex positions in world space
    # (min-image wrt centre, box-centred, scaled by polyhedra_scale).
    polyhedra_vertex_positions = traitlets.List(
        trait=traitlets.Float()).tag(sync=True)
    num_polyhedra = traitlets.Int(0).tag(sync=True)

    # --- slab selection (fractional 0-1) ---
    slab_x_min = traitlets.Float(0.0).tag(sync=True)
    slab_x_max = traitlets.Float(1.0).tag(sync=True)
    slab_y_min = traitlets.Float(0.0).tag(sync=True)
    slab_y_max = traitlets.Float(1.0).tag(sync=True)
    slab_z_min = traitlets.Float(0.0).tag(sync=True)
    slab_z_max = traitlets.Float(1.0).tag(sync=True)

    # --- display ---
    background_color = traitlets.Unicode("#f7f8f5").tag(sync=True)
    orthographic = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        atoms: Atoms,
        shell_target: Any | None = None,
        *,
        atom_scale: float = 0.2,
        bond_cutoff: float | None = None,
        bond_radius: float = 0.08,
        show_bonds: bool = False,
        show_cell: bool = True,
        orthographic: bool = False,
        slab_x: tuple[float, float] = (0.0, 1.0),
        slab_y: tuple[float, float] = (0.0, 1.0),
        slab_z: tuple[float, float] = (0.0, 1.0),
        grain_ids: np.ndarray | None = None,
        polyhedra: dict | bool | None = None,
    ):
        # Store references
        self._atoms = atoms.copy()
        self._shell_target = shell_target
        self._grain_ids = grain_ids

        cell_mat = np.asarray(atoms.cell.array, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_mat)
        self._cell_mat = cell_mat
        self._cell_inv = cell_inv

        # Bond cutoff
        if bond_cutoff is None:
            if shell_target is not None:
                bond_cutoff = float(shell_target.max_pair_outer)
            else:
                bond_cutoff = 3.0

        # Species info
        unique_z = np.unique(atoms.numbers)
        from ase.data import chemical_symbols
        sp_labels = [chemical_symbols[z] for z in unique_z]

        # Bond pair labels
        pair_labels = []
        for i, si in enumerate(sp_labels):
            for j, sj in enumerate(sp_labels):
                if j >= i:
                    pair_labels.append(f"{si}-{sj}")

        # Atom positions (centered at cell center)
        center = 0.5 * np.sum(cell_mat, axis=0)
        pos = atoms.positions - center

        # Atom colors from jmol
        colors = np.array([jmol_colors[z] for z in atoms.numbers], dtype=np.float64)

        # Atom radii (covalent-ish, scaled)
        from ase.data import covalent_radii
        radii = np.array([covalent_radii[z] for z in atoms.numbers], dtype=np.float64)

        # Cell outline vertices (8 corners, centered)
        o = -center
        a, b, c = cell_mat[0], cell_mat[1], cell_mat[2]
        corners = np.array([
            o, o + a, o + b, o + c,
            o + a + b, o + a + c, o + b + c,
            o + a + b + c,
        ], dtype=np.float64)
        edge_pairs = [0,1, 0,2, 0,3, 1,4, 1,5, 2,4, 2,6, 3,5, 3,6, 4,7, 5,7, 6,7]

        # Store for bond recomputation
        self._pos_centered = pos
        self._center = center
        self._unique_z = unique_z
        self._sp_labels = sp_labels
        self._pair_labels = pair_labels

        # Initialize traitlets before super().__init__
        kwargs = dict(
            atom_positions=pos.ravel().tolist(),
            atom_colors=colors.ravel().tolist(),
            atom_radii=radii.tolist(),
            atom_visible=[True] * len(atoms),
            atom_scale=float(atom_scale),
            num_atoms=len(atoms),
            bond_radius=float(bond_radius),
            bond_cutoff=float(bond_cutoff),
            bond_cutoff_max=float(bond_cutoff * 2.0),
            show_bonds=show_bonds,
            show_cell=show_cell,
            orthographic=orthographic,
            cell_vertices=corners.ravel().tolist(),
            cell_edges=edge_pairs,
            species_labels=sp_labels,
            bond_pair_labels=pair_labels,
            bond_pair_visible=[True] * len(pair_labels),
            slab_x_min=float(slab_x[0]),
            slab_x_max=float(slab_x[1]),
            slab_y_min=float(slab_y[0]),
            slab_y_max=float(slab_y[1]),
            slab_z_min=float(slab_z[0]),
            slab_z_max=float(slab_z[1]),
        )

        # Compute initial bonds
        bond_data = self._compute_bonds(bond_cutoff)
        kwargs.update(bond_data)

        # --- polyhedra: auto-pick kind + settings ---
        poly_cfg = self._resolve_polyhedra_config(
            polyhedra, atoms, shell_target, sp_labels,
        )
        kwargs.update(poly_cfg)

        super().__init__(**kwargs)

        # Compute polyhedra if enabled
        if kwargs.get("show_polyhedra"):
            poly_data = self._compute_polyhedra()
            for k, v in poly_data.items():
                self.set_trait(k, v)

        # Observers
        self.observe(self._on_slab_change, names=[
            "slab_x_min", "slab_x_max", "slab_y_min", "slab_y_max",
            "slab_z_min", "slab_z_max",
        ])
        self.observe(self._on_bond_cutoff_change, names=["bond_cutoff"])
        self.observe(self._on_bond_pair_visible_change, names=["bond_pair_visible"])
        self.observe(self._on_polyhedra_config_change, names=[
            "show_polyhedra", "polyhedra_kind",
            "polyhedra_center_symbol", "polyhedra_vertex_symbol",
            "polyhedra_bond_length", "polyhedra_bond_length_tol",
            "polyhedra_angle_tol_deg", "polyhedra_scale",
        ])

    def _compute_bonds(self, cutoff: float) -> dict[str, Any]:
        """Compute bonds and return traitlet update dict."""
        atoms = self._atoms
        pos = self._pos_centered
        cell_mat = self._cell_mat
        cell_inv = self._cell_inv

        bi, bj, bd = neighbor_list("ijd", atoms, cutoff)
        mask = bi < bj
        bi, bj = bi[mask], bj[mask]

        # Minimum-image bond vectors
        raw_delta = atoms.positions[bj] - atoms.positions[bi]
        frac = raw_delta @ cell_inv
        frac -= np.rint(frac)
        bond_vecs = frac @ cell_mat

        starts = pos[bi]
        ends = starts + bond_vecs

        # Bond colors: average of endpoint atom colors
        colors_arr = np.array([jmol_colors[z] for z in atoms.numbers], dtype=np.float64)
        bond_colors = 0.5 * (colors_arr[bi] + colors_arr[bj])

        # Bond pair index (for toggling by species pair)
        unique_z = self._unique_z
        z_to_idx = {int(z): i for i, z in enumerate(unique_z)}
        n_sp = len(unique_z)
        pair_idx_list = []
        for k in range(len(bi)):
            si = z_to_idx[int(atoms.numbers[bi[k]])]
            sj = z_to_idx[int(atoms.numbers[bj[k]])]
            if si > sj:
                si, sj = sj, si
            # Linear index into upper triangle
            pidx = si * n_sp - si * (si - 1) // 2 + (sj - si)
            pair_idx_list.append(pidx)

        # Store for slab filtering
        self._bond_bi = bi
        self._bond_bj = bj
        self._bond_pair_indices_arr = np.array(pair_idx_list, dtype=np.intp)

        return dict(
            bond_starts=starts.ravel().tolist(),
            bond_ends=ends.ravel().tolist(),
            bond_colors=bond_colors.ravel().tolist(),
            bond_visible=[True] * len(bi),
            bond_pair_indices=pair_idx_list,
            num_bonds=len(bi),
        )

    def _on_slab_change(self, change: dict) -> None:
        """Recompute visibility when slab ranges change."""
        frac = self._atoms.get_scaled_positions()
        vis = (
            (frac[:, 0] >= self.slab_x_min) & (frac[:, 0] <= self.slab_x_max)
            & (frac[:, 1] >= self.slab_y_min) & (frac[:, 1] <= self.slab_y_max)
            & (frac[:, 2] >= self.slab_z_min) & (frac[:, 2] <= self.slab_z_max)
        )
        self.atom_visible = vis.tolist()

        # Bonds visible if both endpoints visible
        if hasattr(self, "_bond_bi") and len(self._bond_bi) > 0:
            bvis = vis[self._bond_bi] & vis[self._bond_bj]
            # Also respect pair toggles
            pair_vis = np.array(self.bond_pair_visible, dtype=bool)
            for k in range(len(self._bond_bi)):
                if not pair_vis[self._bond_pair_indices_arr[k]]:
                    bvis[k] = False
            self.bond_visible = bvis.tolist()

    def _on_bond_cutoff_change(self, change: dict) -> None:
        """Recompute bonds when cutoff changes."""
        bond_data = self._compute_bonds(self.bond_cutoff)
        self.set_trait("bond_starts", bond_data["bond_starts"])
        self.set_trait("bond_ends", bond_data["bond_ends"])
        self.set_trait("bond_colors", bond_data["bond_colors"])
        self.set_trait("bond_visible", bond_data["bond_visible"])
        self.set_trait("bond_pair_indices", bond_data["bond_pair_indices"])
        self.set_trait("num_bonds", bond_data["num_bonds"])
        # Re-apply slab filter
        self._on_slab_change({})

    def _on_bond_pair_visible_change(self, change: dict) -> None:
        """Toggle bond visibility by species pair."""
        if not hasattr(self, "_bond_bi") or len(self._bond_bi) == 0:
            return
        pair_vis = np.array(self.bond_pair_visible, dtype=bool)
        atom_vis = np.array(self.atom_visible, dtype=bool)
        bvis = atom_vis[self._bond_bi] & atom_vis[self._bond_bj]
        for k in range(len(self._bond_bi)):
            if not pair_vis[self._bond_pair_indices_arr[k]]:
                bvis[k] = False
        self.bond_visible = bvis.tolist()

    # ---------------------------------------------------------------
    # Polyhedra (tetrahedra / octahedra / cuboctahedra)
    # ---------------------------------------------------------------

    @staticmethod
    def _resolve_polyhedra_config(
        user_cfg: "dict | bool | None",
        atoms: Atoms,
        shell_target: Any | None,
        sp_labels: list[str],
    ) -> dict[str, Any]:
        """Decide which polyhedron to draw + defaults for each setting.

        User-side ``polyhedra`` can be:

        * ``None`` (default): auto-pick based on species - Si/C single
          element -> tetrahedra at half-scale; binary with Si or Ti ->
          tets or octa around the cation; single metal -> cuboctahedra
          at half-scale.  Leaves ``show_polyhedra=False`` by default so
          the classic bond view is shown unless the user opts in.
        * ``True`` / ``False``: same as ``None`` for the detection
          logic but flips ``show_polyhedra`` accordingly.
        * dict: explicit override, merged on top of the auto-detected
          config.
        """
        cfg: dict[str, Any] = dict(show_polyhedra=False)
        auto_kind = None
        auto_center = None
        auto_vertex = None
        auto_bl = None
        auto_scale = 1.0
        auto_color = [0.28, 0.62, 0.95]

        symbols = set(sp_labels)
        # Auto-pick based on species
        if symbols == {"Si"}:
            auto_kind = "tetrahedra"
            auto_center, auto_vertex = "Si", "Si"
            auto_bl = 2.352
            auto_scale = 0.5
            auto_color = [0.35, 0.45, 0.95]
        elif symbols == {"C"}:
            auto_kind = "tetrahedra"
            auto_center, auto_vertex = "C", "C"
            auto_bl = 1.54
            auto_scale = 0.5
            auto_color = [0.35, 0.35, 0.35]
        elif symbols == {"Cu"}:
            auto_kind = "cuboctahedra"
            auto_center, auto_vertex = "Cu", "Cu"
            auto_bl = 2.556
            auto_scale = 0.5
            auto_color = [0.85, 0.45, 0.20]
        elif {"Si", "O"}.issubset(symbols):
            auto_kind = "tetrahedra"
            auto_center, auto_vertex = "Si", "O"
            auto_bl = 1.61
            auto_scale = 1.0
            auto_color = [0.28, 0.62, 0.95]
        elif {"Ti", "O"}.issubset(symbols):
            auto_kind = "octahedra"
            auto_center, auto_vertex = "Ti", "O"
            auto_bl = 1.956
            auto_scale = 1.0
            auto_color = [0.95, 0.55, 0.25]

        # Try to refine bond_length from shell_target pair_peak
        if shell_target is not None and auto_center is not None:
            try:
                import numpy as _np
                sp = _np.asarray(shell_target.species, dtype=_np.int64)
                ci = int(_np.where(sp == atomic_numbers[auto_center])[0][0])
                vi = int(_np.where(sp == atomic_numbers[auto_vertex])[0][0])
                val = float(_np.asarray(shell_target.pair_peak, dtype=_np.float64)[ci, vi])
                if val > 0:
                    auto_bl = val
            except (IndexError, KeyError, AttributeError):
                pass

        if auto_kind is not None:
            cfg["polyhedra_kind"] = auto_kind
            cfg["polyhedra_center_symbol"] = auto_center
            cfg["polyhedra_vertex_symbol"] = auto_vertex
            cfg["polyhedra_bond_length"] = float(auto_bl)
            cfg["polyhedra_scale"] = float(auto_scale)
            cfg["polyhedra_color"] = auto_color

        # Apply user overrides
        if user_cfg is True:
            cfg["show_polyhedra"] = auto_kind is not None
        elif user_cfg is False:
            cfg["show_polyhedra"] = False
        elif isinstance(user_cfg, dict):
            cfg["show_polyhedra"] = True
            for k, v in user_cfg.items():
                tr_name = (
                    k if k.startswith("polyhedra_") or k == "show_polyhedra"
                    else f"polyhedra_{k}"
                )
                cfg[tr_name] = v
        return cfg

    def _compute_polyhedra(self) -> dict[str, Any]:
        """Run the appropriate polyhedron detector and return traitlets
        update dict with vertex positions, face/edge topology, count."""
        from ._plotting import (
            _detect_tetrahedra, _detect_octahedra, _detect_cuboctahedra,
            _polyhedra_vertex_coords,
            _TET_FACES, _TET_EDGES, _OCT_FACES, _OCT_EDGES,
        )
        kind = self.polyhedra_kind
        args = dict(
            center_symbol=self.polyhedra_center_symbol,
            vertex_symbol=self.polyhedra_vertex_symbol,
            bond_length=float(self.polyhedra_bond_length),
            bond_length_tol=float(self.polyhedra_bond_length_tol),
            angle_tol_deg=float(self.polyhedra_angle_tol_deg),
        )
        per_poly = False
        if kind == "octahedra":
            detector = _detect_octahedra
            args["ideal_angle_deg"] = 90.0
            faces, edges = _OCT_FACES, _OCT_EDGES
            n_vertices = 6
        elif kind == "cuboctahedra":
            detector = _detect_cuboctahedra
            faces, edges = [], []
            n_vertices = 12
            per_poly = True
        else:  # tetrahedra default
            detector = _detect_tetrahedra
            args["ideal_angle_deg"] = 109.47
            faces, edges = _TET_FACES, _TET_EDGES
            n_vertices = 4

        polys = detector(self._atoms, **args)
        verts = _polyhedra_vertex_coords(
            polys, self._atoms.positions, self._atoms.cell.array,
            scale=float(self.polyhedra_scale),
        )
        verts = [round(v, 3) for v in verts]

        out: dict[str, Any] = dict(
            polyhedra_vertex_positions=verts,
            num_polyhedra=len(polys),
            polyhedra_n_vertices=n_vertices,
            polyhedra_per_poly_topology=per_poly,
            polyhedra_faces=[list(f) for f in faces],
            polyhedra_edges=[list(e) for e in edges],
        )
        if per_poly:
            out["polyhedra_faces_per_poly"] = [
                [list(f) for f in p["faces"]] for p in polys
            ]
            out["polyhedra_edges_per_poly"] = [
                [list(e) for e in p["edges"]] for p in polys
            ]
        else:
            out["polyhedra_faces_per_poly"] = []
            out["polyhedra_edges_per_poly"] = []
        return out

    def _on_polyhedra_config_change(self, change: dict) -> None:
        """Recompute polyhedra whenever any setting changes."""
        if not self.show_polyhedra:
            # When turning off, we still emit an empty payload so the
            # viewer can clear the mesh.
            self.set_trait("polyhedra_vertex_positions", [])
            self.set_trait("num_polyhedra", 0)
            return
        data = self._compute_polyhedra()
        # Use hold_trait_notifications to batch updates.
        with self.hold_trait_notifications():
            for k, v in data.items():
                self.set_trait(k, v)
