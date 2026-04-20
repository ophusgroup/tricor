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
_G2_HTML_TEMPLATE = (_STATIC_DIR / "g2_viewer.html").read_text()
_OVERVIEW_HTML_TEMPLATE = (_STATIC_DIR / "overview_viewer.html").read_text()


def _detect_tetrahedra(
    atoms,
    *,
    center_symbol: str = "Si",
    vertex_symbol: str = "O",
    bond_length: float | None = None,
    bond_length_tol: float = 0.15,
    ideal_angle_deg: float = 109.47,
    angle_tol_deg: float = 25.0,
) -> list[dict]:
    """Find center atoms whose 4 nearest vertex atoms form a tetrahedron.

    Returns a list of dicts with keys ``center`` (int) and ``vertices``
    (list of 4 ints), in order of increasing center-to-vertex distance.
    Only tetrahedra whose 4 bond lengths are within
    ``bond_length_tol`` of ``bond_length`` AND whose 6 pairwise
    (vertex)-(center)-(vertex) angles are within ``angle_tol_deg`` of
    ``ideal_angle_deg`` are returned.

    When ``bond_length`` is ``None`` the median of all center-vertex
    distances inside 3.5 Å is used as an estimate.
    """
    from ase.data import atomic_numbers

    Zc = atomic_numbers[center_symbol]
    Zv = atomic_numbers[vertex_symbol]
    numbers = np.asarray(atoms.numbers)

    if bond_length is None:
        # Use the 10th percentile of center-vertex distances so the
        # "bond length" sits near the first-neighbour peak even in
        # disordered samples where many longer pairs exist.
        bi, bj, bd = neighbor_list("ijd", atoms, 3.5)
        mask = (numbers[bi] == Zc) & (numbers[bj] == Zv)
        if not np.any(mask):
            return []
        bond_length = float(np.percentile(bd[mask], 10))

    cutoff = float(bond_length) * (1.0 + float(bond_length_tol)) * 1.05
    bi_all, bj_all, bd_all, bD_all = neighbor_list("ijdD", atoms, float(cutoff))

    keep = (
        (numbers[bi_all] == Zc)
        & (numbers[bj_all] == Zv)
        & (bd_all >= bond_length * (1.0 - bond_length_tol))
        & (bd_all <= bond_length * (1.0 + bond_length_tol))
    )
    if not np.any(keep):
        return []
    bi = bi_all[keep]
    bj = bj_all[keep]
    bd = bd_all[keep]
    bD = bD_all[keep]

    order = np.lexsort((bd, bi))
    bi_s = bi[order]
    bj_s = bj[order]
    bD_s = bD[order]

    ideal_rad = float(np.deg2rad(ideal_angle_deg))
    tol_rad = float(np.deg2rad(angle_tol_deg))

    unique_i, start_idx = np.unique(bi_s, return_index=True)
    end_idx = np.concatenate([start_idx[1:], [bi_s.size]])

    tetrahedra: list[dict] = []
    for u, s, e in zip(unique_i, start_idx, end_idx):
        if int(e - s) < 4:
            continue
        js = bj_s[s : s + 4]
        vs = bD_s[s : s + 4]
        norms = np.linalg.norm(vs, axis=1)
        if np.any(norms < 1e-6):
            continue
        unit = vs / norms[:, None]
        cos_ab = np.clip(unit @ unit.T, -1.0, 1.0)
        angles = np.arccos(cos_ab)
        triu = np.triu_indices(4, k=1)
        if float(np.max(np.abs(angles[triu] - ideal_rad))) <= tol_rad:
            tetrahedra.append(
                {
                    "center": int(u),
                    "vertices": [int(j) for j in js],
                }
            )
    return tetrahedra


def _tetrahedra_vertex_coords(
    tetrahedra: list[dict],
    positions: np.ndarray,
    cell_matrix: np.ndarray,
    scale: float = 1.0,
) -> list[float]:
    """Flat list of vertex positions (min-image wrt centre, box-centred).

    When ``scale < 1`` the polyhedron is shrunk toward its centre -
    ``scale=0.5`` puts the rendered vertices exactly at the midpoints
    of the centre-vertex bonds, which is what we want for single-
    element polyhedra (Si tets, Cu cuboctahedra) where the vertex
    atoms are the same species as the centre.
    """
    if not tetrahedra:
        return []
    cell = np.asarray(cell_matrix, dtype=np.float64)
    cell_inv = np.linalg.inv(cell)
    centre = 0.5 * cell.sum(axis=0)
    out = []
    for t in tetrahedra:
        c_pos = positions[t["center"]]
        for v_idx in t["vertices"]:
            v_pos = positions[v_idx]
            disp = v_pos - c_pos
            frac = disp @ cell_inv
            frac -= np.round(frac)
            disp_w = frac @ cell
            adj = c_pos + float(scale) * disp_w - centre
            out.extend(float(x) for x in adj)
    return out


# _polyhedra_vertex_coords is the generalised name; kept as an alias so
# the rest of the module (and external callers) can use either form.
_polyhedra_vertex_coords = _tetrahedra_vertex_coords


# Face / edge topology tables used by the viewers to build translucent
# polyhedra meshes.  Indices refer into the per-polyhedron vertex list
# produced by ``_detect_tetrahedra`` / ``_detect_octahedra``.
_TET_FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
_TET_EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
# Octahedra: ``_detect_octahedra`` orders vertices so (0,1), (2,3),
# (4,5) are antipodal; the 8 faces are all triples that take one vertex
# from each antipodal pair, the 12 edges are every pair *except* the
# three antipodal ones.
_OCT_FACES = [
    [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
    [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
]
_OCT_EDGES = [
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 4], [2, 5], [3, 4], [3, 5],
]


def _resolve_polyhedra_cfg(
    tetrahedra: "dict | None",
    octahedra: "dict | None",
    cuboctahedra: "dict | None" = None,
) -> "dict | None":
    """Normalise user-provided polyhedra dicts into a single config.

    At most one of ``tetrahedra`` / ``octahedra`` / ``cuboctahedra`` may
    be provided.  Returns ``None`` when none are given.  The returned
    dict carries:

    - ``center_symbol``, ``vertex_symbol``, ``bond_length``,
      ``bond_length_tol``, ``ideal_angle_deg``, ``angle_tol_deg`` -
      detector-level config
    - ``scale`` - render-time polyhedron shrinkage (1.0 = vertices at
      actual atom positions, 0.5 = vertices at bond midpoints; handy for
      single-element polyhedra where corners otherwise sit directly on
      neighbouring atoms)
    - ``n_vertices`` (4 / 6 / 12)
    - ``faces``, ``edges`` - shared face / edge tables (``None`` when
      topology is per-polyhedron, e.g. cuboctahedra, in which case the
      detector embeds its own tables)
    - ``detector``, ``per_polyhedron_topology`` (bool)
    """
    provided = [x for x in (tetrahedra, octahedra, cuboctahedra) if x is not None]
    if len(provided) > 1:
        raise ValueError(
            "Pass at most one of 'tetrahedra', 'octahedra', 'cuboctahedra'.",
        )
    per_poly = False
    if tetrahedra is not None:
        cfg = tetrahedra
        n_vertices = 4
        faces, edges = _TET_FACES, _TET_EDGES
        detector = _detect_tetrahedra
        ideal_default = 109.47
        angle_tol_default = 25.0
        scale_default = 1.0
    elif octahedra is not None:
        cfg = octahedra
        n_vertices = 6
        faces, edges = _OCT_FACES, _OCT_EDGES
        detector = _detect_octahedra
        ideal_default = 90.0
        angle_tol_default = 18.0
        scale_default = 1.0
    elif cuboctahedra is not None:
        cfg = cuboctahedra
        n_vertices = 12
        faces, edges = None, None  # per-polyhedron (embedded in each dict)
        detector = _detect_cuboctahedra
        ideal_default = 60.0   # nominal nearest-neighbour angle
        angle_tol_default = 18.0
        scale_default = 0.5     # default to half-size for single-element
        per_poly = True
    else:
        return None
    return {
        "center_symbol": cfg.get("center_symbol", "Si"),
        "vertex_symbol": cfg.get("vertex_symbol", "O"),
        "bond_length": cfg.get("bond_length"),
        "bond_length_tol": float(cfg.get("bond_length_tol", 0.15)),
        "ideal_angle_deg": float(cfg.get("ideal_angle_deg", ideal_default)),
        "angle_tol_deg": float(cfg.get("angle_tol_deg", angle_tol_default)),
        "scale": float(cfg.get("scale", scale_default)),
        "n_vertices": n_vertices,
        "faces": faces,
        "edges": edges,
        "detector": detector,
        "per_polyhedron_topology": per_poly,
    }


def _detect_octahedra(
    atoms,
    *,
    center_symbol: str = "Ti",
    vertex_symbol: str = "O",
    bond_length: float | None = None,
    bond_length_tol: float = 0.18,
    ideal_angle_deg: float = 90.0,
    angle_tol_deg: float = 18.0,
) -> list[dict]:
    """Find center atoms whose 6 nearest vertex atoms form an octahedron.

    Returns a list of dicts ``{center: int, vertices: list[int]}``.  The
    six vertex indices are written so pairs (0, 1), (2, 3), (4, 5) are
    antipodal - i.e. the octahedron is oriented as (±x, ±y, ±z) about
    the center, which the viewer uses to build the 8 triangular faces
    ``[[0,2,4],[0,2,5],[0,3,4],[0,3,5],[1,2,4],[1,2,5],[1,3,4],[1,3,5]]``
    and 12 edges (every pair except the 3 antipodal pairs).

    Acceptance criteria for a candidate centre:

    * Exactly 6 vertex neighbours sit within ``bond_length * (1 \u00b1 tol)``.
    * Among the 15 pairwise centre-vertex unit-vector angles, 3 are
      within ``angle_tol_deg`` of 180\u00b0 (the antipodal pairs) and the
      remaining 12 are within ``angle_tol_deg`` of 90\u00b0.

    When ``bond_length`` is ``None``, the 10th percentile of all
    centre-vertex distances inside 3.5 \u00c5 is used as an estimate.
    """
    from ase.data import atomic_numbers

    Zc = atomic_numbers[center_symbol]
    Zv = atomic_numbers[vertex_symbol]
    numbers = np.asarray(atoms.numbers)

    if bond_length is None:
        bi, bj, bd = neighbor_list("ijd", atoms, 3.5)
        mask = (numbers[bi] == Zc) & (numbers[bj] == Zv)
        if not np.any(mask):
            return []
        bond_length = float(np.percentile(bd[mask], 10))

    cutoff = float(bond_length) * (1.0 + float(bond_length_tol)) * 1.05
    bi_all, bj_all, bd_all, bD_all = neighbor_list("ijdD", atoms, float(cutoff))

    keep = (
        (numbers[bi_all] == Zc)
        & (numbers[bj_all] == Zv)
        & (bd_all >= bond_length * (1.0 - bond_length_tol))
        & (bd_all <= bond_length * (1.0 + bond_length_tol))
    )
    if not np.any(keep):
        return []
    bi = bi_all[keep]
    bj = bj_all[keep]
    bd = bd_all[keep]
    bD = bD_all[keep]

    order = np.lexsort((bd, bi))
    bi_s = bi[order]
    bj_s = bj[order]
    bD_s = bD[order]

    ideal_rad_90 = float(np.deg2rad(ideal_angle_deg))
    ideal_rad_180 = float(np.deg2rad(180.0))
    tol_rad = float(np.deg2rad(angle_tol_deg))

    unique_i, start_idx = np.unique(bi_s, return_index=True)
    end_idx = np.concatenate([start_idx[1:], [bi_s.size]])

    octahedra: list[dict] = []
    for u, s, e in zip(unique_i, start_idx, end_idx):
        if int(e - s) < 6:
            continue
        js = bj_s[s : s + 6]
        vs = bD_s[s : s + 6]
        norms = np.linalg.norm(vs, axis=1)
        if np.any(norms < 1e-6):
            continue
        unit = vs / norms[:, None]
        cos_ab = np.clip(unit @ unit.T, -1.0, 1.0)
        angles = np.arccos(cos_ab)
        triu_i, triu_j = np.triu_indices(6, k=1)
        pair_angles = angles[triu_i, triu_j]

        # Identify the 3 antipodal pairs (angle ~ 180).
        anti_mask = np.abs(pair_angles - ideal_rad_180) <= tol_rad
        if int(np.sum(anti_mask)) != 3:
            continue

        anti_pairs = list(zip(triu_i[anti_mask].tolist(), triu_j[anti_mask].tolist()))
        # Every vertex must appear in exactly one antipodal pair.
        anti_of = [-1] * 6
        ok = True
        for a, b in anti_pairs:
            if anti_of[a] != -1 or anti_of[b] != -1:
                ok = False
                break
            anti_of[a] = b
            anti_of[b] = a
        if not ok or -1 in anti_of:
            continue

        # Remaining 12 angles must all be ~ 90.
        rem_mask = ~anti_mask
        if float(np.max(np.abs(pair_angles[rem_mask] - ideal_rad_90))) > tol_rad:
            continue

        # Re-order so (0,1), (2,3), (4,5) are the antipodal pairs.
        order_local: list[int] = []
        visited = [False] * 6
        for v in range(6):
            if visited[v]:
                continue
            w = anti_of[v]
            order_local.extend([v, w])
            visited[v] = True
            visited[w] = True

        verts = [int(js[k]) for k in order_local]
        octahedra.append({"center": int(u), "vertices": verts})
    return octahedra


def _detect_cuboctahedra(
    atoms,
    *,
    center_symbol: str = "Cu",
    vertex_symbol: str = "Cu",
    bond_length: float | None = None,
    bond_length_tol: float = 0.12,
    ideal_angle_deg: float | None = None,   # unused (API parity)
    angle_tol_deg: float = 22.0,
    distance_tol: float = 0.10,
) -> list[dict]:
    """Find centers with 12 nearest vertex neighbours forming a
    close-packed (FCC) cuboctahedron.

    Acceptance:

    * Exactly 12 vertex neighbours within ``bond_length * (1 \u00b1 tol)``.
    * Max/min of the 12 centre-vertex distances must be within
      ``distance_tol`` of the mean (catches 13th-nearest-neighbour
      interlopers).
    * All 66 pairwise centre-vertex unit-vector angles must lie within
      ``angle_tol_deg`` of one of the four canonical FCC values
      {60\u00b0, 90\u00b0, 120\u00b0, 180\u00b0}.  This is what distinguishes a real
      cuboctahedron from a disordered close-packed shell whose atoms
      happen to fall within the radial tolerance - without this test
      the detector counts amorphous clusters as cuboctahedra because
      they too have ~12 nearest neighbours.

    Each returned dict carries its own ``faces`` (20 triangles, two per
    square face plus the eight corner triangles) and ``edges`` (24
    cuboctahedron edges, square-face diagonals filtered out by length).
    """
    from ase.data import atomic_numbers
    from scipy.spatial import ConvexHull

    Zc = atomic_numbers[center_symbol]
    Zv = atomic_numbers[vertex_symbol]
    numbers = np.asarray(atoms.numbers)

    if bond_length is None:
        bi0, bj0, bd0 = neighbor_list("ijd", atoms, 4.0)
        mask0 = (numbers[bi0] == Zc) & (numbers[bj0] == Zv)
        if not np.any(mask0):
            return []
        bond_length = float(np.percentile(bd0[mask0], 10))

    cutoff = float(bond_length) * (1.0 + float(bond_length_tol)) * 1.05
    bi_all, bj_all, bd_all, bD_all = neighbor_list("ijdD", atoms, float(cutoff))
    keep = (
        (numbers[bi_all] == Zc)
        & (numbers[bj_all] == Zv)
        & (bd_all >= bond_length * (1.0 - bond_length_tol))
        & (bd_all <= bond_length * (1.0 + bond_length_tol))
    )
    if not np.any(keep):
        return []
    bi = bi_all[keep]
    bj = bj_all[keep]
    bd = bd_all[keep]
    bD = bD_all[keep]

    order = np.lexsort((bd, bi))
    bi_s = bi[order]
    bj_s = bj[order]
    bd_s = bd[order]
    bD_s = bD[order]

    unique_i, start_idx = np.unique(bi_s, return_index=True)
    end_idx = np.concatenate([start_idx[1:], [bi_s.size]])

    cubocta: list[dict] = []
    for u, s, e in zip(unique_i, start_idx, end_idx):
        n = int(e - s)
        if n < 12:
            continue
        js = bj_s[s : s + 12]
        vs = bD_s[s : s + 12]
        ds = bd_s[s : s + 12]
        norms = np.linalg.norm(vs, axis=1)
        if np.any(norms < 1e-6):
            continue
        mean_d = float(np.mean(ds))
        if (ds.max() - ds.min()) / max(mean_d, 1e-6) > 2.0 * distance_tol:
            continue

        unit = vs / norms[:, None]
        # Angular spectrum check: every pairwise angle must be close
        # to one of {60, 90, 120, 180} deg.  Amorphous close-packed
        # clusters fail this test; real cuboctahedra pass.
        cos_pairs = np.clip(unit @ unit.T, -1.0, 1.0)
        triu_i, triu_j = np.triu_indices(12, k=1)
        pair_angles_deg = np.rad2deg(np.arccos(cos_pairs[triu_i, triu_j]))
        target_angles = np.array([60.0, 90.0, 120.0, 180.0])
        # distance (in degrees) to the nearest target angle, per pair
        dev = np.abs(pair_angles_deg[:, None] - target_angles[None, :]).min(axis=1)
        if float(dev.max()) > angle_tol_deg:
            continue
        # Convex-hull triangulation of the 12 unit directions gives 20
        # triangles (8 corner tris + 12 split-square tris) on an ideal
        # cuboctahedron.  Works for slightly-distorted shells too.
        try:
            hull = ConvexHull(unit)
        except Exception:
            continue
        simplices = hull.simplices.tolist()

        # Build edge set from simplices, filter out square-face
        # diagonals by length (diagonals ~ sqrt(2) vs real edges ~ 1).
        edge_set: dict[tuple[int, int], float] = {}
        for tri in simplices:
            for a, b in [
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[0], tri[2]),
            ]:
                key = (min(a, b), max(a, b))
                if key in edge_set:
                    continue
                edge_set[key] = float(np.linalg.norm(unit[a] - unit[b]))
        if not edge_set:
            continue
        min_edge = min(edge_set.values())
        edges = [
            [int(a), int(b)]
            for (a, b), length in edge_set.items()
            if length <= min_edge * 1.25
        ]

        cubocta.append({
            "center": int(u),
            "vertices": [int(j) for j in js],
            "faces": [[int(x) for x in tri] for tri in simplices],
            "edges": edges,
        })
    return cubocta


def export_g2_compare_html(
    cells_and_labels,
    output_path: "str | None" = None,
    *,
    r_max: float = 10.0,
    r_step: float = 0.05,
    background_color: str = "#f7f8f5",
    title: str = "",
    show_progress: bool = False,
) -> str:
    """Export a g(r) overlay viewer comparing multiple supercells.

    Every cell must share the same set of species (same reference
    material).  One g(r) curve is drawn per cell for the currently
    selected species pair; the dropdown in the viewer switches which
    pair is shown and a legend identifies each cell by its label.

    Parameters
    ----------
    cells_and_labels
        Accepts any of:

        * ``dict[str, Supercell]`` - keys become legend labels
        * ``list[tuple[Supercell, str]]``
        * ``list[Supercell]`` - each ``cell.label`` is used
    output_path
        Path to write the HTML file.  When ``None`` the HTML string is
        returned instead (for :func:`IPython.display.HTML` / inline
        display); see :func:`plot_g2_compare` for a ready-made Jupyter
        wrapper.
    r_max, r_step
        Radial grid for the measurements.  Finer ``r_step`` gives
        smoother curves; 0.05 A is a good default.
    background_color, title
        Cosmetic.
    show_progress
        Forwarded to each :meth:`Supercell.measure_g3` call (used under
        the hood; ``phi_num_bins`` is set low for speed).

    Returns
    -------
    str
        Resolved output path when ``output_path`` was provided,
        otherwise the HTML source string.
    """
    import json

    from .g3 import G3Distribution
    from ase.data import chemical_symbols

    # Normalise input into [(cell, label), ...]
    pairs: list[tuple] = []
    if isinstance(cells_and_labels, dict):
        for label, cell in cells_and_labels.items():
            pairs.append((cell, str(label)))
    else:
        for item in cells_and_labels:
            if isinstance(item, tuple) and len(item) == 2:
                cell, label = item
                pairs.append((cell, str(label)))
            else:
                pairs.append((item, str(getattr(item, "label", "cell"))))
    if not pairs:
        raise ValueError("cells_and_labels is empty.")

    # Species from the first cell define the pair grid.  Verify that
    # every subsequent cell uses the same species set so the curves
    # line up.
    def _species_of(cell) -> np.ndarray:
        return np.unique(np.asarray(cell.atoms.numbers, dtype=np.int64))

    ref_species = _species_of(pairs[0][0])
    for cell, lab in pairs[1:]:
        if not np.array_equal(_species_of(cell), ref_species):
            raise ValueError(
                f"Cell '{lab}' has species {_species_of(cell)} which "
                f"differs from the first cell's {ref_species}.  "
                "export_g2_compare_html requires all supercells to share "
                "the same species."
            )

    sp_labels = [chemical_symbols[int(z)] for z in ref_species]
    num_species = len(ref_species)

    # Measure every cell on the same grid.
    r_edges_global = None
    r_centres_global = None

    def _norm_profile(y: np.ndarray, r_arr: np.ndarray) -> np.ndarray:
        y = y / np.maximum(r_arr * r_arr, _EPS)
        tail_start = 0.7 * float(r_arr[-1])
        tail_mask = r_arr >= tail_start
        if not np.any(tail_mask):
            tail_mask = np.zeros_like(r_arr, dtype=bool)
            tail_mask[-max(1, r_arr.size // 4):] = True
        tail = y[tail_mask]
        finite = tail[np.isfinite(tail)]
        scale = float(np.mean(finite)) if finite.size else 1.0
        if scale <= _EPS:
            scale = 1.0
        return (y / scale).astype(np.float32)

    series: list[dict] = []
    pair_labels: list[str] = []
    # Build pair labels from the first cell's species ordering.
    for ci in range(num_species):
        for vi in range(ci, num_species):
            pair_labels.append(f"{sp_labels[ci]}-{sp_labels[vi]}")

    for cell, lab in pairs:
        dist = G3Distribution(cell.atoms, label=f"{lab}-g2-compare")
        dist.measure_g3(
            r_max=r_max, r_step=r_step, phi_num_bins=12,
            show_progress=show_progress,
        )
        if dist.g2 is None:
            raise ValueError(f"Distribution measurement for '{lab}' has no g2.")
        r = np.asarray(dist.r, dtype=np.float64)
        if r_centres_global is None:
            r_centres_global = r.tolist()
            r_edges_global = np.asarray(dist.bin_edges, dtype=np.float64).tolist()
        g2 = np.asarray(dist.g2, dtype=np.float64)
        profiles: list[list[float]] = []
        for ci in range(num_species):
            for vi in range(ci, num_species):
                prof = _norm_profile(g2[ci, vi], r)
                profiles.append(prof.tolist())
        series.append({"label": lab, "profiles": profiles})

    # Per-pair peak markers pulled from the first cell's shell_target
    # (if present).
    pair_peaks: list[float] = [0.0] * len(pair_labels)
    shell_target = getattr(pairs[0][0], "_shell_target", None)
    if shell_target is not None:
        st_species = np.asarray(shell_target.species, dtype=np.int64)
        pair_peak_mat = np.asarray(shell_target.pair_peak, dtype=np.float64)
        # Map dist species -> shell_target species index
        idx_map: dict[int, int] = {}
        for ci_dist, z in enumerate(ref_species):
            m = np.where(st_species == int(z))[0]
            if m.size:
                idx_map[ci_dist] = int(m[0])
        out_idx = 0
        for ci in range(num_species):
            for vi in range(ci, num_species):
                ci_st = idx_map.get(ci)
                vi_st = idx_map.get(vi)
                if ci_st is not None and vi_st is not None:
                    val = float(pair_peak_mat[ci_st, vi_st])
                    if np.isfinite(val) and val > 0:
                        pair_peaks[out_idx] = val
                out_idx += 1

    # Default pair: smallest positive peak (typically the short bond).
    default_pair = 0
    positive = [(i, p) for i, p in enumerate(pair_peaks) if p > 0]
    if positive:
        default_pair = min(positive, key=lambda x: x[1])[0]

    data = {
        "num_r": len(r_centres_global),
        "num_pairs": len(pair_labels),
        "r_centers": r_centres_global,
        "r_edges": r_edges_global,
        "pair_labels": pair_labels,
        "pair_peaks": pair_peaks,
        "series": series,
        "default_pair": int(default_pair),
        "background_color": background_color,
        "title": title,
    }

    html = _G2_HTML_TEMPLATE.replace(
        "__TRICOR_DATA_PLACEHOLDER__",
        json.dumps(data),
    )
    if output_path is None:
        return html
    output_path = str(output_path)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def plot_g2_compare(
    cells_and_labels,
    *,
    r_max: float = 10.0,
    r_step: float = 0.05,
    title: str = "",
    height: int = 480,
    show_progress: bool = False,
):
    """Inline Jupyter display of the g(r) overlay-compare viewer.

    Convenience wrapper around :func:`export_g2_compare_html` that
    packages the HTML as an :class:`IPython.display.HTML` object so you
    can do ``tc.plot_g2_compare(cells)`` in a notebook cell.  See that
    function's docstring for the accepted input shapes.
    """
    from IPython.display import HTML
    html = export_g2_compare_html(
        cells_and_labels, None,
        r_max=r_max, r_step=r_step,
        title=title, show_progress=show_progress,
    )
    import html as _html
    escaped = _html.escape(html, quote=True)
    return HTML(
        f'<div class="tricor-g2-compare-wrapper" style="width:100%">'
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%; height:{int(height)}px; '
        f'border:1px solid rgba(0,0,0,0.1); border-radius:6px;"></iframe>'
        f'</div>'
    )


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
    tetrahedra: dict | None = None,
    tetrahedra_color=(0.25, 0.65, 0.95),
    tetrahedra_opacity: float = 0.35,
    octahedra: dict | None = None,
    octahedra_color=(0.95, 0.55, 0.25),
    octahedra_opacity: float = 0.4,
    cuboctahedra: dict | None = None,
    cuboctahedra_color=(0.55, 0.35, 0.85),
    cuboctahedra_opacity: float = 0.4,
) -> str:
    """Export a grid of static 3D structures as a self-contained HTML file.

    Each panel renders the final atoms of one :class:`Supercell` using ASE
    element colours, black outlines, and red bonds.  All panels share a
    camera that auto-rotates; dragging any panel pauses the rotation and
    orbits manually.

    Passing a ``tetrahedra``, ``octahedra``, or ``cuboctahedra`` dict
    switches the per-panel rendering from bonds to translucent polyhedra
    (4-vertex tets, 6-vertex octahedra, or 12-vertex FCC close-packed
    cuboctahedra respectively).  Keys for either dict:

    - ``center_symbol`` (default ``"Si"``) - element at the centre
    - ``vertex_symbol`` (default ``"O"``)  - element at each vertex
    - ``bond_length`` (default auto)        - centre-vertex ideal distance
    - ``bond_length_tol`` (default ``0.15``)
    - ``ideal_angle_deg`` (default ``109.47``)
    - ``angle_tol_deg`` (default ``25.0``)

    Only centres whose four nearest vertex atoms satisfy all of the above
    contribute a tetrahedron; bonds are not drawn in this mode.
    """
    import json
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors

    tetra_cfg = _resolve_polyhedra_cfg(tetrahedra, octahedra, cuboctahedra)
    # Pick the right colour/opacity based on which kwarg was supplied.
    if cuboctahedra is not None:
        poly_color = cuboctahedra_color
        poly_opacity = cuboctahedra_opacity
    elif octahedra is not None:
        poly_color = octahedra_color
        poly_opacity = octahedra_opacity
    else:
        poly_color = tetrahedra_color
        poly_opacity = tetrahedra_opacity

    def _bond_length_from_shell_target(cell, center_sym, vertex_sym):
        from ase.data import atomic_numbers
        st = getattr(cell, "_shell_target", None)
        if st is None:
            return None
        sp = np.asarray(st.species, dtype=np.int64)
        try:
            i = int(np.where(sp == atomic_numbers[center_sym])[0][0])
            j = int(np.where(sp == atomic_numbers[vertex_sym])[0][0])
        except (IndexError, KeyError):
            return None
        v = float(np.asarray(st.pair_peak, dtype=np.float64)[i, j])
        return v if v > 0 else None

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
        cell_mat = np.asarray(atoms.cell.array, dtype=np.float32)
        centre = 0.5 * np.sum(cell_mat, axis=0)
        pos = (atoms.positions - centre).astype(np.float32)
        pos = np.round(pos, 3)
        numbers = atoms.numbers
        colors = np.array([jmol_colors[z] for z in numbers], dtype=np.float32)
        radii = np.array([covalent_radii[z] for z in numbers], dtype=np.float32)

        if tetra_cfg is None:
            # Existing bond-filter path.
            cutoff_lo = pair_peak * (1.0 - bond_length_tol)
            cutoff_hi = pair_peak * (1.0 + bond_length_tol)
            search_cutoff = pair_peak * bond_cutoff_scale

            bi_all, bj_all, bd_all, bD_all = neighbor_list(
                "ijdD", atoms, float(search_cutoff),
            )
            length_ok = (bd_all >= cutoff_lo) & (bd_all <= cutoff_hi)
            bi_all = bi_all[length_ok]
            bj_all = bj_all[length_ok]
            bd_all = bd_all[length_ok]
            bD_all = bD_all[length_ok]

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
            tet_vertices_flat: list[float] = []
            num_tets = 0
            per_poly_faces: list[list[list[int]]] = []
            per_poly_edges: list[list[list[int]]] = []
        else:
            # Polyhedra mode: skip bonds, detect tets / octa / cubocta.
            effective_bond_length = tetra_cfg["bond_length"]
            if effective_bond_length is None:
                effective_bond_length = _bond_length_from_shell_target(
                    cell, tetra_cfg["center_symbol"], tetra_cfg["vertex_symbol"],
                )
            polys = tetra_cfg["detector"](
                atoms,
                center_symbol=tetra_cfg["center_symbol"],
                vertex_symbol=tetra_cfg["vertex_symbol"],
                bond_length=effective_bond_length,
                bond_length_tol=tetra_cfg["bond_length_tol"],
                ideal_angle_deg=tetra_cfg["ideal_angle_deg"],
                angle_tol_deg=tetra_cfg["angle_tol_deg"],
            )
            tet_vertices_flat = _polyhedra_vertex_coords(
                polys, atoms.positions, atoms.cell.array,
                scale=tetra_cfg["scale"],
            )
            tet_vertices_flat = [round(v, 3) for v in tet_vertices_flat]
            num_tets = len(polys)
            if tetra_cfg["per_polyhedron_topology"]:
                per_poly_faces = [p["faces"] for p in polys]
                per_poly_edges = [p["edges"] for p in polys]
            else:
                per_poly_faces = []
                per_poly_edges = []
            bi = np.zeros(0, dtype=np.int32)
            bj = np.zeros(0, dtype=np.int32)

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
            "tetrahedra_vertices": tet_vertices_flat,
            "num_tetrahedra": int(num_tets),
            "polyhedra_faces_per_poly": per_poly_faces,
            "polyhedra_edges_per_poly": per_poly_edges,
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
        "tetrahedra_mode": tetra_cfg is not None,
        "tetrahedra_color": list(poly_color),
        "tetrahedra_opacity": float(poly_opacity),
        # Generic polyhedra topology so the viewer can render either
        # tets (n=4) or octahedra (n=6) without hard-coded tables.
        "polyhedra_n_vertices": int(tetra_cfg["n_vertices"]) if tetra_cfg else 4,
        "polyhedra_faces": (
            tetra_cfg["faces"] if (tetra_cfg and tetra_cfg["faces"] is not None)
            else _TET_FACES
        ),
        "polyhedra_edges": (
            tetra_cfg["edges"] if (tetra_cfg and tetra_cfg["edges"] is not None)
            else _TET_EDGES
        ),
        "polyhedra_per_polyhedron_topology": bool(
            tetra_cfg and tetra_cfg["per_polyhedron_topology"]
        ),
        "polyhedra_scale": float(tetra_cfg["scale"]) if tetra_cfg else 1.0,
    }

    html = _OVERVIEW_HTML_TEMPLATE.replace(
        "__TRICOR_DATA_PLACEHOLDER__",
        json.dumps(data),
    )
    output_path = str(output_path)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def _detect_shell_mask(
    triplet_data: np.ndarray,
    r: np.ndarray,
    *,
    pair_peak: float | None = None,
    smooth_sigma_r: float = 0.25,
) -> np.ndarray:
    """Auto-detect the first NN shell window over r for one triplet of g3.

    Parameters
    ----------
    triplet_data
        One channel of the raw g3 histogram, shape ``(num_r, num_r, num_phi)``.
    r
        Bin-centre radii (Å), shape ``(num_r,)``.
    pair_peak
        Optional hint - the reference first-neighbour distance (Å) for this
        triplet's root bond, typically ``shell_target.pair_peak[center,
        neighbour]``.  When provided, the peak search is seeded here instead
        of at the first local maximum, which is much more robust at small
        cell sizes where g(r) is noisy.
    smooth_sigma_r
        Gaussian standard deviation applied to g(r) before peak / minimum
        detection, in Å.  Default 0.25 Å suppresses single-bin noise spikes
        at the typical ``r_step = 0.1`` Å grid without blurring the first
        shell.  Set to ``0`` to disable smoothing.
    """
    # Pair profile: collapse both angular-partner and phi dimensions.
    profile = triplet_data.sum(axis=(1, 2)) + triplet_data.sum(axis=(0, 2))
    profile = profile / np.maximum(r * r, _EPS)
    finite = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.zeros_like(r, dtype=bool)
    positive = np.flatnonzero(finite > 0)
    if positive.size == 0:
        return mask

    # Smooth before peak / minimum detection to avoid single-bin noise
    # triggering a premature "first minimum" right after the peak.
    r_step = float(r[1] - r[0]) if r.size > 1 else 1.0
    sigma_bins = max(0.0, float(smooth_sigma_r) / max(r_step, _EPS))
    if sigma_bins > 0.0:
        radius = max(1, int(np.ceil(3.0 * sigma_bins)))
        xk = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (xk / sigma_bins) ** 2)
        kernel /= float(kernel.sum())
        pad = np.pad(finite, (radius, radius), mode="edge")
        smoothed = np.convolve(pad, kernel, mode="valid")
    else:
        smoothed = finite

    # Seed the peak search: prefer the user-supplied pair_peak, fall back to
    # the first local maximum of the smoothed profile.
    if pair_peak is not None and np.isfinite(pair_peak) and pair_peak > 0:
        seed = int(np.argmin(np.abs(r - float(pair_peak))))
        # Narrow search window (+/- 15%) so a weak-liquid low-r bump
        # (atoms still close-packed from the random start) doesn't
        # out-vote the true first-neighbour peak near pair_peak.
        window_half = max(3, int(np.ceil(0.15 * float(pair_peak) / max(r_step, _EPS))))
        lo = max(0, seed - window_half)
        hi = min(smoothed.size, seed + window_half + 1)
        peak_bin = int(lo + int(np.argmax(smoothed[lo:hi])))
    else:
        start = int(positive[0])
        peak_bin = None
        for idx in range(max(start, 1), smoothed.size - 1):
            if smoothed[idx] >= smoothed[idx - 1] and smoothed[idx] > smoothed[idx + 1]:
                peak_bin = idx
                break
        if peak_bin is None:
            peak_bin = int(start + int(np.argmax(smoothed[start:])))

    # Bound the search with pair_peak-relative limits so a flat or noisy
    # profile never widens the window into the second shell (on the right)
    # or into the tiny-r numerical divergence (on the left).  Defaults
    # assume a first-shell width of roughly +/- 25% of pair_peak.
    if pair_peak is not None and np.isfinite(pair_peak) and pair_peak > 0:
        lo_lim_r = 0.75 * float(pair_peak)
        hi_lim_r = 1.30 * float(pair_peak)
    else:
        # Without pair_peak, fall back to bounds relative to the detected peak.
        lo_lim_r = 0.70 * float(r[peak_bin])
        hi_lim_r = 1.35 * float(r[peak_bin])
    lo_lim_bin = int(max(0, np.searchsorted(r, lo_lim_r) - 1))
    hi_lim_bin = int(min(r.size - 1, np.searchsorted(r, hi_lim_r)))

    # Walk inward / outward to the first *significant* smoothed local
    # minimum.  "Significant" means below a fraction of the peak - just
    # any noise wiggle used to stop the walk, and for noisy liquids
    # with a nearly-flat profile the window was collapsing to almost a
    # single bin.
    peak_val = float(smoothed[peak_bin])
    # Dip threshold: a candidate minimum has to be at most ``min_ratio``
    # of peak_val to count.  0.85 still lets us pick the obvious valley
    # between first and second shell in crystalline cases.
    min_ratio = 0.85
    dip_thresh = peak_val * min_ratio

    left_bin = lo_lim_bin
    for idx in range(peak_bin - 1, lo_lim_bin, -1):
        if (smoothed[idx] <= smoothed[idx - 1]
                and smoothed[idx] < smoothed[idx + 1]
                and smoothed[idx] <= dip_thresh):
            left_bin = idx
            break

    right_bin = hi_lim_bin
    for idx in range(peak_bin + 1, hi_lim_bin):
        if (smoothed[idx] < smoothed[idx - 1]
                and smoothed[idx] <= smoothed[idx + 1]
                and smoothed[idx] <= dip_thresh):
            right_bin = idx
            break

    mask[left_bin : right_bin + 1] = True
    return mask


def _g3_pair_profile(
    dist: "Any",
    triplet_idx: int,
    r: np.ndarray,
) -> np.ndarray:
    """Return the per-triplet ROOT-bond pair profile g(r), normalised so
    the tail -> 1.0.

    For a triplet ``A | B C`` the root bond is ``A-B`` (center to first
    neighbour).  The heatmap above the profile integrates over this same
    A-B shell to expose how the third atom C is distributed about the
    A-B backbone.
    """
    g2 = getattr(dist, "g2", None)
    g3_index = getattr(dist, "g3_index", None)
    if g2 is not None and g3_index is not None:
        center_ind, neigh1_ind, _neigh2_ind = g3_index[triplet_idx]
        profile = np.asarray(g2[center_ind, neigh1_ind], dtype=np.float64).copy()
    else:
        # Fall back to integrating r2 and phi out of g3, which leaves
        # the r1 profile (the root bond).
        triplet_data = np.asarray(dist.g3[triplet_idx], dtype=np.float64)
        profile = triplet_data.sum(axis=(1, 2))

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

    Integrates the root bond (axis 0, ``r1``) over ``shell_mask``; the
    remaining ``(r2, phi)`` plane shows where the *third* atom sits
    relative to the root bond.  Normalised so that the uniform far-field
    tends to 1.0.
    """
    image = triplet_data[shell_mask, :, :].sum(axis=0)
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
        *,
        polyhedra: "dict | bool | None" = True,
        **kwargs,
    ):
        """Return an interactive 3D structure viewer widget.

        Renders atoms as spheres (coloured by element).  Two overlay
        modes, independently toggle-able in the widget:

        * Bonds - cylinders between atoms within a radial cutoff.
        * Polyhedra - translucent tetrahedra / octahedra /
          cuboctahedra around atoms that pass a distance + angle
          tolerance check (see
          :meth:`export_trajectory_html` / :func:`_detect_tetrahedra`
          for the underlying algorithm).  Enabled by default for
          materials whose coordination polyhedron we can auto-detect
          (Si / C tetrahedra at half-scale, Cu cuboctahedra, SiO2
          tetrahedra, SrTiO3 octahedra).

        Sliders in the side panel let you tune the radial tolerance,
        angular tolerance, centre-vertex bond length, and polyhedra
        scale (0.5 places vertices at bond midpoints, 1.0 at atoms).

        Parameters
        ----------
        shell_target
            Sets the default bond cutoff and bond_length from
            ``shell_target.max_pair_outer`` / ``pair_peak``.  If
            ``None``, uses the shell_target stored from the last
            :meth:`generate` call.
        polyhedra
            Polyhedra config.  ``True`` / ``None`` (default) auto-pick
            kind + settings from species; ``False`` disables polyhedra;
            a ``dict`` overrides individual settings - e.g.
            ``{'kind': 'octahedra', 'center_symbol': 'Ti',
            'vertex_symbol': 'O', 'bond_length': 1.96}``.
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
            polyhedra=polyhedra,
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
        tetrahedra: dict | None = None,
        tetrahedra_color=(0.25, 0.65, 0.95),
        tetrahedra_opacity: float = 0.35,
        octahedra: dict | None = None,
        octahedra_color=(0.95, 0.55, 0.25),
        octahedra_opacity: float = 0.4,
        cuboctahedra: dict | None = None,
        cuboctahedra_color=(0.55, 0.35, 0.85),
        cuboctahedra_opacity: float = 0.4,
        show_bonds: bool | None = None,
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
        show_bonds
            Whether to emit bond cylinders.  ``None`` (default) auto-picks:
            ``True`` when no ``tetrahedra`` are requested, ``False`` when
            they are (tetrahedra supersede bonds).  Pass ``True`` / ``False``
            explicitly to override.

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
            # Global colour range (constant across frames).  We scale
            # to the STEADY-STATE cost - the 99th percentile of the
            # last quarter of frames - rather than percentile-of-all,
            # which is dominated by the initial random-position chaos
            # for liquid-path runs (Cu liquid starting at random
            # positions has early per-atom costs of ~100 even with
            # bond_weight=0.05, which then saturates the colour scale
            # long after those atoms have relaxed).
            cost_min = 0.0
            n_frames_cost = atom_cost.shape[0]
            tail_start = max(0, int(n_frames_cost * 0.75))
            tail = atom_cost[tail_start:].ravel()
            positive = tail[tail > 0.0]
            if positive.size == 0:
                positive = atom_cost.ravel()[atom_cost.ravel() > 0]
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

        # Polyhedra topology (from final frame) if requested; disables bonds.
        tetra_cfg = _resolve_polyhedra_cfg(tetrahedra, octahedra, cuboctahedra)
        if cuboctahedra is not None:
            poly_color = cuboctahedra_color
            poly_opacity = cuboctahedra_opacity
        elif octahedra is not None:
            poly_color = octahedra_color
            poly_opacity = octahedra_opacity
        else:
            poly_color = tetrahedra_color
            poly_opacity = tetrahedra_opacity

        tet_centers: list[int] = []
        tet_vertex_idx: list[int] = []
        per_poly_faces: list[list[list[int]]] = []
        per_poly_edges: list[list[list[int]]] = []

        # Resolve show_bonds auto-default: bonds follow atoms when no
        # polyhedra are requested; polyhedra supersede bonds otherwise.
        if show_bonds is None:
            show_bonds_eff = tetra_cfg is None
        else:
            show_bonds_eff = bool(show_bonds)

        if tetra_cfg is not None:
            effective_bond_length = tetra_cfg["bond_length"]
            if effective_bond_length is None:
                from ase.data import atomic_numbers
                _st = getattr(self, "_shell_target", None)
                if _st is not None:
                    _sp = np.asarray(_st.species, dtype=np.int64)
                    try:
                        ci = int(np.where(_sp == atomic_numbers[tetra_cfg["center_symbol"]])[0][0])
                        vi = int(np.where(_sp == atomic_numbers[tetra_cfg["vertex_symbol"]])[0][0])
                        v = float(np.asarray(_st.pair_peak, dtype=np.float64)[ci, vi])
                        if v > 0:
                            effective_bond_length = v
                    except (IndexError, KeyError):
                        pass
            polys = tetra_cfg["detector"](
                self.atoms,
                center_symbol=tetra_cfg["center_symbol"],
                vertex_symbol=tetra_cfg["vertex_symbol"],
                bond_length=effective_bond_length,
                bond_length_tol=tetra_cfg["bond_length_tol"],
                ideal_angle_deg=tetra_cfg["ideal_angle_deg"],
                angle_tol_deg=tetra_cfg["angle_tol_deg"],
            )
            tet_centers = [t["center"] for t in polys]
            tet_vertex_idx = [v for t in polys for v in t["vertices"]]
            if tetra_cfg["per_polyhedron_topology"]:
                per_poly_faces = [p["faces"] for p in polys]
                per_poly_edges = [p["edges"] for p in polys]

        if show_bonds_eff:
            bi_all, bj_all, bd_all = neighbor_list(
                "ijd", self.atoms, float(bond_cutoff),
            )
            mask = bi_all < bj_all
            bi = bi_all[mask].astype(np.int32)
            bj = bj_all[mask].astype(np.int32)
        else:
            bi = np.zeros(0, dtype=np.int32)
            bj = np.zeros(0, dtype=np.int32)

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

        data["tetrahedra_mode"] = tetra_cfg is not None
        data["tetrahedra_centers"] = list(tet_centers)
        data["tetrahedra_vertex_indices"] = list(tet_vertex_idx)
        data["num_tetrahedra"] = len(tet_centers)
        data["tetrahedra_color"] = list(poly_color)
        data["tetrahedra_opacity"] = float(poly_opacity)
        # Generic polyhedra topology so the viewer can render tets
        # (n=4), octahedra (n=6), or cuboctahedra (n=12).
        data["polyhedra_n_vertices"] = (
            int(tetra_cfg["n_vertices"]) if tetra_cfg else 4
        )
        data["polyhedra_faces"] = (
            tetra_cfg["faces"] if (tetra_cfg and tetra_cfg["faces"] is not None)
            else _TET_FACES
        )
        data["polyhedra_edges"] = (
            tetra_cfg["edges"] if (tetra_cfg and tetra_cfg["edges"] is not None)
            else _TET_EDGES
        )
        data["polyhedra_per_polyhedron_topology"] = bool(
            tetra_cfg and tetra_cfg["per_polyhedron_topology"]
        )
        data["polyhedra_faces_per_poly"] = per_poly_faces
        data["polyhedra_edges_per_poly"] = per_poly_edges
        data["polyhedra_scale"] = float(tetra_cfg["scale"]) if tetra_cfg else 1.0

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
        show_all_triplets: bool = False,
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
        show_all_triplets
            If True, the viewer renders a grid of all triplet heatmaps
            simultaneously (sharing one legend and colour scale) instead
            of the interactive single-panel view.  Useful for multi-
            species cases like SiO\u2082 where it's otherwise unclear which
            triplet channel is being displayed.
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

        # Per-triplet pair_peak from the supercell's shell target (if present).
        # Used to seed the shell-mask peak search; much more robust than
        # falling back to "first local maximum" on a noisy 20x20x20 g(r).
        shell_target = getattr(self, "_shell_target", None)
        pair_peak_matrix = None
        if shell_target is not None:
            pair_peak_matrix = np.asarray(shell_target.pair_peak, dtype=np.float64)
        g3_index = getattr(dist, "g3_index", None)

        # Per-triplet root-bond label ("Si-O" etc.) for the g(r) panel
        # below each heatmap, derived from g3_index + species_labels so it
        # matches however the user labels each species.
        species_labels = [str(s) for s in getattr(dist, "species_labels", None) or []]
        if not species_labels:
            from ase.data import chemical_symbols as _chemical_symbols
            species_labels = [_chemical_symbols[int(z)] for z in dist.species]
        profile_labels: list[str] = []
        for ti in range(num_triplets):
            if g3_index is not None:
                center_ind, neigh1_ind, _neigh2_ind = g3_index[ti]
                profile_labels.append(
                    f"g(r) {species_labels[int(center_ind)]}-"
                    f"{species_labels[int(neigh1_ind)]}"
                )
            else:
                profile_labels.append("g(r)")

        slices = []
        pair_profiles = []
        shell_ranges = []
        for ti in range(num_triplets):
            triplet_data = np.asarray(dist.g3[ti], dtype=np.float64)

            triplet_pair_peak = None
            if pair_peak_matrix is not None and g3_index is not None:
                center_ind, neigh1_ind, _neigh2_ind = g3_index[ti]
                val = float(pair_peak_matrix[int(center_ind), int(neigh1_ind)])
                if np.isfinite(val) and val > 0:
                    triplet_pair_peak = val

            shell_mask = _detect_shell_mask(
                triplet_data, r, pair_peak=triplet_pair_peak,
            )
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

        # Default triplet: prefer a canonical tetrahedral channel when
        # present (Si|O O for silica, Si|Si Si for silicon, etc.).
        # Labels are like "Si | O O"; compare the center/neighbour
        # symbols after normalising whitespace.
        default_triplet = 0
        preferred = [
            "Si | O O", "Si | Si Si", "C | C C", "Cu | Cu Cu",
            "Ti | O O", "Sr | O O",
        ]
        norm_labels = [str(lab).replace("  ", " ").strip() for lab in labels]
        for candidate in preferred:
            if candidate in norm_labels:
                default_triplet = norm_labels.index(candidate)
                break

        data = {
            "num_r": int(r.size),
            "num_phi": int(phi_deg.size),
            "num_triplets": num_triplets,
            "r_centers": r.tolist(),
            "r_edges": r_edges.tolist(),
            "phi_centers_deg": phi_deg.tolist(),
            "phi_edges_deg": phi_edges_deg.tolist(),
            "triplet_labels": labels,
            "profile_labels": profile_labels,
            "default_triplet": int(default_triplet),
            "slices": slices,
            "pair_profiles": pair_profiles,
            "shell_ranges": shell_ranges,
            "background_color": background_color,
            "title": title,
            "show_all": bool(show_all_triplets),
        }

        html = _G3_HTML_TEMPLATE.replace(
            "__TRICOR_DATA_PLACEHOLDER__",
            json.dumps(data),
        )
        output_path = str(output_path)
        with open(output_path, "w") as f:
            f.write(html)
        return output_path

    def export_g2_html(
        self: "Supercell",
        output_path: "str | None" = None,
        *,
        r_max: float = 10.0,
        r_step: float = 0.05,
        background_color: str = "#f7f8f5",
        title: str = "",
        show_progress: bool = False,
    ) -> str:
        """Export a standalone interactive 2D g(r) viewer.

        Shows the per-species-pair pair-correlation function g_{AB}(r) as
        a single g(r) plot with a dropdown to switch species pair, and
        an "overlay all" checkbox to compare all pairs on one axis.
        Essentially the bottom panel of :meth:`export_g3_html` lifted
        out on its own with a species-pair selector - useful as a
        quick 2-body PDF viewer without any angular content.

        Parameters
        ----------
        output_path
            Path to write the HTML file.  When ``None`` the HTML is not
            written to disk; the raw HTML string is still returned so
            the caller can embed it directly (for example via
            :func:`IPython.display.HTML`) - see also :meth:`plot_g2`
            for a ready-made Jupyter wrapper.
        r_max, r_step
            Radial grid for the measurement.  A finer ``r_step`` (default
            0.05 \u00c5) gives smoother curves than the coarse 0.1 \u00c5
            grid used by the g3 viewer.
        background_color, title
            Cosmetic.
        show_progress
            Forwarded to the underlying :meth:`measure_g3` call (this
            exporter reuses the g3 machinery because the g2 array is
            a by-product; ``phi_num_bins`` is set low for speed).

        Returns
        -------
        str
            The resolved output path when ``output_path`` was provided,
            otherwise the HTML source string.
        """
        import json

        from .g3 import G3Distribution
        from ase.data import chemical_symbols

        dist = G3Distribution(self.atoms, label=f"{self.label}-export-g2")
        # Low phi_num_bins because we don't use the angular data here;
        # the speedup is significant for large boxes.
        dist.measure_g3(
            r_max=r_max,
            r_step=r_step,
            phi_num_bins=12,
            show_progress=show_progress,
        )
        if dist.g2 is None:
            raise ValueError("Measured distribution has no g2 array.")

        r = np.asarray(dist.r, dtype=np.float64)
        r_edges = np.asarray(dist.bin_edges, dtype=np.float64)

        species = np.asarray(dist.species, dtype=np.int64)
        sp_labels = [chemical_symbols[int(z)] for z in species]

        # g2[ci, vi] is the pair-specific RDF; (ci, vi) and (vi, ci)
        # differ by a constant prefactor but are otherwise identical.
        # Emit one curve per unique (ci <= vi) pair and drop the
        # symmetric duplicate.
        num_species = len(species)
        g2 = np.asarray(dist.g2, dtype=np.float64)

        # Normalise each pair profile so the tail converges to 1.0
        # (consistent with the g3 viewer's profile panel).
        def _norm_profile(y: np.ndarray, r_arr: np.ndarray) -> np.ndarray:
            y = y / np.maximum(r_arr * r_arr, _EPS)
            tail_start = 0.7 * float(r_arr[-1])
            tail_mask = r_arr >= tail_start
            if not np.any(tail_mask):
                tail_mask = np.zeros_like(r_arr, dtype=bool)
                tail_mask[-max(1, r_arr.size // 4):] = True
            tail = y[tail_mask]
            finite = tail[np.isfinite(tail)]
            scale = float(np.mean(finite)) if finite.size else 1.0
            if scale <= _EPS:
                scale = 1.0
            return (y / scale).astype(np.float32)

        pair_labels: list[str] = []
        profiles: list[list[float]] = []
        pair_peaks: list[float] = []
        shell_target = getattr(self, "_shell_target", None)
        pair_peak_matrix = None
        if shell_target is not None:
            pair_peak_matrix = np.asarray(
                shell_target.pair_peak, dtype=np.float64
            )

        # Resolve a shell-target species index for each dist species.
        st_species_index: dict[int, int] = {}
        if shell_target is not None:
            st_species = np.asarray(shell_target.species, dtype=np.int64)
            for ci_dist, z in enumerate(species):
                match = np.where(st_species == int(z))[0]
                if match.size:
                    st_species_index[ci_dist] = int(match[0])

        for ci in range(num_species):
            for vi in range(ci, num_species):
                prof = _norm_profile(g2[ci, vi], r)
                pair_labels.append(f"{sp_labels[ci]}-{sp_labels[vi]}")
                profiles.append(prof.tolist())
                peak = 0.0
                if pair_peak_matrix is not None:
                    ci_st = st_species_index.get(ci)
                    vi_st = st_species_index.get(vi)
                    if ci_st is not None and vi_st is not None:
                        v = float(pair_peak_matrix[ci_st, vi_st])
                        if np.isfinite(v) and v > 0:
                            peak = v
                pair_peaks.append(peak)

        # Default pair: prefer shortest-bond cross-species if available,
        # else first pair.
        default_pair = 0
        if pair_peaks:
            # Exclude self-pairs with zero peak; pick the smallest positive peak.
            positive = [(i, p) for i, p in enumerate(pair_peaks) if p > 0]
            if positive:
                default_pair = min(positive, key=lambda x: x[1])[0]

        data = {
            "num_r": int(r.size),
            "num_pairs": len(pair_labels),
            "r_centers": r.tolist(),
            "r_edges": r_edges.tolist(),
            "pair_labels": pair_labels,
            "pair_peaks": pair_peaks,
            "profiles": profiles,
            "default_pair": int(default_pair),
            "background_color": background_color,
            "title": title,
        }

        html = _G2_HTML_TEMPLATE.replace(
            "__TRICOR_DATA_PLACEHOLDER__",
            json.dumps(data),
        )
        if output_path is None:
            return html
        output_path = str(output_path)
        with open(output_path, "w") as f:
            f.write(html)
        return output_path

    def plot_g2(
        self: "Supercell",
        *,
        r_max: float = 10.0,
        r_step: float = 0.05,
        title: str = "",
        height: int = 420,
        show_progress: bool = False,
    ):
        """Return an inline Jupyter display of the g(r) pair-correlation
        viewer.

        Convenience wrapper around :meth:`export_g2_html` that packages
        the HTML as an :class:`IPython.display.HTML` object so you can
        just do ``cells['MRO'].plot_g2()`` in a notebook cell.  The
        viewer is embedded via a ``srcdoc`` iframe so it renders
        isolated from the surrounding notebook CSS / JS.

        Parameters
        ----------
        r_max, r_step, title, show_progress
            Forwarded to :meth:`export_g2_html`.
        height
            Iframe height in pixels.
        """
        from IPython.display import HTML
        html = self.export_g2_html(
            None, r_max=r_max, r_step=r_step,
            title=title, show_progress=show_progress,
        )
        import html as _html
        escaped = _html.escape(html, quote=True)
        # Wrap the iframe in a <div> so IPython's HTML helper doesn't
        # trigger its "Consider using IPython.display.IFrame instead"
        # warning - IFrame requires a URL or file path and can't embed
        # our self-contained HTML directly, so srcdoc is what we want.
        return HTML(
            f'<div class="tricor-g2-wrapper" style="width:100%">'
            f'<iframe srcdoc="{escaped}" '
            f'style="width:100%; height:{int(height)}px; '
            f'border:1px solid rgba(0,0,0,0.1); border-radius:6px;"></iframe>'
            f'</div>'
        )

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


