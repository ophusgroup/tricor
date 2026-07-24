import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

Z_NI, Z_CR = 28, 24


def _greedy_merge(atoms, ID, min_spacing):
    """Greedily drop atoms within min_spacing of an already-kept atom.
    A merged site becomes Cr if any atom in its cluster was Cr."""
    tree = cKDTree(atoms)
    keep = np.ones(len(atoms), bool)
    ID = ID.copy()
    for i in range(len(atoms)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(atoms[i], min_spacing):
            if j != i and keep[j]:
                keep[j] = False
                if ID[j] == Z_CR:
                    ID[i] = Z_CR
    return atoms[keep], ID[keep]


def make_sft_cell(plane_order, dz=None, min_spacing=0.8, cell_dim=(40, 40, 40), plot=True):
    """
    Stacking-fault tetrahedron from a close-packed stacking sequence.

    One 1/4 "flattened" sector is built, then tiled into the four tetrahedral
    orientations (identity + three 180-deg diagonal rotations).  Overlapping
    atoms produced by the tiling are removed by a greedy merge, the cell is
    centred in `cell_dim`, and atoms outside the box are cropped.

    plane_order : ints in {0,1,2} = stacking pos (A/B/C) of each plane.
        (0,1,2)*3 -> fcc ; (0,1)*6 -> hcp ; (0,1,2,0,1,2,1,2,0,1,2) -> 1 fault
    dz          : inter-plane spacing; None -> close-packed sqrt(2/3) (NN = 1).
    min_spacing : atoms closer than this are merged (greedy).
    cell_dim    : box size; structure is centred at cell_dim/2 and cropped to it.

    Returns atoms (M,3), ID (M,) atomic numbers (Ni=28, Cr=24 on fault planes).
    """
    plane_order = np.asarray(plane_order, dtype=int)
    N = plane_order.size
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    o = (a1 + a2) / 3.0
    oB = np.array([o, o - a1, o - a2])
    if dz is None:
        dz = np.sqrt(2.0 / 3.0)                         # close-packed {111} spacing

    # Ni/Cr per plane: "hcp-like" (Cr) if the two neighbouring planes match
    is_cr = np.zeros(N, bool)
    for k in range(1, N - 1):
        is_cr[k] = plane_order[k - 1] == plane_order[k + 1]
    if N >= 2:
        is_cr[0], is_cr[-1] = is_cr[1], is_cr[-2]
    plane_Z = np.where(is_cr, Z_CR, Z_NI)

    # --- full compact tetrahedron, stacking along local z (1,3,6,... per plane) ---
    xy, zz, pid = [], [], []
    for k in range(N):
        abc = np.array([(a, b, k - a - b)
                        for a in range(k + 1) for b in range(k + 1 - a)])
        q = abc @ oB + ((plane_order[k] - k) % 3) * o
        xy.append(q); zz.append(np.full(len(q), k * dz)); pid.append(np.full(len(q), k))
    tetra = np.column_stack([np.vstack(xy), np.concatenate(zz)])
    tetra_Z = plane_Z[np.concatenate(pid)]

    # centre on centroid, rotate stacking axis (z) -> cubic [111].
    # (the compact crystal already IS the four D2 sectors; the greedy merge below
    #  cleans up any coincident atoms, so duplication during tiling is harmless.)
    tetra -= tetra.mean(0)
    zc = np.array([1, 1, 1.]) / np.sqrt(3)
    xc = np.array([1, -1, 0.]) / np.sqrt(2)
    R = np.column_stack([xc, np.cross(zc, xc), zc])
    atoms, ID = tetra @ R.T, tetra_Z

    # --- greedy merge, centre in the cell, crop to the box -------------------
    atoms, ID = _greedy_merge(atoms, ID, min_spacing)
    cell = np.asarray(cell_dim, float)
    atoms += cell / 2 - atoms.mean(0)
    inbox = np.all((atoms >= 0) & (atoms <= cell), axis=1)
    atoms, ID = atoms[inbox], ID[inbox]

    if plot:
        _plot_projections(atoms, ID, cell)
    return atoms, ID


def _proj_basis(d):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    ref = np.array([0, 0, 1.]) if abs(d[2]) < 0.9 else np.array([1, 0, 0.])
    u = np.cross(d, ref); u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _plot_projections(atoms, ID, cell, dirs=(("001", (0, 0, 1)),
                                             ("011", (0, 1, 1)), ("111", (1, 1, 1)))):
    c = cell / 2
    col = np.where(ID == Z_CR, "tab:red", "0.55")
    lim = np.abs(atoms - c).max() * 1.1
    fig, axes = plt.subplots(1, len(dirs), figsize=(5 * len(dirs), 5))
    for ax, (name, d) in zip(np.atleast_1d(axes), dirs):
        u, v = _proj_basis(d)
        ax.scatter((atoms - c) @ u, (atoms - c) @ v, c=col, s=22, edgecolor="k", linewidth=0.2)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"[{name}]  (Cr=red, Ni=grey)")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    for name, po in [("fcc", (0, 1, 2) * 3), ("hcp", (0, 1) * 6),
                     ("fault", (0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2))]:
        atoms, ID = make_sft_cell(po, plot=False)
        d, _ = cKDTree(atoms).query(atoms, k=2)
        print(f"{name:6s}: {len(atoms):4d} atoms, minNN={d[:,1].min():.4f}, "
              f"Cr={(ID==Z_CR).sum()} Ni={(ID==Z_NI).sum()}, "
              f"centroid={np.round(atoms.mean(0),2)}")
    make_sft_cell((0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2), plot=True)
    plt.savefig("scratch/make_sft_cell_demo.png", dpi=130)
