"""Dataset infrastructure for tricor ML training.

Two responsibilities:

1. **Graph construction.** Build PBC-aware ``edge_index`` + ``edge_vec``
   tensors from atomic positions + a cubic box.  Used both at training
   (per sample) and at inference (single big cell).

2. **HDF5 dataset reader.** Iterate the (Voronoi tile, FIRE-final)
   pairs produced by ``scripts/ml_generate_data.py`` in batches
   suitable for graph mini-batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def build_pbc_graph(
    positions: np.ndarray | torch.Tensor,
    box_dim: Sequence[float],
    r_cut: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a PBC-aware neighbour graph.

    Returns
    -------
    edge_index : torch.LongTensor of shape (2, E)
        ``edge_index[0]`` is the source atom index, ``edge_index[1]``
        is the destination.  Edges are directed; each unordered pair
        appears twice (i→j and j→i).
    edge_vec : torch.Tensor of shape (E, 3)
        Min-image displacement ``pos[dst] - pos[src]`` for each edge.

    Notes
    -----
    Uses a naive O(N²) all-pairs distance computation, mod the PBC
    min-image transform.  This is fine for training cells (≤ 5000
    atoms — ~25 M pair tests, ~50 ms on GPU).  Inference on
    100×100×500 cells (~400 k atoms) will need a cell-list neighbour
    search; that's a Phase 4 optimization.
    """
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).float()
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")
    box = torch.tensor(box_dim, dtype=positions.dtype, device=positions.device)
    n = positions.shape[0]

    # Pairwise displacements (j - i) with min-image PBC correction.
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)        # (N, N, 3)
    delta = delta - torch.round(delta / box) * box                 # min image
    d2 = (delta * delta).sum(dim=-1)                               # (N, N)

    # Mask self-edges and beyond cutoff
    mask = (d2 > 0.0) & (d2 < r_cut * r_cut)
    src, dst = torch.nonzero(mask, as_tuple=True)                  # (E,)
    edge_index = torch.stack([src, dst], dim=0).long()             # (2, E)
    edge_vec = delta[src, dst]                                     # (E, 3)
    return edge_index, edge_vec


def _build_pbc_graph_celllist(
    positions: np.ndarray | torch.Tensor,
    box_dim: Sequence[float],
    r_cut: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-list PBC neighbour graph via :class:`scipy.spatial.cKDTree`.

    Complexity ``O(N log N)`` build + ``O(M)`` pair extraction where
    ``M`` is the number of pairs within ``r_cut`` (≈ ``N × ⟨neighbours⟩``).
    For 600 k-atom cells the all-pairs version is ~1 minute / chunk;
    this version returns in a few seconds total.

    Always returns CPU tensors — the caller moves them to the model's
    device.  Falls back to the all-pairs ``_build_pbc_graph_allpairs``
    if SciPy is unavailable (small training-cell case stays correct
    without the extra dependency check).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover — scipy is a hard dep of ase
        return _build_pbc_graph_allpairs(positions, box_dim, r_cut,
                                         chunk_size=4096)

    if isinstance(positions, torch.Tensor):
        out_dtype = positions.dtype
        out_device = positions.device
        pos_np = positions.detach().cpu().numpy().astype(np.float64)
    else:
        out_dtype = torch.float32
        out_device = torch.device("cpu")
        pos_np = np.asarray(positions, dtype=np.float64)
    box_np = np.asarray(box_dim, dtype=np.float64)

    # cKDTree with boxsize requires all positions in [0, box).
    pos_wrapped = pos_np - np.floor(pos_np / box_np) * box_np
    tree = cKDTree(pos_wrapped, boxsize=box_np)
    pairs = tree.query_pairs(r_cut, output_type="ndarray")    # (M, 2), i<j
    if len(pairs) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=out_device),
            torch.zeros(0, 3, dtype=out_dtype, device=out_device),
        )
    src = pairs[:, 0]
    dst = pairs[:, 1]
    delta = pos_wrapped[dst] - pos_wrapped[src]
    delta -= np.round(delta / box_np) * box_np

    # Materialise both directions so the model sees a symmetric graph.
    src_full = np.concatenate([src, dst])
    dst_full = np.concatenate([dst, src])
    delta_full = np.concatenate([delta, -delta])

    edge_index = torch.from_numpy(np.stack([src_full, dst_full])).long()
    edge_vec = torch.from_numpy(delta_full).to(out_dtype)
    if out_device.type != "cpu":
        edge_index = edge_index.to(out_device)
        edge_vec = edge_vec.to(out_device)
    return edge_index, edge_vec


def _build_pbc_graph_allpairs(
    positions: np.ndarray | torch.Tensor,
    box_dim: Sequence[float],
    r_cut: float,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference O(N²) chunked PBC graph — kept as a fallback / sanity check.

    For small cells (≤ a few thousand atoms) this is competitive with
    the cell-list path and useful for unit-test parity checking.
    """
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).float()
    box = torch.tensor(box_dim, dtype=positions.dtype, device=positions.device)
    n = positions.shape[0]

    src_list: list[torch.Tensor] = []
    dst_list: list[torch.Tensor] = []
    vec_list: list[torch.Tensor] = []
    r_cut_sq = r_cut * r_cut

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        pos_chunk = positions[i_start:i_end]                       # (C, 3)
        delta = positions.unsqueeze(0) - pos_chunk.unsqueeze(1)    # (C, N, 3)
        delta = delta - torch.round(delta / box) * box
        d2 = (delta * delta).sum(dim=-1)                           # (C, N)
        mask = (d2 > 0.0) & (d2 < r_cut_sq)
        ci, dst = torch.nonzero(mask, as_tuple=True)
        src = ci + i_start
        src_list.append(src.long())
        dst_list.append(dst.long())
        vec_list.append(delta[ci, dst])

    if not src_list:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=positions.device),
            torch.zeros(0, 3, dtype=positions.dtype, device=positions.device),
        )

    edge_index = torch.stack(
        [torch.cat(src_list), torch.cat(dst_list)], dim=0
    )
    edge_vec = torch.cat(vec_list, dim=0)
    return edge_index, edge_vec


def build_pbc_graph_chunked(
    positions: np.ndarray | torch.Tensor,
    box_dim: Sequence[float],
    r_cut: float,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient PBC graph for large cells.

    Auto-routes to a SciPy ``cKDTree`` cell-list (O(N log N), C code)
    when available — this is the production path for inference on
    100 k+ atom cells where the O(N²) all-pairs path takes minutes.

    The ``chunk_size`` argument is retained for the all-pairs fallback
    only; the cell-list path ignores it.
    """
    return _build_pbc_graph_celllist(positions, box_dim, r_cut)


class TricorMLDataset(Dataset):
    """HDF5-backed dataset of (Voronoi tile, FIRE-final) pairs.

    Each sample is a dict consumable directly by the training loop's
    collate function (:func:`collate_cells`).

    File format
    -----------
    See ``scratch/PLAN.md`` "Dataset schema" section for the exact
    HDF5 layout.  In short: top-level groups ``/cell_NNNN`` each
    holding ``voronoi_pos``, ``fire_pos``, ``species_idx`` datasets +
    ``grain_size``, ``box_dim``, ``regime`` attributes.
    """

    def __init__(
        self,
        h5_path: str | Path,
        r_cut: float = 5.0,
        # If True, also yield intermediate fire_trajectory frames as
        # extra "(input, target)" pairs — gives ~32× more training
        # data per cell because every FIRE frame becomes a target for
        # the Voronoi tile.  Disabled by default; enable once the
        # base regression is debugged.
        use_trajectory_frames: bool = False,
        # Restrict to cells whose largest box side is ≤ this many Å.
        # Useful for fast training on a "small cells only" subset.
        max_box_side: float | None = None,
        # Restrict to a subset of regimes (e.g. ['amorphous',
        # 'nanocrystalline']).  ``None`` keeps everything.
        regime_filter: list[str] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.r_cut = float(r_cut)
        self.use_trajectory_frames = use_trajectory_frames
        self.max_box_side = max_box_side
        self.regime_filter = regime_filter

        # Build index: list of (cell_key, frame_idx_or_None).
        # frame_idx_or_None == None means "use fire_pos as the target".
        self._index: list[tuple[str, int | None]] = []
        with h5py.File(self.h5_path, "r") as f:
            for key in f.keys():
                if not key.startswith("cell_"):
                    continue
                g = f[key]
                if max_box_side is not None:
                    box = np.asarray(g.attrs.get("box_dim", [0., 0., 0.]))
                    if float(box.max()) > float(max_box_side) + 1e-3:
                        continue
                if regime_filter is not None:
                    if str(g.attrs.get("regime", "")) not in regime_filter:
                        continue
                self._index.append((key, None))
                if use_trajectory_frames and "fire_trajectory" in g:
                    n_frames = g["fire_trajectory"].shape[0]
                    # Skip the first frame (== voronoi_pos) and the
                    # last (== fire_pos).
                    for fi in range(1, n_frames - 1):
                        self._index.append((key, fi))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        key, frame_idx = self._index[idx]
        with h5py.File(self.h5_path, "r") as f:
            g = f[key]
            voronoi_pos = np.asarray(g["voronoi_pos"], dtype=np.float32)
            if frame_idx is None:
                target_pos = np.asarray(g["fire_pos"], dtype=np.float32)
            else:
                target_pos = np.asarray(
                    g["fire_trajectory"][frame_idx], dtype=np.float32,
                )
            species_idx = np.asarray(g["species_idx"], dtype=np.int64)
            grain_size = float(g.attrs["grain_size"])
            box_dim = np.asarray(g.attrs["box_dim"], dtype=np.float32)
            regime = str(g.attrs.get("regime", ""))
        return {
            "voronoi_pos": torch.from_numpy(voronoi_pos),
            "target_pos": torch.from_numpy(target_pos),
            "species_idx": torch.from_numpy(species_idx),
            "grain_size": float(grain_size),
            "box_dim": torch.from_numpy(box_dim),
            "regime": regime,
        }


class TricorMLStepDataset(Dataset):
    """(x_t, x_{t+stride}) pairs sampled from FIRE trajectories.

    Used to train an *iterative* force-style predictor — the network
    learns to map any trajectory frame to the one ``stride`` captured
    frames later (i.e. it learns *one step of FIRE*, not the entire
    relaxation in one shot).  At inference the model is iterated
    ``K`` times to relax a Voronoi tile.

    Concretely: for each cell with ``n_frames`` captured FIRE frames,
    we emit ``n_frames - stride`` (input, target) pairs.  With the
    default 32 captured frames and ``stride=1`` that's 31 pairs/cell,
    covering the *entire* relaxation trajectory at every depth.

    Why this beats one-shot training (``fire_pos`` as target):

    - Each pair is a *small* displacement, much easier to fit than
      the cumulative Voronoi-to-FIRE-final jump.
    - The model is trained to be the identity near convergence (so
      it stays stable as inference iterations proceed).
    - Forces are equivariant; the model's output minus its input is
      the equivariant 3-vector ``∝ −∇E``.

    Parameters
    ----------
    h5_path
        Same HDF5 layout as :class:`TricorMLDataset` (each cell must
        have a ``fire_trajectory`` dataset).
    r_cut
        Radial cutoff for PBC graph construction (used in
        :func:`collate_cells`).
    stride
        Number of *captured* frames between input and target.  ``1``
        (default) predicts the next captured frame (~8 raw FIRE
        steps).  ``2``/``3`` predict bigger jumps but are harder to
        learn.
    max_box_side, regime_filter
        Same semantics as :class:`TricorMLDataset`.
    """

    def __init__(
        self,
        h5_path: str | Path,
        r_cut: float = 5.0,
        stride: int = 1,
        max_box_side: float | None = None,
        regime_filter: list[str] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.r_cut = float(r_cut)
        self.stride = int(stride)
        self.max_box_side = max_box_side
        self.regime_filter = regime_filter
        # Each entry: (cell_key, t)
        self._index: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("cell_"):
                    continue
                g = f[key]
                if "fire_trajectory" not in g:
                    continue
                if max_box_side is not None:
                    box = np.asarray(g.attrs.get("box_dim", [0., 0., 0.]))
                    if float(box.max()) > float(max_box_side) + 1e-3:
                        continue
                if regime_filter is not None:
                    if str(g.attrs.get("regime", "")) not in regime_filter:
                        continue
                n_frames = g["fire_trajectory"].shape[0]
                for t in range(n_frames - stride):
                    self._index.append((key, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        key, t = self._index[idx]
        with h5py.File(self.h5_path, "r") as f:
            g = f[key]
            traj = g["fire_trajectory"]
            voronoi_pos = np.asarray(traj[t], dtype=np.float32)
            target_pos = np.asarray(traj[t + self.stride], dtype=np.float32)
            species_idx = np.asarray(g["species_idx"], dtype=np.int64)
            grain_size = float(g.attrs["grain_size"])
            box_dim = np.asarray(g.attrs["box_dim"], dtype=np.float32)
            regime = str(g.attrs.get("regime", ""))
        return {
            "voronoi_pos": torch.from_numpy(voronoi_pos),
            "target_pos": torch.from_numpy(target_pos),
            "species_idx": torch.from_numpy(species_idx),
            "grain_size": float(grain_size),
            "box_dim": torch.from_numpy(box_dim),
            "regime": regime,
        }


def collate_cells(batch: list[dict], r_cut: float = 5.0) -> dict:
    """Collate a list of per-cell samples into a graph-batched tensor dict.

    The standard PyTorch DataLoader stacks tensors along a new batch
    axis — but our cells have variable atom counts and variable edge
    counts, so we need graph batching: stitch all atoms / edges into
    a single big graph with a per-atom ``batch`` vector mapping each
    atom to its source cell.

    Returns a dict with:
        positions      : (sum_N, 3)
        target_pos     : (sum_N, 3)
        species_idx    : (sum_N,)
        edge_index     : (2, sum_E)
        edge_vec       : (sum_E, 3)
        cond           : (B, 1)              ← grain_size per cell
        batch          : (sum_N,)            ← cell index per atom
        n_atoms_per_cell : (B,) int          ← for unbatching predictions
    """
    positions_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    species_list: list[torch.Tensor] = []
    edge_index_list: list[torch.Tensor] = []
    edge_vec_list: list[torch.Tensor] = []
    cond_list: list[float] = []
    batch_list: list[torch.Tensor] = []
    box_list: list[torch.Tensor] = []
    n_atoms_per_cell: list[int] = []

    atom_offset = 0
    for b, sample in enumerate(batch):
        pos = sample["voronoi_pos"]
        target = sample["target_pos"]
        species = sample["species_idx"]
        box = sample["box_dim"].tolist()
        n = pos.shape[0]

        ei, ev = build_pbc_graph(pos, box, r_cut)

        positions_list.append(pos)
        target_list.append(target)
        species_list.append(species)
        edge_index_list.append(ei + atom_offset)
        edge_vec_list.append(ev)
        cond_list.append(sample["grain_size"])
        batch_list.append(torch.full((n,), b, dtype=torch.long))
        box_list.append(sample["box_dim"].float())
        n_atoms_per_cell.append(n)
        atom_offset += n

    return {
        "positions": torch.cat(positions_list, dim=0),
        "target_pos": torch.cat(target_list, dim=0),
        "species_idx": torch.cat(species_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_vec": torch.cat(edge_vec_list, dim=0),
        "cond": torch.tensor(cond_list, dtype=torch.float32).unsqueeze(-1),
        "batch": torch.cat(batch_list, dim=0),
        # (B, 3) — per-cell orthorhombic box dims, used for PBC-correcting
        # the per-atom displacement target in displacement_loss.
        "box_dim": torch.stack(box_list, dim=0),
        "n_atoms_per_cell": torch.tensor(n_atoms_per_cell, dtype=torch.long),
    }


__all__ = [
    "TricorMLDataset",
    "build_pbc_graph",
    "build_pbc_graph_chunked",
    "collate_cells",
]
