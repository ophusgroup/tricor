"""Data loading for the MACE+wall pilot — fork of tricor.relaxml.data_shelltgt.

Reads MACE+wall trajectory NPZs produced by
``scripts/macerelax/generation/generate_mace_trajectories.py``, plus shell_target arrays
appended by ``scripts/macerelax/generation/add_shell_target_to_pilot.py``.

The MACE pilot uses a 6-dimensional global conditioning vector instead
of the relaxml-baseline 9-dim spring-weight vector — the spring fields
don't apply under MACE+wall. See WEIGHT_FEATURE_KEYS below for the new
field list and MACE_RELAX_PILOT.md for the rationale.

Per-graph shell_target arrays are still consumed via the same
``ShellTargetData`` subclass and the ``ShellTargetEncoder`` module is
reused unchanged from the relaxml model — the conditioning path for
"this composition's reference equilibrium structure" is composition-
intrinsic and works identically for MACE.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from tricor.flowmatch.flow_utils import (
    periodic_radius_graph_cell_list,
    periodic_radius_graph_chunked,
)


def _periodic_graph(
    pos: torch.Tensor, cutoff: float, cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast periodic radius graph: O(N) cell-list when applicable, else O(N^2) chunked.

    The cell-list path requires an orthogonal cell with each axis >= 3*cutoff.
    Our 50 A Si cells satisfy this; the fallback is for robustness.
    """
    try:
        return periodic_radius_graph_cell_list(pos, cutoff, cell)
    except ValueError:
        return periodic_radius_graph_chunked(pos, cutoff, cell=cell)


# Order and normalization of the global weight vector fed to the model.
# Each feature is divided by its scale before being embedded so the
# network sees values in roughly [0, ~1.5].
#
# MACE pilot conditioning (6 dims), motivated by what the GNN can't
# easily derive from local positions alone (see MACE_RELAX_PILOT.md):
#   - grain_size:           grain-scale morphology; receptive field can't see grains
#   - log_num_grains:       density of grain boundaries
#   - crystalline_fraction: distinguishes mixed-phase regimes
#   - rel_density:          intended density signal (vs deriving from positions)
#   - wall_global_min:      physical floor for pair distances under MACE+wall
#   - log_fmax_initial:     strain budget at the start of the trajectory
#
# Dropped vs the shell_relax baseline: bond_weight, angle_weight,
# repulsion_weight, hard_core_scale, nonbond_push_scale,
# displacement_sigma — these were shell_relax generator knobs and don't
# apply to MACE+wall trajectories (the generator is fixed; no per-traj
# behavior knobs to expose to the model).
WEIGHT_FEATURE_KEYS: tuple[str, ...] = (
    "grain_size",
    "log_num_grains",            # computed as log1p(num_grains)
    "crystalline_fraction",
    "rel_density",
    "wall_global_min",
    "log_fmax_initial",          # computed as log10(max(fmax_initial, 1e-3))
)

WEIGHT_FEATURE_SCALES: dict[str, float] = {
    "grain_size":          25.0,    # Å; covers 0 (liquid) → 30 (crystalline_30)
    "log_num_grains":       5.0,    # log1p(8 grains) ≈ 2.2 → /5 ≈ 0.44
    "crystalline_fraction": 1.0,    # already in [0, 1]
    "rel_density":          1.0,    # 0.88-1.00 in the pilot
    "wall_global_min":      2.0,    # Si-O ~1.6 → /2 ≈ 0.8; ionic systems larger
    "log_fmax_initial":     2.0,    # log10(fmax) typically 0-2 (1-100 eV/Å)
}

NUM_WEIGHT_FEATURES: int = len(WEIGHT_FEATURE_KEYS)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _min_image_displacement(
    pos_from: torch.Tensor, pos_to: torch.Tensor, cell: torch.Tensor,
) -> torch.Tensor:
    """Shortest-vector displacement from pos_from to pos_to under PBC.

    Matches the convention used in tricor.differentiable_pdf_fast.
    """
    inv_cell = torch.linalg.inv(cell)
    delta = pos_to - pos_from
    delta_frac = delta @ inv_cell.T
    delta_frac = delta_frac - torch.round(delta_frac)
    return delta_frac @ cell


def _weight_vector_from_row(row: dict, num_grains: int | None = None) -> np.ndarray:
    """Build the normalized (NUM_WEIGHT_FEATURES,) global vector from a
    manifest row produced by scripts/macerelax/generation/generate_mace_trajectories.py."""
    if num_grains is None:
        num_grains = int(float(row["num_grains"]))
    fmax_initial = float(row["fmax_initial"])
    vals = {
        "grain_size":           float(row["grain_size"]),
        "log_num_grains":       float(np.log1p(num_grains)),
        "crystalline_fraction": float(row["crystalline_fraction"]),
        "rel_density":          float(row["rel_density"]),
        "wall_global_min":      float(row["wall_global_min"]),
        "log_fmax_initial":     float(np.log10(max(fmax_initial, 1e-3))),
    }
    out = np.array([
        vals[k] / WEIGHT_FEATURE_SCALES[k] for k in WEIGHT_FEATURE_KEYS
    ], dtype=np.float32)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class ShellTargetData(Data):
    """Data subclass that knows how to batch per-pair / per-triplet attrs.

    PyG's default __cat_dim__ is 0 for tensors and __inc__ is 0 for non
    edge_index keys.  For ``shell_pair_batch`` and ``shell_trip_batch``
    we want PyG to *increment* the value by 1 per graph when batching
    (so graph-i's pairs end up labeled with i), so that the model can
    use the index to scatter-sum into per-graph slots.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("shell_pair_batch", "shell_trip_batch"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("shell_pair_batch", "shell_trip_batch"):
            return 1
        return super().__inc__(key, value, *args, **kwargs)


class _TrajectoryCache:
    """LRU-bounded per-worker cache of trajectory .npz contents.

    With ~6k atoms × ~41 snapshots × 12 bytes ≈ 3 MB per trajectory plus
    a few MB of best_positions/etc., each cache entry is ~5–10 MB.  With
    many workers and shuffled pair access patterns, every worker
    eventually touches most trajectories per epoch, so an unbounded
    cache balloons to GB × workers × ranks (observed: ~880 GB on the
    10k-trajectory big dataset before we added this cap).

    ``max_entries=None`` falls back to the legacy unbounded behavior.
    """

    def __init__(self, data_root: Path, max_entries: Optional[int] = 500) -> None:
        self._data_root = data_root
        self._max_entries = max_entries
        # OrderedDict for O(1) LRU eviction: move_to_end on access, popitem
        # on overflow.
        self._cache: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, filename: str) -> dict:
        cached = self._cache.get(filename)
        if cached is not None:
            self._cache.move_to_end(filename)
            return cached
        with np.load(self._data_root / filename) as npz:
            files = set(npz.files)
            if "shell_pair_species" not in files:
                raise KeyError(
                    f"{filename} has no shell_target arrays.  Run "
                    f"scripts/relaxml/add_shell_target_to_npz.py against "
                    f"the source dir to retrofit existing trajectories, "
                    f"or regenerate."
                )
            entry = {
                "positions": np.asarray(npz["positions"], dtype=np.float32),
                "cell": np.asarray(npz["cell"], dtype=np.float32),
                "species": np.asarray(npz["species_numbers"], dtype=np.int64),
                # Per-trajectory shell_target — same for every (snap_i, snap_j)
                # pair drawn from this trajectory.
                "shell_pair_species":     np.asarray(npz["shell_pair_species"],     dtype=np.int64),
                "shell_pair_features":    np.asarray(npz["shell_pair_features"],    dtype=np.float32),
                "shell_triplet_species":  np.asarray(npz["shell_triplet_species"],  dtype=np.int64),
                "shell_triplet_features": np.asarray(npz["shell_triplet_features"], dtype=np.float32),
            }
        self._cache[filename] = entry
        if self._max_entries is not None:
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)  # evict LRU
        return entry


class RelaxMLDataset(Dataset):
    """Consecutive-snapshot pairs from the surrogate trajectory corpus.

    Parameters
    ----------
    manifest_rows
        Subset of the manifest CSV rows for this split.
    data_root
        Directory containing the .npz files (usually the same directory
        as the manifest).
    cutoff
        Graph construction cutoff in Å.
    k_stride_snapshots
        Distance between the two snapshots of a training pair, in
        snapshot units.  ``1`` means consecutive snapshots (= 5 tricor
        steps with the default trajectory stride of 5).
    rotate
        Apply a random SO(3) rotation to positions, edges and target
        during ``get`` (on for training, off for validation).

    Notes
    -----
    Species are read per trajectory from each .npz's ``species_numbers``
    field and emitted as a long tensor of atomic numbers; the model
    embeds them via nn.Embedding(max_z, ...).
    """

    def __init__(
        self,
        manifest_rows: list[dict],
        data_root: Union[str, Path],
        cutoff: float = 5.0,
        k_stride_snapshots: int = 1,
        rotate: bool = True,
        cache_max_entries: Optional[int] = 500,
    ) -> None:
        super().__init__()
        if not manifest_rows:
            raise ValueError("manifest_rows is empty")
        self.rows = list(manifest_rows)
        self.data_root = Path(data_root)
        self.cutoff = float(cutoff)
        self.k_stride_snapshots = int(k_stride_snapshots)
        self.rotate = bool(rotate)
        self.cache_max_entries = cache_max_entries

        # Precompute per-trajectory weight vectors (cheap, read from manifest).
        self._weight_vectors: list[np.ndarray] = [
            _weight_vector_from_row(r) for r in self.rows
        ]

        # Peek into each trajectory to read its snapshot count once.  We
        # don't keep positions in memory at construction time; the cache
        # below fills on demand.
        self._num_snapshots: list[int] = []
        for row in self.rows:
            with np.load(self.data_root / row["filename"]) as npz:
                self._num_snapshots.append(int(npz["positions"].shape[0]))

        # Build the flat pair index.
        self._pair_index: list[tuple[int, int]] = []
        for i, S in enumerate(self._num_snapshots):
            last_valid_start = S - 1 - self.k_stride_snapshots
            if last_valid_start < 0:
                continue
            for s in range(last_valid_start + 1):
                self._pair_index.append((i, s))

        if not self._pair_index:
            raise ValueError(
                f"No valid (snapshot_i, snapshot_{{i+{self.k_stride_snapshots}}}) "
                f"pairs found — k_stride_snapshots={self.k_stride_snapshots} is "
                f"too large for the available trajectories."
            )

        self._cache = _TrajectoryCache(
            self.data_root, max_entries=self.cache_max_entries,
        )

    def len(self) -> int:
        return len(self._pair_index)

    def get(self, idx: int) -> Data:
        traj_idx, snap_start = self._pair_index[idx]
        row = self.rows[traj_idx]
        entry = self._cache.get(row["filename"])

        snap_end = snap_start + self.k_stride_snapshots
        pos_i = torch.tensor(entry["positions"][snap_start], dtype=torch.float32)
        pos_j = torch.tensor(entry["positions"][snap_end], dtype=torch.float32)
        cell = torch.tensor(entry["cell"], dtype=torch.float32)
        species = entry["species"]

        # Node features: raw atomic numbers (long).  The model embeds
        # them via nn.Embedding(max_z, node_dim).
        z = torch.tensor(species, dtype=torch.long)

        # Target: min-image displacement from current to next.
        target = _min_image_displacement(pos_i, pos_j, cell)

        # Periodic graph built on the *current* state (what the model
        # sees at inference).  Edge features mirror flowmatch:
        # [dx, dy, dz, r] with dx/dy/dz the minimum-image vector.
        edge_index, edge_vec = _periodic_graph(pos_i, self.cutoff, cell)
        edge_len = edge_vec.norm(dim=-1, keepdim=True)
        edge_attr = torch.hstack([edge_vec, edge_len])

        # Global conditioning vector — broadcast inside the model.
        w = torch.tensor(self._weight_vectors[traj_idx], dtype=torch.float32)

        # Random SO(3) rotation for equivariance augmentation.  Rotate
        # positions, edge displacement vectors, and target together; the
        # edge length is invariant so we don't touch it.
        if self.rotate:
            R = torch.tensor(
                Rotation.random().as_matrix(), dtype=torch.float32,
            )
            pos = pos_i @ R
            edge_attr_rot = edge_attr.clone()
            edge_attr_rot[:, :3] = edge_attr[:, :3] @ R
            target = target @ R
        else:
            pos = pos_i
            edge_attr_rot = edge_attr

        # Shell_target arrays: same for every snapshot pair from this
        # trajectory (computed from the reference cell at generation
        # time).  Per-trajectory cached.
        sps = torch.tensor(entry["shell_pair_species"], dtype=torch.long)
        spf = torch.tensor(entry["shell_pair_features"], dtype=torch.float32)
        sts = torch.tensor(entry["shell_triplet_species"], dtype=torch.long)
        stf = torch.tensor(entry["shell_triplet_features"], dtype=torch.float32)

        return ShellTargetData(
            z=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr_rot,
            w=w.unsqueeze(0),                # (1, NUM_WEIGHT_FEATURES)
            target_displacement=target,
            shell_pair_species=sps,
            shell_pair_features=spf,
            shell_pair_batch=torch.zeros(sps.shape[0], dtype=torch.long),
            shell_trip_species=sts,
            shell_trip_features=stf,
            shell_trip_batch=torch.zeros(sts.shape[0], dtype=torch.long),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Lightning DataModule
# ──────────────────────────────────────────────────────────────────────────────


class RelaxMLDataModule(pl.LightningDataModule):
    """Lightning DataModule that splits the manifest by trajectory.

    Split is by config (not by pair) so no trajectory's snapshots end up
    in both train and val — prevents trivial memorization via adjacent-
    snapshot leakage.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        cutoff: float = 5.0,
        k_stride_snapshots: int = 1,
        rotate: bool = True,
        batch_size: int = 4,
        num_workers: int = 0,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        # Max trajectories held in each worker's _TrajectoryCache.  At
        # ~5–10 MB per entry, ``cache_max_entries=500`` × NUM_WORKERS *
        # len(GPU_IDS) keeps the total dataloader cache budget under ~50
        # GB on a 16-worker DDP setup.  Set None for unbounded (legacy)
        # — fine for small datasets but blew up to ~880 GB on big_v1.
        cache_max_entries: Optional[int] = 500,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["manifest_path"])
        self.manifest_path = Path(manifest_path)
        self.data_root = self.manifest_path.parent
        self._ds_kwargs = dict(
            cutoff=cutoff,
            k_stride_snapshots=k_stride_snapshots,
            cache_max_entries=cache_max_entries,
        )
        self.rotate = bool(rotate)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)

        self.train_set: Optional[RelaxMLDataset] = None
        self.val_set: Optional[RelaxMLDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        rows = _load_manifest(self.manifest_path)
        if not rows:
            raise ValueError(f"Empty manifest at {self.manifest_path}")
        n_total = len(rows)
        n_val = max(1, int(round(n_total * self.val_fraction)))

        rng = np.random.default_rng(self.split_seed)
        order = rng.permutation(n_total)
        train_rows = [rows[i] for i in order[:-n_val]]
        val_rows = [rows[i] for i in order[-n_val:]]

        self.train_set = RelaxMLDataset(
            train_rows, self.data_root, rotate=self.rotate, **self._ds_kwargs,
        )
        self.val_set = RelaxMLDataset(
            val_rows, self.data_root, rotate=False, **self._ds_kwargs,
        )

    def _loader_kwargs(self) -> dict:
        """Shared DataLoader kwargs.  ``persistent_workers`` avoids
        re-spawning workers between epochs, and ``prefetch_factor``
        buffers batches ahead of the GPU so brief loader stalls don't
        starve compute.  Both require ``num_workers > 0``."""
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pinned (page-locked) host memory only speeds up the
            # host→GPU copy; it is wasted (and warns) on a CPU-only run.
            pin_memory=torch.cuda.is_available(),
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set, shuffle=True, **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set, shuffle=False, **self._loader_kwargs(),
        )
