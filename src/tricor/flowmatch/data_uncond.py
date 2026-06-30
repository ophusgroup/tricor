"""Data loading for unconditional flow matching training.

Simpler than the conditional dataset — no g2/ADF precomputation needed.
Just loads structures, samples noise, computes interpolation + target
velocity, and builds periodic graphs.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

import ase.io
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import OneHotEncoder

from graphite.nn import periodic_radius_graph

from .flow_utils import (
    periodic_interpolation,
    per_species_ot_assignment,
    sample_uniform_in_cell,
)

import lightning.pytorch as pl


def _load_atoms(source: Union[list, str, Path]) -> list:
    if isinstance(source, (str, Path)):
        data_dir = Path(source)
        files = sorted(
            list(data_dir.glob("*.extxyz"))
            + list(data_dir.glob("*.vasp"))
            + list(data_dir.glob("*.cif"))
            + list(data_dir.glob("*.xyz"))
        )
        if not files:
            raise FileNotFoundError(f"No structure files found in {data_dir}")
        return [ase.io.read(f) for f in files]
    return list(source)


class UncondFlowMatchDataset(Dataset):
    """Dataset for unconditional flow matching training.

    No spectral labels needed — just structures and noise.

    On each access:
      1. Sample flow time t ~ U(0, 1)
      2. Sample noise x0 ~ Uniform(cell)
      3. Optionally solve per-species OT assignment
      4. Compute interpolated positions x_t and target velocity (x1 - x0)
      5. Build periodic graph from x_t
      6. Apply random SO(3) rotation

    Args:
        structures: List of ASE Atoms or path to directory of structure files.
        cutoff: Graph construction cutoff (default 5.0 A).
        species: List of atomic numbers. If None, inferred from structures.
        dup: Duplicates per structure for noise diversity (default 128).
        use_ot: Per-species optimal transport assignment (default True).
    """

    def __init__(
        self,
        structures: Union[list, str, Path],
        cutoff: float = 5.0,
        species: Optional[list[int]] = None,
        dup: int = 128,
        use_ot: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.dup = dup
        self.use_ot = use_ot

        atoms_list = _load_atoms(structures)
        for atoms in atoms_list:
            atoms.wrap()

        if species is not None:
            unique_numbers = np.array(sorted(species))
        else:
            unique_numbers = np.unique(
                np.concatenate([np.unique(a.numbers) for a in atoms_list])
            )
        self.atom_encoder = OneHotEncoder(sparse_output=False)
        self.atom_encoder.fit(unique_numbers.reshape(-1, 1))
        self.num_species = len(unique_numbers)
        self.species_list = unique_numbers.tolist()

        # Store entries (no spectral labels, just positions + species)
        self._entries = []
        for atoms in atoms_list:
            z_onehot = self.atom_encoder.transform(atoms.numbers.reshape(-1, 1))
            sp_local = np.searchsorted(unique_numbers, atoms.numbers)

            self._entries.append({
                "z": torch.tensor(z_onehot, dtype=torch.float32),
                "positions": torch.tensor(atoms.positions, dtype=torch.float32),
                "cell": torch.tensor(atoms.cell.array, dtype=torch.float32),
                "sp_local": torch.tensor(sp_local, dtype=torch.long),
            })

        # Index map for duplicates
        self._index_map = []
        for i in range(len(self._entries)):
            for _ in range(dup):
                self._index_map.append(i)

    def len(self):
        return len(self._index_map)

    def get(self, idx):
        entry = self._entries[self._index_map[idx]]

        x1 = entry["positions"].clone()
        cell = entry["cell"].clone()
        z = entry["z"].clone()
        sp_local = entry["sp_local"].clone()
        N = x1.shape[0]

        # Sample noise
        x0 = sample_uniform_in_cell(N, cell)

        # Per-species OT assignment
        if self.use_ot:
            perm = per_species_ot_assignment(x0, x1, sp_local, cell)
            x0 = x0[perm]

        # Sample flow time
        t_val = torch.rand(1).item()
        t = torch.full((N, 1), t_val, dtype=torch.float32)

        # Periodic interpolation
        x_t, target_velocity = periodic_interpolation(x0, x1, t, cell)

        # Build periodic graph from x_t
        edge_index, edge_vec = periodic_radius_graph(x_t, self.cutoff, cell=cell)
        edge_len = edge_vec.norm(dim=-1, keepdim=True)
        edge_attr = torch.hstack([edge_vec, edge_len])

        # Random rotation (rotate positions, edges, and target velocity together)
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
        x_t = x_t @ R
        edge_attr_rot = edge_attr.clone()
        edge_attr_rot[:, :3] = edge_attr[:, :3] @ R
        target_velocity = target_velocity @ R

        return Data(
            z=z,
            pos=x_t,
            edge_index=edge_index,
            edge_attr=edge_attr_rot,
            t=t,
            target_velocity=target_velocity,
        )


class UncondFlowMatchDataModule(pl.LightningDataModule):
    """Lightning DataModule for unconditional flow matching.

    Args:
        structures: List of ASE Atoms or directory path.
        cutoff: Graph cutoff (default 5.0 A).
        species: Atomic numbers.
        dup: Noise replicas per structure (default 128).
        use_ot: Per-species OT assignment (default True).
        batch_size: Batch size (default 32).
        num_workers: DataLoader workers (default 0).
        val_fraction: Validation split (default 0.1).
    """

    def __init__(
        self,
        structures: Union[list, str, Path],
        cutoff: float = 5.0,
        species: Optional[list[int]] = None,
        dup: int = 128,
        use_ot: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["structures"])
        self._structures = structures
        self._ds_kwargs = dict(
            cutoff=cutoff, species=species, dup=dup, use_ot=use_ot,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

    def setup(self, stage=None):
        all_atoms = _load_atoms(self._structures)

        n_val = max(1, int(len(all_atoms) * self.val_fraction))
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(all_atoms))
        train_atoms = [all_atoms[i] for i in indices[:-n_val]]
        val_atoms = [all_atoms[i] for i in indices[-n_val:]]

        self.train_set = UncondFlowMatchDataset(train_atoms, **self._ds_kwargs)
        self.val_set = UncondFlowMatchDataset(val_atoms, **self._ds_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, shuffle=True,
            batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, shuffle=False,
            batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
