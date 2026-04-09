"""Data loading for GLASS score model training.

Loads ASE Atoms objects, builds periodic graphs, and applies VE-SDE forward
noise for denoising score matching training. Follows the data pipeline in
graphite/notebooks/amorph-gen/lit/datamodules/structure_xanes.py with
adaptations for GLASS (Guo & Schwalbe-Koda, arXiv:2603.23210):

  - Supports loading from directories of structure files (extxyz, POSCAR, CIF)
    or from a list of ASE Atoms objects directly
  - VE-SDE noise with k=0.8 A (GLASS default)
  - GLASS trained on small cells (216-500 atoms) with true PBC; if your
    structures are large, regenerate smaller cells rather than carving
    sub-volumes (sub-volumes lack correct PBC at their boundaries)
  - 128 duplicates per structure for decorrelated noise (GLASS Sec. S1.4)
  - Random SO(3) rotation augmentation
  - 90/10 train/val split
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
from graphite.diffusion import VarianceExplodingDiffuser

import lightning.pytorch as pl


def _load_atoms(source: Union[list, str, Path]) -> list[ase.Atoms]:
    """Load structures from a list or directory."""
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


class StructureDataset(Dataset):
    """Dataset of atomic structures for score model training.

    Each __getitem__ call returns a PyG Data object with forward-diffused
    positions and the corresponding noise, ready for denoising score matching.

    Args:
        structures: Either a list of ASE Atoms objects, or a path to a
            directory containing structure files.
        cutoff: Neighbor cutoff for periodic graph construction (default 5.0 A).
        k: Maximum noise level for VE-SDE (default 0.8 A).
        dup: Number of duplicates per structure for decorrelated noise
            (default 128).
        species: Optional list of atomic numbers. If None, inferred from
            the structures.
    """

    def __init__(
        self,
        structures: Union[list, str, Path],
        cutoff: float = 5.0,
        k: float = 0.8,
        dup: int = 128,
        species: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.diffuser = VarianceExplodingDiffuser(k=k)
        self.dup = dup

        atoms_list = _load_atoms(structures)
        for atoms in atoms_list:
            atoms.wrap()

        # One-hot encoder for atom types
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

        # Store the raw ASE Atoms (sub-volume extraction happens in get())
        # and duplicate indices for decorrelated noise
        self._atoms_list = atoms_list
        self._index_map = []
        for struct_idx in range(len(atoms_list)):
            for _ in range(dup):
                self._index_map.append(struct_idx)

    def len(self):
        return len(self._index_map)

    def get(self, idx):
        struct_idx = self._index_map[idx]
        atoms = self._atoms_list[struct_idx]

        # Convert to PyG Data
        z = self.atom_encoder.transform(atoms.numbers.reshape(-1, 1))
        data = Data(
            z=torch.tensor(z, dtype=torch.float),
            pos=torch.tensor(atoms.positions, dtype=torch.float),
            cell=np.array(atoms.cell),
            num_atoms=len(atoms),
        )

        data = self._diffuse_pos(data)
        data = self._atomic_graph(data)
        data = self._random_rotate(data)
        return data

    def _diffuse_pos(self, data):
        """Apply VE-SDE forward noise: x_tilde = x + sigma*eps, sigma = k*t."""
        t = torch.rand(1).clip(self.diffuser.t_min, self.diffuser.t_max)
        data.t = t.expand(data.pos.size(0), 1)
        data.pos, data.eps_r = self.diffuser.forward_noise(data.pos, data.t)
        data.sigma_r = self.diffuser.sigma(data.t)
        return data

    def _atomic_graph(self, data):
        """Build periodic graph with minimum-image convention."""
        cell = torch.tensor(data.cell, dtype=torch.float)
        data.edge_index, edge_vec = periodic_radius_graph(
            data.pos, self.cutoff, cell=cell
        )
        data.edge_len = edge_vec.norm(dim=-1, keepdim=True)
        data.edge_attr = torch.hstack([edge_vec, data.edge_len])
        return data

    def _random_rotate(self, data):
        """Apply random SO(3) rotation for augmentation."""
        R = torch.tensor(
            Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device
        )
        data.pos = data.pos @ R
        data.edge_attr[:, :3] = data.edge_attr[:, :3] @ R
        data.eps_r = data.eps_r @ R
        return data


class StructureDataModule(pl.LightningDataModule):
    """Lightning DataModule for GLASS training.

    Args:
        structures: List of ASE Atoms or path to structure file directory.
        cutoff: Neighbor cutoff radius (default 5.0 A).
        k: VE-SDE noise scale (default 0.8 A).
        dup: Noise replicas per structure (default 128).
        species: Optional list of atomic numbers.
        batch_size: Batch size (default 32, GLASS Sec. S1.4).
        num_workers: DataLoader workers (default 8, GLASS Sec. S1.4).
        val_fraction: Fraction of structures held out for validation (default 0.1).
    """

    def __init__(
        self,
        structures: Union[list, str, Path],
        cutoff: float = 5.0,
        k: float = 0.8,
        dup: int = 128,
        species: Optional[list[int]] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["structures"])
        self.structures = structures
        self.cutoff = cutoff
        self.k = k
        self.dup = dup
        self.species = species
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

    def setup(self, stage=None):
        all_atoms = _load_atoms(self.structures)

        # Train/val split on structures (not on noise replicas)
        n_val = max(1, int(len(all_atoms) * self.val_fraction))
        n_train = len(all_atoms) - n_val

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(all_atoms))
        train_atoms = [all_atoms[i] for i in indices[:n_train]]
        val_atoms = [all_atoms[i] for i in indices[n_train:]]

        self.train_set = StructureDataset(
            train_atoms, self.cutoff, self.k, self.dup, self.species,
        )
        self.val_set = StructureDataset(
            val_atoms, self.cutoff, self.k, self.dup, self.species,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
