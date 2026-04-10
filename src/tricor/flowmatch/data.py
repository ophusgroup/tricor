"""Data loading for conditional flow matching training.

For each structure, precomputes g2(r) and ADF(phi) as conditioning labels
using tricor's DifferentiablePDFADF. On each access:
  1. Sample flow time t ~ U(0, 1)
  2. Sample noise x0 ~ Uniform(cell)
  3. Optionally solve per-species OT assignment to straighten paths
  4. Compute interpolated positions x_t and target velocity u = x1 - x0
  5. Build periodic graph from x_t
  6. Apply random SO(3) rotation
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

from tricor.differentiable_pdf import DifferentiablePDFADF
from .flow_utils import (
    periodic_interpolation,
    per_species_ot_assignment,
    sample_uniform_in_cell,
    wrap_periodic,
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


class FlowMatchDataset(Dataset):
    """Dataset for conditional flow matching on atomic structures.

    Precomputes g2 and ADF for each structure at init time. On each
    access, samples a fresh noise configuration, solves OT assignment,
    and returns the interpolated state with conditioning labels.

    Args:
        structures: List of ASE Atoms or path to directory.
        cutoff: Graph construction cutoff (default 5.0 A).
        r_max: PDF cutoff for conditioning labels (default 10.0 A).
        r_step: PDF radial bin width (default 0.05 A).
        phi_num_bins: ADF angular bins (default 90).
        sigma_r: PDF Gaussian bandwidth (default 0.15 A).
        sigma_phi: ADF Gaussian bandwidth (default 0.1 rad).
        species: List of atomic numbers.
        dup: Duplicates per structure for diverse noise samples.
        use_ot: Whether to use per-species OT assignment (default True).
    """

    def __init__(
        self,
        structures: Union[list, str, Path],
        cutoff: float = 5.0,
        r_max: float = 10.0,
        r_step: float = 0.05,
        phi_num_bins: int = 90,
        sigma_r: float = 0.15,
        sigma_phi: float = 0.1,
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

        # Species setup
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

        # Build PDF/ADF calculator
        calc = DifferentiablePDFADF(
            r_max=r_max,
            r_step=r_step,
            phi_num_bins=phi_num_bins,
            sigma_r=sigma_r,
            sigma_phi=sigma_phi,
            species=self.species_list,
        ).double()
        self.num_r = calc.num_r
        self.num_triplets = calc.num_triplets
        self.phi_num_bins = phi_num_bins

        # Precompute g2 and ADF for each structure
        self._entries = []
        for atoms in atoms_list:
            positions = torch.tensor(atoms.positions, dtype=torch.float64)
            sp_tensor = torch.tensor(atoms.numbers, dtype=torch.long)
            cell_tensor = torch.tensor(atoms.cell.array, dtype=torch.float64)

            with torch.no_grad():
                g2, adf = calc.compute(positions, sp_tensor, cell_tensor)

            # Normalize: divide by number of center atoms so magnitudes
            # are comparable across structures of different sizes
            n_atoms = len(atoms)
            g2 = g2 / max(n_atoms, 1)
            adf = adf / max(n_atoms, 1)

            # Composition fractions
            comp_frac = np.zeros(self.num_species, dtype=np.float32)
            for i, Z in enumerate(self.species_list):
                comp_frac[i] = (atoms.numbers == Z).sum() / n_atoms

            # One-hot encoding
            z_onehot = self.atom_encoder.transform(atoms.numbers.reshape(-1, 1))

            # Local species indices (for OT assignment)
            sp_local = np.searchsorted(unique_numbers, atoms.numbers)

            entry = {
                "z": torch.tensor(z_onehot, dtype=torch.float32),
                "pos": torch.tensor(atoms.positions, dtype=torch.float32),
                "cell": torch.tensor(atoms.cell.array, dtype=torch.float32),
                "sp_local": torch.tensor(sp_local, dtype=torch.long),
                "g2": g2.float(),
                "adf": adf.float(),
                "comp_frac": torch.tensor(comp_frac, dtype=torch.float32),
                "num_atoms": n_atoms,
            }
            self._entries.append(entry)

        # Index map for duplicates
        self._index_map = []
        for i in range(len(self._entries)):
            for _ in range(dup):
                self._index_map.append(i)

    def len(self):
        return len(self._index_map)

    def get(self, idx):
        entry = self._entries[self._index_map[idx]]

        x1 = entry["pos"].clone()  # data positions
        cell = entry["cell"].clone()
        z = entry["z"].clone()
        sp_local = entry["sp_local"].clone()
        N = x1.shape[0]

        # Sample noise (uniform in cell)
        x0 = sample_uniform_in_cell(N, cell)

        # Per-species OT assignment (reorder x0 to pair with x1)
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

        # Random rotation
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
        x_t = x_t @ R
        edge_attr_rot = edge_attr.clone()
        edge_attr_rot[:, :3] = edge_attr[:, :3] @ R
        target_velocity = target_velocity @ R

        data = Data(
            z=z,
            pos=x_t,
            edge_index=edge_index,
            edge_attr=edge_attr_rot,
            t=t,
            target_velocity=target_velocity,
            g2_target=entry["g2"].unsqueeze(0),       # (1, ns, ns, nr) — batched later
            adf_target=entry["adf"].unsqueeze(0),     # (1, nt, np) — batched later
            comp_frac=entry["comp_frac"].unsqueeze(0), # (1, ns)
            cell_matrix=cell,
        )
        return data


class FlowMatchDataModule(pl.LightningDataModule):
    """Lightning DataModule for flow matching training.

    Args:
        structures: List of ASE Atoms or directory path.
        cutoff: Graph cutoff (default 5.0 A).
        r_max: PDF cutoff for labels (default 10.0 A).
        r_step: Radial bin width (default 0.05 A).
        phi_num_bins: Angular bins (default 90).
        sigma_r: PDF bandwidth (default 0.15 A).
        sigma_phi: ADF bandwidth (default 0.1 rad).
        species: Atomic numbers.
        dup: Noise replicas per structure (default 128).
        use_ot: Per-species OT assignment (default True).
        batch_size: Batch size (default 16).
        num_workers: DataLoader workers (default 0).
        val_fraction: Validation split (default 0.1).
    """

    def __init__(
        self,
        structures: Union[list, str, Path],
        cutoff: float = 5.0,
        r_max: float = 10.0,
        r_step: float = 0.05,
        phi_num_bins: int = 90,
        sigma_r: float = 0.15,
        sigma_phi: float = 0.1,
        species: Optional[list[int]] = None,
        dup: int = 128,
        use_ot: bool = True,
        batch_size: int = 16,
        num_workers: int = 0,
        val_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["structures"])
        self._structures = structures
        self._ds_kwargs = dict(
            cutoff=cutoff, r_max=r_max, r_step=r_step,
            phi_num_bins=phi_num_bins, sigma_r=sigma_r, sigma_phi=sigma_phi,
            species=species, dup=dup, use_ot=use_ot,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

    def setup(self, stage=None):
        all_atoms = _load_atoms(self._structures)

        n_val = max(1, int(len(all_atoms) * self.val_fraction))
        n_train = len(all_atoms) - n_val

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(all_atoms))
        train_atoms = [all_atoms[i] for i in indices[:n_train]]
        val_atoms = [all_atoms[i] for i in indices[n_train:]]

        self.train_set = FlowMatchDataset(train_atoms, **self._ds_kwargs)
        self.val_set = FlowMatchDataset(val_atoms, **self._ds_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, shuffle=True,
            batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, shuffle=False,
            batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True,
        )
