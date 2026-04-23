# tricor

[![Documentation Status](https://readthedocs.org/projects/tricor/badge/?version=latest)](https://tricor.readthedocs.io/en/latest/?badge=latest)

Generate disordered atomic supercells guided by three-body (g3) distributions, spanning the full spectrum from liquid to nanocrystalline.  Designed for machine-learning training data generation.

**[Documentation](https://tricor.readthedocs.io)**

## Installation

```bash
uv sync
```

## Quick start - Si

```python
from ase.build import bulk
import tricor as tc

atoms = bulk('Si', 'diamond', a=5.431)
shell_target = tc.CoordinationShellTarget.from_atoms(
    atoms,
    phi_num_bins=90,
)

cell = tc.Supercell.from_atoms(
    atoms,
    cell_dim_angstroms=(40, 40, 40),
    r_max=10,
    r_step=0.1,
    phi_num_bins=90,
    relative_density=0.96,
    rng_seed=42,
)
cell.generate(
    shell_target,
    num_steps=150,
    grain_size=13.0,
    bond_weight=1.9,
    angle_weight=0.9,
    repulsion_weight=2.5,
    hard_core_scale=0.95,
    nonbond_push_scale=0.7,
    displacement_sigma=0.04,
)

cell.measure_g3()
cell.plot_g3()

cell.plot_structure(output='structure.mp4')
```

## Quick start - SiC (binary)

```python
atoms = bulk('SiC', 'zincblende', a=4.36)
shell_target = tc.CoordinationShellTarget.from_atoms(
    atoms,
    phi_num_bins=90,
)

cell = tc.Supercell.from_atoms(
    atoms,
    cell_dim_angstroms=(40, 40, 40),
    r_max=10,
    r_step=0.1,
    phi_num_bins=90,
    relative_density=0.96,
    rng_seed=42,
)
cell.generate(
    shell_target,
    num_steps=150,
    grain_size=13.0,
    bond_weight=1.9,
    angle_weight=0.9,
    repulsion_weight=2.5,
    hard_core_scale=0.95,
    nonbond_push_scale=0.7,
    displacement_sigma=0.04,
)

cell.measure_g3()
cell.plot_g3()  # browse Si | Si Si, C | C C, etc.
```

## Structure generation

`generate()` builds disordered structures by combining Voronoi grain
construction with spring-network relaxation.

| Parameter | Controls |
|-----------|----------|
| `grain_size` | Crystallite diameter (A). `None` = no grains (liquid). |
| `bond_weight` | Spring strength for bond distances. Larger = tighter. |
| `angle_weight` | Spring strength for bond angles. Larger = tighter. |
| `hard_core_scale` | Scales the minimum bond distance wall. < 1 = softer (liquid). |
| `nonbond_push_scale` | Scales the non-bonded clearance distance. < 1 = broader 2nd shell. |
| `displacement_sigma` | Gaussian jitter on grain atoms (A). Broadens crystalline peaks. |
| `relative_density` | Density relative to crystal (set on `from_atoms`). |

### Recommended presets for Si

Available as `Supercell.PRESETS`:

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    rng_seed=42,
)
cell.generate(shell_target, **tc.Supercell.PRESETS["MRO"])
```

| Regime | grain_size | bond_wt | angle_wt | rep_wt | hc_scale | nbp_scale | disp_sigma |
|--------|-----------|---------|----------|--------|----------|-----------|------------|
| liquid | None | 0.4 | 0.5 | 0.5 | 0.75 | 0.7 | - |
| amorphous | 6 | 1.2 | 0.6 | 1.5 | 0.9 | 0.5 | 0.08 |
| SRO | 10 | 2.2 | 1.0 | 2.0 | 0.95 | 0.6 | 0.04 |
| MRO | 13 | 1.9 | 0.9 | 2.5 | 0.95 | 0.7 | 0.04 |
| MRO_more | 18 | 2.0 | 1.0 | default | 0.95 | 0.9 | 0.04 |
| nanocrystalline_10 | 15 | 2.8 | 1.3 | default | default | default | 0.02 |
| nanocrystalline_20 | 20 | 3.0 | 1.5 | default | default | default | 0.02 |

All presets use `relative_density=0.96`. Abbreviations: `rep_wt` = `repulsion_weight`, `hc_scale` = `hard_core_scale`, `nbp_scale` = `nonbond_push_scale`.

### Optional: target g3 for comparison

To compare the supercell against a target distribution, create one
explicitly and pass it as the initial distribution:

```python
dist = tc.G3Distribution(atoms)
dist.measure_g3(
    r_max=10,
    r_step=0.1,
    phi_num_bins=90,
)
target = dist.target_g3(
    target_r_min=5.0,
    target_r_max=8.0,
    r_sigma=0.05,
    phi_sigma_deg=3.0,
)

cell = tc.Supercell(
    target,
    cell_dim_angstroms=(40, 40, 40),
    relative_density=0.96,
)
cell.generate(
    shell_target,
    grain_size=13.0,
    bond_weight=1.9,
    angle_weight=0.9,
    repulsion_weight=2.5,
    hard_core_scale=0.95,
    nonbond_push_scale=0.7,
    displacement_sigma=0.04,
)
cell.measure_g3()
cell.plot_g3_compare()
```

## Core classes

- **`G3Distribution`** - measures rooted three-body angle/distance histograms from atomic structures
- **`CoordinationShellTarget`** - extracts first-shell coordination targets (bond lengths, angles, coordination numbers) from a reference crystal
- **`Supercell`** - generates and optimises disordered supercells

## Dependencies

- `numpy`, `matplotlib`, `ase`, `anywidget`
- Optional: `h5py`, `zarr` (for data I/O)
