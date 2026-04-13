# tricor

Generate disordered atomic supercells guided by three-body (g3) distributions, spanning the full spectrum from liquid to nanocrystalline.  Designed for machine-learning training data generation.

## Installation

```bash
uv sync
```

## Quick start

```python
from ase.build import bulk
import tricor as tc

# Reference crystal
atoms = bulk('Si', 'diamond', a=5.431)
shell_target = tc.CoordinationShellTarget.from_atoms(atoms, phi_num_bins=90)

# Create supercell and generate structure
cell = tc.Supercell.from_atoms(
    atoms,
    cell_dim_angstroms=(40, 40, 40),
    r_max=10,
    r_step=0.1,
    phi_num_bins=90,
    relative_density=0.92,
    rng_seed=42,
)
cell.generate(
    shell_target,
    grain_size=12.0,
    crystalline_fraction=0.5,
    bond_weight=2.0,
    angle_weight=0.6,
)

# View the g3 distribution
cell.plot_g3()

# 3D rotating movie
cell.plot_structure(shell_target, output='structure.mp4')
```

## Structure generation

`generate()` builds disordered structures by combining Voronoi grain
construction with spring-network relaxation.  The structure is controlled
by physical parameters:

| Parameter | Controls |
|-----------|----------|
| `grain_size` | Crystallite diameter in Angstrom. `None` = no grains (liquid/amorphous). |
| `crystalline_fraction` | Fraction of volume filled by crystalline grains (0&ndash;1). |
| `bond_weight` | Harmonic spring strength for bond distances. Larger = tighter. |
| `angle_weight` | Spring strength for bond angles. Larger = tighter. Near-zero = liquid. |
| `relative_density` | Density relative to crystal (set on `from_atoms`). Lower = fewer close-packed artifacts. |

### Recommended presets for Si

Available as `Supercell.PRESETS`:

```python
# Use a preset:
preset = tc.Supercell.PRESETS["SRO"]
cell = tc.Supercell.from_atoms(
    atoms, (40, 40, 40),
    relative_density=preset.pop("relative_density", 1.0),
    rng_seed=42,
)
cell.generate(shell_target, **preset)
```

| Regime | grain_size | crystalline_fraction | bond_weight | angle_weight | density |
|--------|-----------|---------------------|-------------|--------------|---------|
| liquid | None | &mdash; | 1.0 | 0.12 | 0.86 |
| amorphous | 4 | 1.0 | 1.5 | 0.3 | 0.88 |
| SRO | 8 | 1.0 | 2.0 | 1.0 | 0.90 |
| MRO | 12 | 0.5 | 2.0 | 0.6 | 0.92 |
| mixed | 18 | 0.5 | 2.5 | 1.0 | 0.94 |
| nanocrystalline | 25 | 1.0 | 3.0 | 1.5 | 0.96 |

### Optional: target g3 for comparison

To compare against a target distribution, create one explicitly and pass
it as the initial distribution:

```python
dist = tc.G3Distribution(atoms)
dist.measure_g3(r_max=10, r_step=0.1, phi_num_bins=90)
target = dist.target_g3(
    target_r_min=5.0,
    target_r_max=8.0,
    r_sigma=0.05,
    phi_sigma_deg=3.0,
)

cell = tc.Supercell(target, cell_dim_angstroms=(40, 40, 40), relative_density=0.92)
cell.generate(shell_target, grain_size=12.0, crystalline_fraction=0.5)
cell.plot_g3_compare()  # side-by-side comparison
```

## Core classes

- **`G3Distribution`** &mdash; measures rooted three-body angle/distance histograms from atomic structures
- **`CoordinationShellTarget`** &mdash; extracts first-shell coordination targets (bond lengths, angles, coordination numbers) from a reference crystal
- **`Supercell`** &mdash; generates and optimises disordered supercells

## Dependencies

- `numpy`, `matplotlib`, `ase`, `anywidget`
- Optional: `h5py`, `zarr` (for data I/O)
