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
    relative_density=0.98,
    rng_seed=42,
)

# Pick a regime:
cell.generate(shell_target, grain_size=15.0, crystalline_fraction=0.5)

# Measure and compare g3 distributions
cell.measure_g3(force=True, show_progress=True)
cell.plot_g3_compare()

# 3D rotating movie
cell.plot_structure(shell_target, output='structure.mp4')
```

## Structure generation regimes

`generate()` covers the full disorder spectrum via `grain_size`, `crystalline_fraction`, `r_broadening`, and `phi_broadening`:

| Regime | Example |
|--------|---------|
| Liquid | `generate(shell_target, grain_size=None, phi_broadening=180)` |
| Amorphous glass | `generate(shell_target, grain_size=None, r_broadening=0.1, phi_broadening=10)` |
| Diamond-like amorphous | `generate(shell_target, grain_size=None, r_broadening=0.1, phi_broadening=3)` |
| Short-range order | `generate(shell_target, grain_size=5.0, crystalline_fraction=0.3)` |
| Mixed crystalline-amorphous | `generate(shell_target, grain_size=15.0, crystalline_fraction=0.5)` |
| Nanocrystalline | `generate(shell_target, grain_size=15.0, crystalline_fraction=1.0)` |

## Core classes

- **`G3Distribution`** &mdash; measures rooted three-body angle/distance histograms from atomic structures
- **`CoordinationShellTarget`** &mdash; extracts first-shell coordination targets (bond lengths, angles, coordination numbers) from a reference crystal
- **`Supercell`** &mdash; generates and optimises disordered supercells

## Dependencies

- `numpy`, `matplotlib`, `ase`, `anywidget`
- Optional: `h5py`, `zarr` (for data I/O)
