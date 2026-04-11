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

`generate()` covers the full disorder spectrum via `grain_size`, `crystalline_fraction`, `r_broadening`, and `phi_broadening`.  The `relative_density` on `from_atoms` should decrease for more disordered structures.  Recommended settings for Si:

```python
cases = [
    #                   density  steps  generate() kwargs
    ("liquid",          0.86, 80,  dict(grain_size=None, r_broadening=0.3, phi_broadening=25)),
    ("amorphous",       0.88, 150, dict(grain_size=4.0, crystalline_fraction=1.0, r_broadening=0.2, phi_broadening=12)),
    ("diamond_glass",   0.90, 150, dict(grain_size=8.0, crystalline_fraction=1.0, r_broadening=0.1, phi_broadening=3)),
    ("SRO",             0.92, 150, dict(grain_size=12.0, crystalline_fraction=0.5, r_broadening=0.08, phi_broadening=5)),
    ("mixed_50_50",     0.94, 150, dict(grain_size=18.0, crystalline_fraction=0.5, r_broadening=0.05, phi_broadening=3)),
    ("nanocrystalline", 0.96, 150, dict(grain_size=25.0, crystalline_fraction=1.0, r_broadening=0.03, phi_broadening=2)),
]

for name, density, steps, kw in cases:
    cell = tc.Supercell.from_atoms(
        atoms, (40, 40, 40),
        r_max=10, r_step=0.1, phi_num_bins=90,
        relative_density=density, rng_seed=42,
    )
    cell.generate(shell_target, num_steps=steps, **kw)
```

| Parameter | Controls |
|-----------|----------|
| `grain_size` | Crystallite diameter in Angstrom. `None` = no grains (liquid/amorphous). |
| `crystalline_fraction` | Fraction of volume filled by crystalline grains (0&ndash;1). |
| `r_broadening` | Radial disorder &sigma; in Angstrom. Larger = looser bond distances. |
| `phi_broadening` | Angular disorder &sigma; in degrees. Larger = looser bond angles. |
| `relative_density` | Density relative to crystal. Lower = fewer close-packed artifacts. |

## Core classes

- **`G3Distribution`** &mdash; measures rooted three-body angle/distance histograms from atomic structures
- **`CoordinationShellTarget`** &mdash; extracts first-shell coordination targets (bond lengths, angles, coordination numbers) from a reference crystal
- **`Supercell`** &mdash; generates and optimises disordered supercells

## Dependencies

- `numpy`, `matplotlib`, `ase`, `anywidget`
- Optional: `h5py`, `zarr` (for data I/O)
