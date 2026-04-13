# Quick Start

## Installation

```bash
git clone https://github.com/cophus/tricor.git
cd tricor
uv sync
```

## Single-species example (Si)

```python
from ase.build import bulk
import tricor as tc

# 1. Reference crystal
atoms = bulk('Si', 'diamond', a=5.431)
shell_target = tc.CoordinationShellTarget.from_atoms(atoms, phi_num_bins=90)

# 2. Create supercell
cell = tc.Supercell.from_atoms(
    atoms,
    cell_dim_angstroms=(40, 40, 40),
    r_max=10,
    r_step=0.1,
    phi_num_bins=90,
    relative_density=0.92,
    rng_seed=42,
)

# 3. Generate structure (MRO example)
cell.generate(
    shell_target,
    grain_size=12.0,
    crystalline_fraction=0.5,
    bond_weight=2.0,
    angle_weight=0.6,
    num_steps=150,
)

# 4. Measure and view g3
cell.measure_g3()
cell.plot_g3()

# 5. Interactive 3D viewer
cell.view_structure()

# 6. Export movie
cell.plot_structure(output='structure.mp4')
```

## Binary example (SiC)

```python
atoms = bulk('SiC', 'zincblende', a=4.36)
shell_target = tc.CoordinationShellTarget.from_atoms(atoms, phi_num_bins=90)

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

cell.measure_g3()
cell.plot_g3()  # browse Si-Si-Si, Si-C-Si, C-C-C, ... triplets
```

## Using presets

Recommended parameter sets for Si are available as `Supercell.PRESETS`:

```python
import tricor as tc

# View available presets
for name, params in tc.Supercell.PRESETS.items():
    print(f"{name}: {params}")
```

```python
# Use a preset
preset = tc.Supercell.PRESETS["nanocrystalline"].copy()
density = preset.pop("relative_density", 1.0)

cell = tc.Supercell.from_atoms(atoms, (40, 40, 40), relative_density=density, rng_seed=42)
cell.generate(shell_target, **preset)
```
