# Examples

## Setup

```python
from ase.build import bulk
import tricor as tc

atoms = bulk('Si', 'diamond', a=5.431)
shell_target = tc.CoordinationShellTarget.from_atoms(
    atoms,
    phi_num_bins=90,
)
```

## Generate all disorder regimes

```python
cases = [
    ("liquid",          0.86, 80,  dict(
        grain_size=None, bond_weight=1.0, angle_weight=0.12)),
    ("amorphous",       0.88, 150, dict(
        grain_size=4.0, bond_weight=1.5, angle_weight=0.3)),
    ("SRO",             0.90, 150, dict(
        grain_size=8.0, bond_weight=2.0, angle_weight=1.0)),
    ("MRO",             0.92, 150, dict(
        grain_size=12.0, crystalline_fraction=0.5,
        bond_weight=2.0, angle_weight=0.6)),
    ("mixed",           0.94, 150, dict(
        grain_size=18.0, crystalline_fraction=0.5,
        bond_weight=2.5, angle_weight=1.0)),
    ("nanocrystalline", 0.96, 150, dict(
        grain_size=25.0, bond_weight=3.0, angle_weight=1.5)),
]

cells = {}
for name, density, steps, kw in cases:
    cell = tc.Supercell.from_atoms(
        atoms,
        cell_dim_angstroms=(40, 40, 40),
        r_max=10,
        r_step=0.1,
        phi_num_bins=90,
        relative_density=density,
        rng_seed=42,
    )
    cell.generate(
        shell_target,
        num_steps=steps,
        **kw,
    )
    cell.measure_g3()
    cells[name] = cell
```

## View g3 distributions

```python
for name, cell in cells.items():
    print(name)
    display(cell.plot_g3())
```

## Render movies

```python
for idx, (name, cell) in enumerate(cells.items(), start=1):
    cell.plot_structure(
        output=f"structure{idx:02d}_{name}.mp4",
        fps=60,
        duration=6.0,
        width=512,
        height=512,
    )
```

## SiC binary example

```python
atoms_sic = bulk('SiC', 'zincblende', a=4.36)
shell_target_sic = tc.CoordinationShellTarget.from_atoms(
    atoms_sic,
    phi_num_bins=90,
)

cell_sic = tc.Supercell.from_atoms(
    atoms_sic,
    cell_dim_angstroms=(30, 30, 30),
    r_max=10,
    r_step=0.2,
    phi_num_bins=90,
    relative_density=0.92,
    rng_seed=42,
)
cell_sic.generate(
    shell_target_sic,
    grain_size=12.0,
    crystalline_fraction=0.5,
    bond_weight=2.0,
    angle_weight=0.6,
)
cell_sic.measure_g3()
cell_sic.plot_g3()
```

## Using presets

```python
preset = tc.Supercell.PRESETS["nanocrystalline"].copy()
density = preset.pop("relative_density", 1.0)

cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=density,
    rng_seed=42,
)
cell.generate(shell_target, **preset)
cell.measure_g3()
cell.plot_g3()
```
