# Structure Generation

## Overview

`generate()` builds disordered structures in two steps:

1. **Grain construction**  -  the periodic box is Voronoi-tessellated. A fraction of cells are filled with randomly oriented crystal; the rest with random atom positions.

2. **Shell relaxation**  -  all atoms are simultaneously relaxed via bond, angle, and repulsion springs. Grain-interior atoms are frozen to preserve crystalline order.

See [Algorithm](algorithm.md) for the mathematical details.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grain_size` | float or None | None | Crystallite diameter (Angstrom). `None` = no grains. |
| `crystalline_fraction` | float | 1.0 | Volume fraction filled by crystalline grains (0 to 1). |
| `bond_weight` | float | 1.0 | Harmonic spring strength for bond distances. |
| `angle_weight` | float | 0.5 | Spring strength for bond angles. |
| `displacement_sigma` | float | 0.0 | Thermal jitter on grain atoms (Angstrom). |
| `num_steps` | int | 200 | Relaxation iterations. |
| `relative_density` | float | 1.0 | Set on `from_atoms`. Density relative to crystal. |

## Disorder regimes

### Liquid

No grains. Weak angle constraint. Only nearest-neighbour distances enforced.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=0.86,
    rng_seed=42,
)
cell.generate(
    shell_target,
    grain_size=None,
    bond_weight=1.0,
    angle_weight=0.12,
    num_steps=80,
)
```

### Amorphous

Very small grains with moderate constraints. Short-range order in the first shell.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=0.88,
    rng_seed=42,
)
cell.generate(
    shell_target,
    grain_size=4.0,
    bond_weight=1.5,
    angle_weight=0.3,
)
```

### Short-range order (SRO)

Small crystalline grains with strong angular constraints. Order extends to the 2nd nearest-neighbour shell.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=0.90,
    rng_seed=42,
)
cell.generate(
    shell_target,
    grain_size=8.0,
    bond_weight=2.0,
    angle_weight=1.0,
)
```

### Medium-range order (MRO)

Larger grains, partially filling the box. Crystalline and amorphous regions coexist.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
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
```

### Mixed crystalline-amorphous

Large grains filling half the box. Clear crystalline domains with amorphous boundaries.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=0.94,
    rng_seed=42,
)
cell.generate(
    shell_target,
    grain_size=18.0,
    crystalline_fraction=0.5,
    bond_weight=2.5,
    angle_weight=1.0,
)
```

### Nanocrystalline

Large grains filling the entire box. Thin disordered grain boundaries.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=0.96,
    rng_seed=42,
)
cell.generate(
    shell_target,
    grain_size=25.0,
    bond_weight=3.0,
    angle_weight=1.5,
)
```

<!-- Uncomment when images are available:
![Disorder spectrum from liquid to nanocrystalline](images/disorder_spectrum.png)
-->

## Presets

Recommended parameter sets for Si are provided as a class attribute:

```python
tc.Supercell.PRESETS
```

| Regime | grain_size | cryst_frac | bond_weight | angle_weight | density |
|--------|-----------|------------|-------------|--------------|---------|
| liquid | None |  -  | 1.0 | 0.12 | 0.86 |
| amorphous | 4 | 1.0 | 1.5 | 0.3 | 0.88 |
| SRO | 8 | 1.0 | 2.0 | 1.0 | 0.90 |
| MRO | 12 | 0.5 | 2.0 | 0.6 | 0.92 |
| mixed | 18 | 0.5 | 2.5 | 1.0 | 0.94 |
| nanocrystalline | 25 | 1.0 | 3.0 | 1.5 | 0.96 |

These are tuned for Si (diamond cubic). Other materials will need different values  -  in particular, close-packed structures (FCC, HCP) need lower `angle_weight` since their angular distributions are broader.

## Multi-species systems

For binary compounds (SiC, GaAs, etc.), the bond topology respects per-species-pair coordination targets. In SiC zincblende, each Si bonds to 4 C atoms and vice versa  -  no Si-Si or C-C bonds form in the first shell.

```python
atoms = bulk('SiC', 'zincblende', a=4.36)
shell_target = tc.CoordinationShellTarget.from_atoms(
    atoms,
    phi_num_bins=90,
)

# coordination_target shows the expected bonding:
# Si->C: 4, Si->Si: 0, C->Si: 4, C->C: 0
print(shell_target.coordination_target)
```

## Accessing the structure

After `generate()`, the ASE Atoms object is available at `cell.atoms`:

```python
# Write to file
cell.atoms.write('supercell.cif')
cell.atoms.write('supercell.xyz')

# Access positions, numbers, cell
positions = cell.atoms.positions      # (N, 3) array
numbers = cell.atoms.numbers          # (N,) array of atomic numbers
cell_matrix = cell.atoms.cell.array   # (3, 3) cell vectors
```
