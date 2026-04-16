# Structure Generation

## Overview

`generate()` builds disordered structures in two steps:

1. **Grain construction** - the periodic box is Voronoi-tessellated. A fraction of cells are filled with randomly oriented crystal; the rest with random atom positions.

2. **Shell relaxation** - all atoms are simultaneously relaxed via bond, angle, and repulsion springs. Grain-interior atoms are frozen to preserve crystalline order.

See [Algorithm](algorithm/index.md) for the mathematical details.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grain_size` | float or None | None | Crystallite diameter (A). `None` = no grains. |
| `bond_weight` | float | 1.0 | Spring strength for bond distances. |
| `angle_weight` | float | 0.5 | Spring strength for bond angles. |
| `repulsion_weight` | float | 3.0 | Strength of overlap/clearance repulsion. |
| `hard_core_scale` | float | 1.0 | Scales the min bond distance wall. < 1 = softer. |
| `nonbond_push_scale` | float | 1.0 | Scales non-bonded clearance distance. < 1 = broader 2nd shell. |
| `displacement_sigma` | float | 0.0 | Gaussian jitter on grain atoms (A). |
| `num_steps` | int | 200 | Relaxation iterations. |
| `relative_density` | float | 1.0 | Set on `from_atoms`. Density relative to crystal. |

## Disorder regimes

### Liquid

No grains. Soft hard-core wall and non-bonded push allow broad first and second shells. Moderate angle weight prevents close-packed artifacts.

```python
cell = tc.Supercell.from_atoms(
    atoms,
    (40, 40, 40),
    relative_density=0.96,
    rng_seed=42,
)
cell.generate(
    shell_target,
    num_steps=100,
    grain_size=None,
    bond_weight=0.4,
    angle_weight=0.5,
    repulsion_weight=0.5,
    hard_core_scale=0.75,
    nonbond_push_scale=0.7,
)
```

### Amorphous

Small grains (~6 A) provide ~2 visible maxima in the radial profile. Softened hard-core and non-bonded push keep peaks broad.

```python
cell.generate(
    shell_target,
    num_steps=150,
    grain_size=6.0,
    bond_weight=1.2,
    angle_weight=0.6,
    hard_core_scale=0.9,
    nonbond_push_scale=0.8,
    displacement_sigma=0.08,
)
```

### Short-range order (SRO)

Medium grains (~10 A) with stronger spring weights give ~3 visible maxima.

```python
cell.generate(
    shell_target,
    num_steps=200,
    grain_size=10.0,
    bond_weight=2.2,
    angle_weight=1.0,
    hard_core_scale=0.95,
    nonbond_push_scale=0.9,
    displacement_sigma=0.04,
)
```

### Medium-range order (MRO)

Larger grains (~13 A) with moderate jitter. 4-5 visible maxima, all broad and decreasing in amplitude.

```python
cell.generate(
    shell_target,
    num_steps=150,
    grain_size=13.0,
    bond_weight=1.9,
    angle_weight=0.9,
    hard_core_scale=0.95,
    nonbond_push_scale=0.9,
    displacement_sigma=0.04,
)
```

### MRO (extended)

Same as MRO but with larger grains (~18 A) extending visible correlations to ~8-10 A.

```python
cell.generate(
    shell_target,
    num_steps=150,
    grain_size=18.0,
    bond_weight=2.0,
    angle_weight=1.0,
    hard_core_scale=0.95,
    nonbond_push_scale=0.9,
    displacement_sigma=0.04,
)
```

### Nanocrystalline (small grains)

15 A grains with strong spring weights and minimal jitter. Sharp crystalline peaks with disordered grain boundaries.

```python
cell.generate(
    shell_target,
    num_steps=200,
    grain_size=15.0,
    bond_weight=2.8,
    angle_weight=1.3,
    displacement_sigma=0.02,
)
```

### Nanocrystalline (large grains)

20 A grains filling the box. Very sharp crystalline peaks.

```python
cell.generate(
    shell_target,
    num_steps=150,
    grain_size=20.0,
    bond_weight=3.0,
    angle_weight=1.5,
    displacement_sigma=0.02,
)
```

## Presets

Recommended parameter sets for Si are provided as a class attribute:

```python
tc.Supercell.PRESETS
```

| Regime | grain_size | bond_weight | angle_weight | hard_core_scale | nonbond_push_scale | disp_sigma |
|--------|-----------|-------------|--------------|-----------------|-------------------|------------|
| liquid | None | 0.4 | 0.5 | 0.75 | 0.7 | - |
| amorphous | 6 | 1.2 | 0.6 | 0.9 | 0.8 | 0.08 |
| SRO | 10 | 2.2 | 1.0 | 0.95 | 0.9 | 0.04 |
| MRO | 13 | 1.9 | 0.9 | 0.95 | 0.9 | 0.04 |
| MRO_more | 18 | 2.0 | 1.0 | 0.95 | 0.9 | 0.04 |
| nanocrystalline_10 | 15 | 2.8 | 1.3 | 1.0 | 1.0 | 0.02 |
| nanocrystalline_20 | 20 | 3.0 | 1.5 | 1.0 | 1.0 | 0.02 |

All presets use `relative_density=0.96`. These are tuned for Si (diamond cubic). Other materials will need different values - in particular, close-packed structures (FCC, HCP) need lower `angle_weight` since their angular distributions are broader.

## Multi-species systems

For binary compounds (SiC, GaAs, etc.), the bond topology respects per-species-pair coordination targets. In SiC zincblende, each Si bonds to 4 C atoms and vice versa - no Si-Si or C-C bonds form in the first shell.

```python
atoms = bulk('SiC', 'zincblende', a=4.36)
shell_target = tc.CoordinationShellTarget.from_atoms(
    atoms,
    phi_num_bins=90,
)

print(shell_target.coordination_target)
```

## Accessing the structure

After `generate()`, the ASE Atoms object is available at `cell.atoms`:

```python
cell.atoms.write('supercell.cif')
cell.atoms.write('supercell.xyz')

positions = cell.atoms.positions      # (N, 3) array
numbers = cell.atoms.numbers          # (N,) array of atomic numbers
cell_matrix = cell.atoms.cell.array   # (3, 3) cell vectors
```
