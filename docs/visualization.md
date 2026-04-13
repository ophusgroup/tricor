# Visualization

## g3 distribution viewer

After measuring the g3 distribution, view it interactively:

```python
cell.measure_g3()
cell.plot_g3()
```

This opens an interactive widget with:
- **Heatmap** — 2D slice of the g3 distribution (r vs phi) for a selected radial shell
- **Radial profile** — pair correlation function with drag-to-select shell range
- **Channel selector** — switch between triplet types (e.g. Si-Si-Si, Si-C-Si)
- **Normalize toggle** — show raw counts or density-normalized
- **Auto-shell toggle** — when unchecked, the shell selection stays fixed when switching channels

## g3 comparison

To compare the supercell g3 against a target distribution:

```python
# Create a target g3
dist = tc.G3Distribution(atoms)
dist.measure_g3(r_max=10, r_step=0.1, phi_num_bins=90)
target = dist.target_g3(target_r_min=5.0, target_r_max=8.0, r_sigma=0.05, phi_sigma_deg=3.0)

# Build supercell with the target as reference
cell = tc.Supercell(target, cell_dim_angstroms=(40,40,40), relative_density=0.92, rng_seed=42)
cell.generate(shell_target, grain_size=12.0, crystalline_fraction=0.5)
cell.measure_g3()
cell.plot_g3_compare()
```

## 3D structure viewer

Interactive WebGL viewer in Jupyter:

```python
cell.view_structure()
```

Controls:
- **Drag** to rotate, **scroll** to zoom, **right-click drag** to pan
- **Atom size** slider
- **Bond radius** slider
- **Bond cutoff** slider — recomputes bonds live
- **Bond type checkboxes** — toggle per species pair (Si-Si, Si-C, C-C)
- **Slab x/y/z** sliders — clip the view to a fractional range
- **Show/hide** cell outline and bonds

Custom initial settings:

```python
cell.view_structure(
    atom_scale=0.6,
    bond_cutoff=3.0,
    slab_z=(0.0, 0.5),  # show bottom half only
)
```

## Rotating movie

Export a 360-degree rotating movie:

```python
# MP4 (requires ffmpeg)
cell.plot_structure(output='structure.mp4', fps=60, duration=6.0)

# GIF fallback
cell.plot_structure(output='structure.gif', fps=30, duration=4.0)
```

Options:

```python
cell.plot_structure(
    output='structure.mp4',
    width=1024,           # pixel width (default 1024)
    height=1024,          # pixel height (default 1024)
    fps=60,               # frames per second
    duration=6.0,         # seconds
    elevation=15.0,       # camera elevation (degrees)
    atom_size=10.0,       # scatter marker size
    colormap='Reds',      # depth colormap for crystalline bonds
    background='white',
)
```

## Shell relaxation history

View the loss convergence from `generate()` or `shell_relax()`:

```python
cell.plot_shell_relax()
```

Shows total loss, bond loss, angle loss, and repulsion loss per step.
