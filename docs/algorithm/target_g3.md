# Target g3 Construction

A target g3 distribution represents the desired structural correlations for a supercell. It is built from a crystalline reference measurement by blurring and blending toward the random limit. Target construction is independent of supercell generation and is used for comparison or as input to ML training.

## Construction pipeline

Starting from a measured crystalline g3:

1. **Reduce.** Divide out the ideal density factor to obtain the reduced distribution $\tilde{g}_3$, which approaches 1.0 in the random limit.

2. **Blur in $\phi$.** Gaussian convolution along the angular axis with reflected boundaries at $\phi = 0$ and $\phi = \pi$. The blur sigma grows linearly with radius:

$$\sigma_\phi(r) = \frac{\sigma_{\phi,\text{ref}}}{r_\text{ref}} \cdot r$$

3. **Blur in $r$.** 2D Gaussian kernel applied to both radial axes ($r_{01}$ and $r_{02}$) via einsum. The blur sigma also scales with radius.

4. **Blend toward random.** Smooth Hermite cubic interpolation between the blurred crystalline distribution and the random limit:

$$\tilde{g}_3^{\text{target}} = (1 - m) \cdot \tilde{g}_3^{\text{blurred}} + m \cdot 1.0$$

where the mixing factor transitions from 0 to 1 between `target_r_min` and `target_r_max`:

$$m(r_\text{eff}) = s^2 (3 - 2s), \quad s = \text{clamp}\left(\frac{r_\text{eff} - r_\text{min}}{r_\text{max} - r_\text{min}},\; 0,\; 1\right)$$

with $r_\text{eff} = \max(r_{01}, r_{02})$.

5. **Un-reduce.** Multiply back by the ideal density factor to recover the raw histogram form.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `target_r_min` | Radius where the transition from crystalline to random begins |
| `target_r_max` | Radius where the distribution is fully random |
| `r_sigma` | Radial blur width in Angstrom (at `r_sigma_at`) |
| `r_sigma_at` | Reference radius where the radial blur equals `r_sigma` |
| `phi_sigma_deg` | Angular blur width in degrees (at `r_sigma_at`) |

## Usage

Target construction is a pure `G3Distribution` operation with no dependence on `Supercell`:

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
    r_sigma_at=2.34,
    phi_sigma_deg=3.0,
)

# View the target
target.plot_g3()

# Use for comparison after generating a supercell
cell = tc.Supercell(
    target,
    cell_dim_angstroms=(40, 40, 40),
)
cell.generate(shell_target, grain_size=15.0)
cell.measure_g3()
cell.plot_g3_compare()
```

## The g2 pair distribution

The same blurring and blending pipeline is also applied to the pair distribution $g_2$, using a 1D version of each operation. The target g2 is stored alongside the target g3 in the returned `G3Distribution` object.
