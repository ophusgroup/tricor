# Algorithm

This page describes the mathematical details of tricor's structure generation pipeline.

## Three-body distribution (g3)

The rooted three-body distribution $g_3$ captures pairwise distance and angular correlations between atomic triplets. For a center atom at position $\mathbf{r}_0$ with neighbours at $\mathbf{r}_1$ and $\mathbf{r}_2$:

- $r_{01} = |\mathbf{r}_1 - \mathbf{r}_0|$  -  distance to first neighbour
- $r_{02} = |\mathbf{r}_2 - \mathbf{r}_0|$  -  distance to second neighbour
- $\phi = \arccos\left(\frac{(\mathbf{r}_1 - \mathbf{r}_0) \cdot (\mathbf{r}_2 - \mathbf{r}_0)}{r_{01} \, r_{02}}\right)$  -  bond angle

The raw histogram is accumulated over all ordered triplets $(i, j, k)$ where $j \le k$ (to avoid double-counting), binned into a 4D array:

$$g_3[\text{triplet\_type}, r_{01}, r_{02}, \phi]$$

where `triplet_type` indexes the species combination (e.g. Si-Si-Si, Si-Si-C, ...).

### Reduced coordinates

The random-limit (ideal gas) g3 scales as $r_{01}^2 \cdot r_{02}^2 \cdot \sin\phi$. The **reduced** g3 is:

$$\tilde{g}_3 = \frac{g_3}{A \cdot r_{01}^2 \cdot r_{02}^2 \cdot \sin\phi}$$

where $A$ is a per-channel amplitude estimated from the far-field mean. In reduced coordinates, $\tilde{g}_3 \to 1$ in the random limit.

### Target g3 construction

A target distribution is built from the crystalline measurement by:

1. **Reduce**  -  divide out the ideal density factor
2. **Blur in $\phi$**  -  Gaussian convolution with reflected boundaries at $\phi = 0$ and $\phi = \pi$. Sigma grows linearly with radius.
3. **Blur in $r$**  -  2D Gaussian kernel applied to both radial axes
4. **Blend toward random**  -  smooth Hermite cubic interpolation:

$$\tilde{g}_3^{\text{target}} = (1 - m) \cdot \tilde{g}_3^{\text{blurred}} + m \cdot 1.0$$

where the mixing factor $m(r_\text{eff})$ transitions from 0 to 1 between `target_r_min` and `target_r_max`, with $r_\text{eff} = \max(r_{01}, r_{02})$.

5. **Un-reduce**  -  multiply back by the ideal density factor

## Coordination shell target

`CoordinationShellTarget` extracts first-shell structural targets from a reference crystal:

- **$r_\text{peak}$**  -  mean nearest-neighbour distance per species pair
- **$r_\text{inner}$, $r_\text{outer}$**  -  first-shell radial boundaries
- **$K_{ij}$**  -  coordination number (neighbours of species $j$ around species $i$)
- **$\phi_\text{mode}$**  -  most probable bond angle per triplet type
- **$\phi(\theta)$**  -  full angular distribution histogram

These are extracted from the reference crystal using ASE's periodic `neighbor_list`.

## Voronoi grain construction

When `grain_size` is specified, the periodic box is filled with crystalline grains:

1. **Seed placement**  -  $N_\text{seeds}$ random points are placed in the box, where

$$N_\text{seeds} = \left\lceil \frac{V_\text{box}}{V_\text{grain}} \right\rceil, \quad V_\text{grain} = \frac{4}{3}\pi \left(\frac{d_\text{grain}}{2}\right)^3$$

2. **Cell assignment**  -  $\lfloor f_\text{cryst} \cdot N_\text{seeds} \rfloor$ seeds are randomly marked as crystalline; the rest as amorphous.

3. **Crystal tiling**  -  the reference unit cell is tiled to fill the supercell box. Each atom is assigned to its nearest Voronoi seed (minimum-image PBC distance).

4. **Per-grain rotation**  -  atoms in each crystalline cell are rotated around their seed centre by a random rotation $Q \in SO(3)$ (generated via QR decomposition of a random Gaussian matrix).

5. **Amorphous fill**  -  atoms in amorphous Voronoi cells are replaced with random positions at the target density.

6. **Overlap removal**  -  atoms closer than `pair_inner` are removed.

7. **Stoichiometry correction**  -  excess atoms are trimmed (preferring to remove grain atoms over fill atoms) to match the target composition.

### Grain size clamping and inflation

Small grain sizes are clamped to a minimum of $3 \times r_\text{peak}$ to ensure at least one complete coordination shell. The construction grain size is then inflated:

$$d_\text{construction} = \max(d_\text{user}, 3 r_\text{peak}) + 2 \times 0.75 \, r_\text{peak}$$

to compensate for boundary disorder, which erodes the crystalline core by approximately $r_\text{peak}$ on each side.

## Shell relaxation

The spring-network relaxation simultaneously moves all atoms to match first-shell targets. Three force terms act on each atom:

### Bond springs

For each bonded pair $(i, j)$ with target distance $r_\text{target}$:

$$\mathbf{F}_{ij}^\text{bond} = w_\text{bond} \cdot (r_{ij} - r_\text{target}) \cdot \hat{\mathbf{r}}_{ij}$$

### Angle springs

For each bonded triplet $(a, \text{center}, b)$ with target angle $\phi_\text{target}$:

$$\mathbf{F}_a^\text{angle} = \frac{w_\text{angle} \cdot (\phi - \phi_\text{target})}{r_a} \cdot \mathbf{e}_{\perp,a}$$

where $\mathbf{e}_{\perp,a}$ is the component of $\hat{\mathbf{r}}_b$ perpendicular to $\hat{\mathbf{r}}_a$ in the plane of the triplet.

### Repulsion

Two repulsive terms prevent overlaps and create a clean shell gap:

**Hard core**  -  for any pair closer than `pair_inner`:

$$F^\text{hard} = 4 w_\text{rep} \cdot \left(\frac{r_\text{hard}}{r} - 1\right) \cdot \left(1 + \frac{r_\text{hard}}{r} - 1\right)$$

**Non-bonded clearance**  -  for non-bonded pairs closer than $1.5 \times r_\text{peak}$:

$$F^\text{push} = w_\text{rep} \cdot \left(\frac{r_\text{push}}{r} - 1\right) \cdot \left(1 + \frac{r_\text{push}}{r} - 1\right)$$

### Bond topology

The bond graph is rebuilt periodically using a greedy algorithm:

1. Sort all neighbour pairs by distance (nearest first)
2. Accept a bond $(i, j)$ only if:
   - Neither atom has reached its coordination target $K$
   - Neither atom has exceeded its **per-species-pair** coordination target $K_{ij}$
   - The new bond makes angles $\ge 60°$ with all existing bonds at both endpoints
3. Second pass: fill remaining under-coordinated atoms without the angle constraint

### Integration

FIRE-inspired dynamics with momentum:

$$\mathbf{v}_{n+1} = \begin{cases} 0.8 \, \mathbf{v}_n + \Delta t \, \mathbf{F}_n & \text{if } \mathbf{v}_n \cdot \mathbf{F}_n > 0 \\ 0 & \text{otherwise} \end{cases}$$

Positions are updated and wrapped into the periodic box via fractional coordinates. The step size decays multiplicatively each iteration.

### Grain-aware freezing

Atoms deep inside crystalline grains are identified by their **boundary depth**: half the gap between the distance to the nearest foreign seed and the distance to the atom's own seed. Atoms with boundary depth exceeding $0.5 \times r_\text{peak}$ are classified as **interior** and have their forces and velocities zeroed  -  they remain at their crystalline positions throughout relaxation.

## Figures

<!-- Add pre-rendered figures from the notebook here.
     Save as PNG and place in docs/images/.

     Example:
     ![g3 distribution for nanocrystalline Si](images/si_nano_g3.png)
     ![Structure movie frame for MRO Si](images/si_mro_frame.png)
     ![Disorder spectrum: liquid to nanocrystalline](images/disorder_spectrum.png)
-->

*Figures will be added from notebook outputs. Save widget screenshots or movie frames as PNGs in `docs/images/`.*
