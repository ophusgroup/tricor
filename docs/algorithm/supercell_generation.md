# Supercell Generation

Disordered supercells are built in two stages: grain construction to set the initial atom positions, followed by spring-network relaxation to refine the local geometry.

## Voronoi grain construction

When `grain_size` is specified, the periodic box is filled with crystalline grains:

1. **Seed placement.** $N_\text{seeds}$ random points are placed in the box, where

$$N_\text{seeds} = \left\lceil \frac{V_\text{box}}{V_\text{grain}} \right\rceil, \quad V_\text{grain} = \frac{4}{3}\pi \left(\frac{d_\text{grain}}{2}\right)^3$$

2. **Cell assignment.** $\lfloor f_\text{cryst} \cdot N_\text{seeds} \rfloor$ seeds are randomly marked as crystalline; the rest as amorphous.

3. **Crystal tiling.** The reference unit cell is tiled to fill the supercell box. Each atom is assigned to its nearest Voronoi seed (minimum-image PBC distance).

4. **Per-grain rotation.** Atoms in each crystalline cell are rotated around their seed centre by a random rotation $Q \in SO(3)$ (generated via QR decomposition of a random Gaussian matrix).

5. **Amorphous fill.** Atoms in amorphous Voronoi cells are replaced with random positions at the target density.

6. **Overlap removal.** Atoms closer than `pair_inner` are removed.

7. **Stoichiometry correction.** Excess atoms are trimmed (preferring to remove grain atoms over fill atoms) to match the target composition.

### Grain size clamping and inflation

Small grain sizes are clamped to a minimum of $3 \times r_\text{peak}$ to ensure at least one complete coordination shell. The construction grain size is then inflated:

$$d_\text{construction} = \max(d_\text{user}, 3 r_\text{peak}) + 2 \times 0.75 \, r_\text{peak}$$

to compensate for boundary disorder, which erodes the crystalline core by approximately $r_\text{peak}$ on each side.

## Shell relaxation

The spring-network relaxation simultaneously moves all atoms to match first-shell targets. Three force terms act on each atom.

### Bond springs

For each bonded pair $(i, j)$ with target distance $r_\text{target}$:

$$\mathbf{F}_{ij}^\text{bond} = w_\text{bond} \cdot (r_{ij} - r_\text{target}) \cdot \hat{\mathbf{r}}_{ij}$$

### Angle springs

For each bonded triplet $(a, \text{center}, b)$ with target angle $\phi_\text{target}$:

$$\mathbf{F}_a^\text{angle} = \frac{w_\text{angle} \cdot (\phi - \phi_\text{target})}{r_a} \cdot \mathbf{e}_{\perp,a}$$

where $\mathbf{e}_{\perp,a}$ is the component of $\hat{\mathbf{r}}_b$ perpendicular to $\hat{\mathbf{r}}_a$ in the plane of the triplet.

### Repulsion

Two repulsive terms prevent overlaps and create a clean shell gap:

**Hard core** repulsion acts on any pair closer than `pair_inner`:

$$F^\text{hard} = 4 w_\text{rep} \cdot \left(\frac{r_\text{hard}}{r} - 1\right) \cdot \left(1 + \frac{r_\text{hard}}{r} - 1\right)$$

**Non-bonded clearance** acts on non-bonded pairs closer than $1.5 \times r_\text{peak}$:

$$F^\text{push} = w_\text{rep} \cdot \left(\frac{r_\text{push}}{r} - 1\right) \cdot \left(1 + \frac{r_\text{push}}{r} - 1\right)$$

### Bond topology

The bond graph is rebuilt periodically using a greedy algorithm:

1. Sort all neighbour pairs by distance (nearest first).
2. Accept a bond $(i, j)$ only if:
   - Neither atom has reached its coordination target $K$
   - Neither atom has exceeded its **per-species-pair** coordination target $K_{ij}$
   - The new bond makes angles $\ge 60°$ with all existing bonds at both endpoints
3. Second pass: fill remaining under-coordinated atoms without the angle constraint.

### Integration

FIRE-inspired dynamics with momentum:

$$\mathbf{v}_{n+1} = \begin{cases} 0.8 \, \mathbf{v}_n + \Delta t \, \mathbf{F}_n & \text{if } \mathbf{v}_n \cdot \mathbf{F}_n > 0 \\ 0 & \text{otherwise} \end{cases}$$

Positions are updated and wrapped into the periodic box via fractional coordinates. The step size decays multiplicatively each iteration.

### Grain-aware freezing

Atoms deep inside crystalline grains are identified by their **boundary depth**: half the gap between the distance to the nearest foreign seed and the distance to the atom's own seed. Atoms with boundary depth exceeding $0.5 \times r_\text{peak}$ are classified as **interior** and have their forces and velocities zeroed, so they remain at their crystalline positions throughout relaxation.
