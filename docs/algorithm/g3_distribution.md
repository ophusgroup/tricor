# Three-Body Distribution (g3)

## Definition

The rooted three-body distribution $g_3$ captures pairwise distance and angular correlations between atomic triplets. For a center atom at position $\mathbf{r}_0$ with neighbours at $\mathbf{r}_1$ and $\mathbf{r}_2$:

- $r_{01} = |\mathbf{r}_1 - \mathbf{r}_0|$ is the distance to the first neighbour
- $r_{02} = |\mathbf{r}_2 - \mathbf{r}_0|$ is the distance to the second neighbour
- $\phi = \arccos\left(\frac{(\mathbf{r}_1 - \mathbf{r}_0) \cdot (\mathbf{r}_2 - \mathbf{r}_0)}{r_{01} \, r_{02}}\right)$ is the bond angle

The raw histogram is accumulated over all ordered triplets $(i, j, k)$ where $j \le k$ (to avoid double-counting), binned into a 4D array:

$$g_3[\text{triplet\_type}, r_{01}, r_{02}, \phi]$$

where `triplet_type` indexes the species combination (e.g. Si-Si-Si, Si-Si-C, ...).

## Reduced coordinates

The random-limit (ideal gas) g3 scales as $r_{01}^2 \cdot r_{02}^2 \cdot \sin\phi$. The **reduced** g3 is:

$$\tilde{g}_3 = \frac{g_3}{A \cdot r_{01}^2 \cdot r_{02}^2 \cdot \sin\phi}$$

where $A$ is a per-channel amplitude estimated from the far-field mean. In reduced coordinates, $\tilde{g}_3 \to 1$ in the random limit.

## Coordination shell target

`CoordinationShellTarget` extracts first-shell structural targets from a reference crystal:

- $r_\text{peak}$ is the mean nearest-neighbour distance per species pair
- $r_\text{inner}$, $r_\text{outer}$ are the first-shell radial boundaries
- $K_{ij}$ is the coordination number (neighbours of species $j$ around species $i$)
- $\phi_\text{mode}$ is the most probable bond angle per triplet type
- $\phi(\theta)$ is the full angular distribution histogram

These are extracted from the reference crystal using ASE's periodic `neighbor_list`.

## Multi-species indexing

For a system with $S$ species, the number of unique triplet types is:

$$N_\text{triplets} = \frac{S^2 (S + 1)}{2}$$

For example, SiC ($S = 2$) has 6 triplet types: C-C-C, C-C-Si, C-Si-Si (rooted at C), and Si-C-C, Si-C-Si, Si-Si-Si (rooted at Si). The `g3_index` array stores the species mapping for each channel, and `g3_lookup` provides symmetric lookup so that both orderings of the two neighbours map to the same channel.
