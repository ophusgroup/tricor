# tricor

Generate disordered atomic supercells guided by three-body (g3) distributions, spanning the full spectrum from liquid to nanocrystalline. Designed for machine-learning training data generation.

## Overview

tricor builds periodic supercells with controllable disorder — from fully liquid to nanocrystalline — by combining Voronoi grain construction with spring-network relaxation. The resulting structures are characterized by their rooted three-body (g3) distributions, which capture both radial and angular correlations.

**Key features:**

- Generate structures from liquid to nanocrystalline in seconds
- Voronoi grain construction with per-species-pair bond topology
- Interactive g3 distribution visualization (Jupyter widgets)
- Interactive 3D structure viewer (Three.js)
- Rotating MP4/GIF movie export
- Works with any crystal structure (Si, SiC, metals, etc.)

```{toctree}
:maxdepth: 2

quickstart
algorithm
generation
visualization
api
```
