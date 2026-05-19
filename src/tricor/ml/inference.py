"""ML inference for tricor.

Loads a trained per-material EGNN checkpoint and uses it to map a
Voronoi-tiled atomic configuration to predicted FIRE-quenched
positions.  Optionally runs ``K`` real FIRE steps afterwards as a
safety net.

The user-facing entry point is :meth:`tricor.Supercell.generate` with
``backend="ml"`` or ``"ml+fire"`` — this module is the implementation
behind it.

Checkpoint layout (matches what :mod:`tricor.ml.train` writes):

.. code-block:: python

    {
        "model_state": OrderedDict,
        "config": {
            "n_species": int,
            "hidden_dim": int,
            "n_layers": int,
            "species_embedding_dim": int,
            "cond_dim": int,
            "cond_input_dim": int,
            "r_cut": float,
        },
    }
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .egnn import EGNN
from .dataset import build_pbc_graph_chunked


def _bond_relax_sweep(
    positions: np.ndarray,
    box: np.ndarray,
    species_idx: np.ndarray,
    pair_peak: np.ndarray,
    pair_hard_min: np.ndarray,
    pair_outer: np.ndarray,
    coordination_target: np.ndarray,
    n_iter: int = 40,
    attract_frac: float = 0.2,
    repel_frac: float = 1.0,
    max_step: float = 0.2,
) -> np.ndarray:
    """Combined attract-to-bond-peak + repel-from-hard-core sweep.

    This is the production cleanup for ML-generated cells.  Compared
    with the FIRE quench followed by a repulsion pass:

    - Faster (each sweep is one :class:`scipy.spatial.cKDTree`
      query at O(N log N), not a full FIRE step with neighbour-list
      rebuilds).  ~1–2 s/sweep at 200³ Å × 600 k atoms.
    - Doesn't have the FIRE "bond springs pull pairs through the
      hard core in tight regions" failure mode that the iter+rep
      cells exposed.  Every sweep is overlap-aware by construction.
    - Drives the Si–O (or whichever-pair) peak to the actual bond
      target rather than parking atoms at the hard-core wall.

    Per pair within ``max_cutoff``:

    - If ``coordination_target[s_i, s_j] > 0``  AND
      ``dist <= pair_outer[s_i, s_j]``:
      pull toward ``pair_peak[s_i, s_j]`` at rate ``attract_frac``.
    - If ``dist < pair_hard_min[s_i, s_j]``:
      push apart at rate ``repel_frac``.

    Per-atom step is clipped to ``max_step`` Å to prevent runaway
    motion in dense regions.

    Parameters
    ----------
    positions
        (N, 3) atom positions in Å.  Modified in place.
    box
        (3,) orthorhombic box side lengths.
    species_idx
        (N,) per-atom species index.
    pair_peak, pair_hard_min, pair_outer
        (S, S) per-species-pair matrices from
        ``CoordinationShellTarget``.
    coordination_target
        (S, S) per-species-pair NN count.  Non-zero entries identify
        the *bonded* pairs that should be pulled to their peak.
    n_iter
        Number of sweeps.  20 typically reaches the bond peak to
        within 0.01 Å.
    attract_frac, repel_frac
        Per-sweep gap-closing fractions.  Attract is slower than
        repel to avoid overshooting through the bond peak.
    max_step
        Per-atom displacement cap (Å) per sweep.
    """
    from scipy.spatial import cKDTree

    box_np = np.asarray(box, dtype=np.float64)
    sp = np.asarray(species_idx)
    pp = np.asarray(pair_peak, dtype=np.float64)
    phc = np.asarray(pair_hard_min, dtype=np.float64)
    po = np.asarray(pair_outer, dtype=np.float64)
    bonded = np.asarray(coordination_target, dtype=np.float64) > 0
    pull_cut = float(po[bonded].max()) if bonded.any() else 0.0
    max_cut = max(pull_cut, float(phc.max())) * 1.05
    if max_cut <= 0.0:
        return positions
    pos = positions
    for _ in range(int(n_iter)):
        wrap = pos - np.floor(pos / box_np) * box_np
        tree = cKDTree(wrap, boxsize=box_np)
        pairs = tree.query_pairs(max_cut, output_type="ndarray")
        if not len(pairs):
            break
        si = sp[pairs[:, 0]]
        sj = sp[pairs[:, 1]]
        delta = wrap[pairs[:, 1]] - wrap[pairs[:, 0]]
        delta -= np.round(delta / box_np) * box_np
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-9)
        unit = delta / dist[:, None]

        hc = phc[si, sj]
        peak = pp[si, sj]
        outer = po[si, sj]
        is_bond = bonded[si, sj]

        attract = np.where(
            is_bond & (dist <= outer), (dist - peak) * attract_frac, 0.0)
        repel = np.where(dist < hc, -(hc - dist) * repel_frac, 0.0)
        signed = attract + repel

        disp = unit * signed[:, None]
        atom_disp = np.zeros_like(pos)
        np.add.at(atom_disp, pairs[:, 0], disp)
        np.add.at(atom_disp, pairs[:, 1], -disp)
        # Per-atom clipping after accumulation (handles atoms in dense
        # regions that receive many overlapping pair contributions).
        mag = np.linalg.norm(atom_disp, axis=1).clip(min=1e-9)
        scale = np.clip(max_step / mag, 0.0, 1.0)
        atom_disp *= scale[:, None]
        pos += atom_disp
    return pos


def _enforce_hard_core(
    positions: np.ndarray,
    box: np.ndarray,
    species_idx: np.ndarray,
    pair_hard_min: np.ndarray,
    n_iter: int = 2,
    push_fraction: float = 0.5,
) -> np.ndarray:
    """Push apart any pair below its species-pair hard-core distance.

    Iteratively: find offenders via :class:`scipy.spatial.cKDTree`,
    move each pair apart along their bond vector by
    ``push_fraction × deficit``.  Converges in 2–5 iterations for
    realistic cells.

    Vectorised at the pair level (no per-atom Python loop) and
    O(N log N) per iteration thanks to the KDTree.  At 200³ Å × 600 k
    atoms: ~3 s per iteration on CPU.

    Used as a **projection step** between EGNN iterations: keeps the
    configuration feasible (no atom overlaps) so the model never sees
    out-of-distribution clustered states.  See
    :func:`predict_iteratively`.

    Parameters
    ----------
    positions
        (N, 3) atom positions in Å.  Modified in place.
    box
        (3,) orthorhombic box side lengths.
    species_idx
        (N,) per-atom species index into ``pair_hard_min``.
    pair_hard_min
        (S, S) per-species-pair hard-core distance (Å).  Typically
        ``shell_target.pair_hard_min``.
    n_iter
        Maximum number of cleanup iterations.  Early-terminates if
        no offenders remain.
    push_fraction
        Per-iteration push amount as a fraction of the deficit.
        ``0.5`` is the natural choice (atoms meet in the middle);
        larger values can overshoot.

    Returns
    -------
    np.ndarray
        ``positions``, modified in place.
    """
    from scipy.spatial import cKDTree

    pair_hc = np.asarray(pair_hard_min, dtype=np.float64)
    box_np = np.asarray(box, dtype=np.float64)
    species_np = np.asarray(species_idx)
    pos = positions  # (mutated in place)
    max_cutoff = float(pair_hc.max()) * 1.05
    if max_cutoff <= 0.0:
        return pos
    for _ in range(int(n_iter)):
        wrapped = pos - np.floor(pos / box_np) * box_np
        tree = cKDTree(wrapped, boxsize=box_np)
        pairs = tree.query_pairs(max_cutoff, output_type="ndarray")
        if len(pairs) == 0:
            break
        si = species_np[pairs[:, 0]]
        sj = species_np[pairs[:, 1]]
        target = pair_hc[si, sj]
        delta = wrapped[pairs[:, 1]] - wrapped[pairs[:, 0]]
        delta -= np.round(delta / box_np) * box_np
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-9)
        mask = dist < target
        if not mask.any():
            break
        unit = delta[mask] / dist[mask, None]
        push_amt = (target[mask] - dist[mask]) * float(push_fraction)
        push = unit * push_amt[:, None]
        np.add.at(pos, pairs[mask, 0], -push)
        np.add.at(pos, pairs[mask, 1], push)
    return pos


def load_model(checkpoint_path: str | Path,
               device: str | torch.device = "auto") -> EGNN:
    """Load an EGNN checkpoint and return it ready for inference.

    Parameters
    ----------
    checkpoint_path : path
        Path to a ``.pt`` file written by :mod:`tricor.ml.train`.
    device : str or torch.device
        ``"auto"`` (default) picks CUDA, then MPS, then CPU.

    Returns
    -------
    EGNN
        Eval-mode model on the requested device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = EGNN(
        n_species=cfg["n_species"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        species_embedding_dim=cfg["species_embedding_dim"],
        cond_dim=cfg.get("cond_dim", 0),
        cond_input_dim=cfg.get("cond_input_dim", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    # Stash the r_cut on the model for convenience
    model.r_cut = float(cfg.get("r_cut", 5.0))
    model.device = device
    return model


@torch.no_grad()
def predict_positions(
    model: EGNN,
    voronoi_positions: np.ndarray,
    species_idx: np.ndarray,
    box_dim: tuple[float, float, float] | np.ndarray,
    grain_size: float,
    r_cut: float | None = None,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Predict FIRE-quenched positions for a single supercell.

    Parameters
    ----------
    model : EGNN
        From :func:`load_model`.
    voronoi_positions : (N, 3) float array
        Post-Voronoi-tiling positions (output of the existing
        ``_grain.py`` pipeline before FIRE runs).
    species_idx : (N,) int array
        Per-atom virtual species index — matches what's stored in
        ``cell._atom_shell_species_index`` or ``cell._atom_species_index``.
    box_dim : (3,) float
        Orthorhombic cell side lengths.
    grain_size : float
        Conditioning scalar.  Set to 0.0 if the cell has no grain
        construction (amorphous / liquid).
    r_cut : float, optional
        Override the model's saved ``r_cut`` for graph construction.
    chunk_size : int
        Chunk size for the PBC graph builder.  Higher = faster, more
        memory.  4096 is safe for ~50 GB GPUs at 400k atoms.

    Returns
    -------
    np.ndarray of shape (N, 3)
        Predicted positions in the same Å units as the input.
    """
    device = getattr(model, "device", torch.device("cpu"))
    r_cut = float(r_cut if r_cut is not None else getattr(model, "r_cut", 5.0))

    pos = torch.as_tensor(voronoi_positions, dtype=torch.float32, device=device)
    species = torch.as_tensor(species_idx, dtype=torch.long, device=device)
    cond = torch.tensor([[float(grain_size)]], dtype=torch.float32, device=device)

    edge_index, edge_vec = build_pbc_graph_chunked(
        pos, list(box_dim), r_cut, chunk_size=chunk_size,
    )
    edge_index = edge_index.to(device)
    edge_vec = edge_vec.to(device)

    pred = model(
        positions=pos,
        species_idx=species,
        edge_index=edge_index,
        edge_vec=edge_vec,
        cond=cond,
        batch=None,
    )
    # Wrap into the box
    box_t = torch.tensor(box_dim, dtype=pred.dtype, device=device)
    pred = pred - torch.floor(pred / box_t) * box_t
    return pred.detach().cpu().numpy()


@torch.no_grad()
def predict_iteratively(
    model: EGNN,
    voronoi_positions: np.ndarray,
    species_idx: np.ndarray,
    box_dim: "tuple[float, float, float] | np.ndarray",
    grain_size: float,
    n_steps: int = 15,
    r_cut: float | None = None,
    momentum: float = 0.0,
    step_clip: float | None = None,
    repulsion_iters_per_step: int = 0,
    final_repulsion_iters: int = 0,
    pair_hard_min: np.ndarray | None = None,
) -> np.ndarray:
    """Iteratively relax positions by chaining ``n_steps`` EGNN forward
    passes — the network is trained as a *next-step* predictor
    (:class:`tricor.ml.dataset.TricorMLStepDataset`), so each call
    applies ~one FIRE-step's worth of relaxation.

    Parameters
    ----------
    model
        :class:`EGNN` trained on (x_t, x_{t+stride}) trajectory pairs.
    voronoi_positions
        Initial atom positions (post-Voronoi tile).
    species_idx, box_dim, grain_size, r_cut
        Same as :func:`predict_positions`.
    n_steps
        Number of EGNN iterations.  15–25 is typical for full
        relaxation; the same number works across all regimes because
        each call moves atoms by a bounded amount.
    momentum
        Heavy-ball momentum coefficient applied to per-step
        displacement (``Δ_t = output − input``).  ``0.0`` (default)
        is pure gradient descent: ``x_{t+1} = output``.  Larger
        values (e.g. ``0.9``) damp short-period oscillations near
        convergence.
    step_clip
        Optional per-atom displacement cap (Å) applied each
        iteration — a safety net for the first few steps when atoms
        can be far from any training distribution.

    Returns
    -------
    np.ndarray of shape (N, 3)
        Final relaxed positions, wrapped into ``[0, box)``.
    """
    device = getattr(model, "device", torch.device("cpu"))
    r_cut = float(r_cut if r_cut is not None else getattr(model, "r_cut", 5.0))

    pos = torch.as_tensor(voronoi_positions, dtype=torch.float32, device=device)
    species = torch.as_tensor(species_idx, dtype=torch.long, device=device)
    cond = torch.tensor([[float(grain_size)]], dtype=torch.float32, device=device)
    box_t = torch.tensor(box_dim, dtype=pos.dtype, device=device)

    # Pre-compute pair_hard_min numpy view if we're going to repulsion-clean.
    box_np = np.asarray(box_dim, dtype=np.float64)
    species_np = np.asarray(species_idx)
    do_rep_per_step = int(repulsion_iters_per_step) > 0 and pair_hard_min is not None
    do_rep_final   = int(final_repulsion_iters)   > 0 and pair_hard_min is not None

    velocity = torch.zeros_like(pos)
    for step in range(int(n_steps)):
        edge_index, edge_vec = build_pbc_graph_chunked(pos, list(box_dim), r_cut)
        edge_index = edge_index.to(device)
        edge_vec = edge_vec.to(device)
        pred_pos = model(
            positions=pos,
            species_idx=species,
            edge_index=edge_index,
            edge_vec=edge_vec,
            cond=cond,
            batch=None,
        )
        # Min-image displacement so PBC wrap doesn't blow up the step.
        delta = pred_pos - pos
        delta = delta - torch.round(delta / box_t) * box_t
        if step_clip is not None:
            dmag = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp(min=1e-9)
            scale = torch.clamp(step_clip / dmag, max=1.0)
            delta = delta * scale
        if momentum > 0.0:
            velocity = momentum * velocity + delta
            pos = pos + velocity
        else:
            pos = pos + delta
        # Wrap into box
        pos = pos - torch.floor(pos / box_t) * box_t

        # Projection step: push any sub-hard-core pairs apart so the
        # next iteration's input stays in the model's training
        # distribution.  This is what stops iter-K collapse at scale.
        if do_rep_per_step:
            pos_np = pos.detach().cpu().numpy().astype(np.float64)
            pos_np = _enforce_hard_core(
                pos_np, box_np, species_np, pair_hard_min,
                n_iter=int(repulsion_iters_per_step),
            )
            pos = torch.from_numpy(pos_np.astype(np.float32)).to(device)

    # Final cleanup pass with more iterations — drives sub-NN bond
    # count to zero if the per-step pass left residuals.
    if do_rep_final:
        pos_np = pos.detach().cpu().numpy().astype(np.float64)
        pos_np = _enforce_hard_core(
            pos_np, box_np, species_np, pair_hard_min,
            n_iter=int(final_repulsion_iters),
        )
        pos = torch.from_numpy(pos_np.astype(np.float32)).to(device)
    return pos.detach().cpu().numpy()


def predict_and_optionally_relax(
    cell,
    shell_target,
    model: EGNN,
    grain_size: float,
    fire_cleanup_steps: int = 0,
    chunk_size: int | None = None,
    iterative_steps: int = 0,
    iterative_momentum: float = 0.0,
    iterative_step_clip: float | None = None,
    iterative_repulsion_per_step: int = 0,
    final_repulsion_iters: int = 0,
    post_fire_repulsion_iters: int = 0,
    # Combined attract-to-peak + repel-from-hard-core sweep applied
    # AFTER the iter loop (and AFTER FIRE cleanup if used).  Pulls
    # bonded pairs to their target distance — the right replacement
    # for FIRE in the ML pipeline.  See :func:`_bond_relax_sweep`.
    bond_relax_iters: int = 0,
    **shell_relax_kwargs,
) -> None:
    """Mutate ``cell.atoms.positions`` in place with the ML prediction,
    optionally followed by ``fire_cleanup_steps`` of real FIRE.

    Parameters
    ----------
    chunk_size : int, optional
        PBC graph builder chunk size.  None (default) auto-scales
        based on atom count to keep peak memory ~1 GB per chunk
        (use a larger value if you have plenty of GPU RAM).  Pass a
        specific integer to override.

    Notes
    -----
    - The Voronoi-tile positions are read from ``cell.atoms.positions``
      at the time of this call.  The caller is responsible for having
      run the Voronoi tile step already (or pass a pre-tiled cell).
    - ``fire_cleanup_steps == 0`` (default) means pure-ML — no FIRE
      runs.  Set to ≥ 10 for a "safety net" pass.
    """
    species_idx = (
        getattr(cell, "_atom_shell_species_index", None)
        if getattr(cell, "_atom_shell_species_index", None) is not None
        else cell._atom_species_index
    )
    box_dim = np.diag(np.asarray(cell.atoms.cell.array, dtype=np.float32))

    if chunk_size is None:
        # Auto-scale: each PBC graph chunk allocates (chunk_size × N × 3) floats.
        # Aim for ≤ ~1 GB per chunk (~250 M floats).  For N atoms, that's
        # chunk_size ≤ 250e6 / (N * 3).  Cap at 4096 (the default) and
        # floor at 64.
        n_atoms = int(np.asarray(cell.atoms.positions).shape[0])
        target = int(250_000_000 / max(n_atoms * 3, 1))
        chunk_size = int(max(64, min(4096, target)))

    if iterative_steps > 0:
        # Pull pair_hard_min from the shell target so the repulsion
        # projection step can enforce per-species hard-core distances.
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        new_pos = predict_iteratively(
            model,
            voronoi_positions=np.asarray(cell.atoms.positions, dtype=np.float32),
            species_idx=np.asarray(species_idx, dtype=np.int64),
            box_dim=box_dim,
            grain_size=float(grain_size),
            n_steps=int(iterative_steps),
            momentum=float(iterative_momentum),
            step_clip=iterative_step_clip,
            repulsion_iters_per_step=int(iterative_repulsion_per_step),
            final_repulsion_iters=int(final_repulsion_iters),
            pair_hard_min=pair_hard_min,
        )
    else:
        new_pos = predict_positions(
            model,
            voronoi_positions=np.asarray(cell.atoms.positions, dtype=np.float32),
            species_idx=np.asarray(species_idx, dtype=np.int64),
            box_dim=box_dim,
            grain_size=float(grain_size),
            chunk_size=chunk_size,
        )
    cell.atoms.positions = new_pos.astype(np.float64)

    if fire_cleanup_steps > 0:
        # Run a short FIRE quench for verification / cleanup
        kw = dict(shell_relax_kwargs)
        kw["num_steps"] = int(fire_cleanup_steps)
        kw["show_progress"] = False
        kw["capture_trajectory"] = False
        cell.shell_relax(shell_target, **kw)

        # FIRE can introduce sub-hard-core overlaps in tight regions
        # (grain boundaries, dense triplets) because bond/angle forces
        # can pull atoms together faster than FIRE's own repulsion
        # term resolves.  Run a final repulsion sweep on the post-FIRE
        # positions to clean those up.
        if post_fire_repulsion_iters > 0:
            pair_hard_min = np.asarray(
                shell_target.pair_hard_min, dtype=np.float64)
            box_np = np.diag(np.asarray(cell.atoms.cell.array, dtype=np.float64))
            pos_np = np.asarray(cell.atoms.positions, dtype=np.float64)
            pos_np = _enforce_hard_core(
                pos_np, box_np, np.asarray(species_idx),
                pair_hard_min, n_iter=int(post_fire_repulsion_iters),
            )
            cell.atoms.positions = pos_np

    # Combined bond-attract + hard-core-repel sweep.  Run last so it
    # also corrects any residual overlaps from FIRE.  Pulls every
    # bonded pair to its target distance (so Si–O lands at 1.61 Å,
    # not 1.46 Å at the hard-core wall) while keeping non-bonded
    # pairs at safe separations.
    if bond_relax_iters > 0:
        box_np = np.diag(np.asarray(cell.atoms.cell.array, dtype=np.float64))
        pos_np = np.asarray(cell.atoms.positions, dtype=np.float64)
        pos_np = _bond_relax_sweep(
            pos_np, box_np, np.asarray(species_idx),
            pair_peak=np.asarray(shell_target.pair_peak, dtype=np.float64),
            pair_hard_min=np.asarray(shell_target.pair_hard_min, dtype=np.float64),
            pair_outer=np.asarray(shell_target.pair_outer, dtype=np.float64),
            coordination_target=np.asarray(
                shell_target.coordination_target, dtype=np.float64),
            n_iter=int(bond_relax_iters),
        )
        cell.atoms.positions = pos_np


__all__ = [
    "load_model",
    "predict_iteratively",
    "predict_positions",
    "predict_and_optionally_relax",
]
