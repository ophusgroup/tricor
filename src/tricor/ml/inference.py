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


def predict_and_optionally_relax(
    cell,
    shell_target,
    model: EGNN,
    grain_size: float,
    fire_cleanup_steps: int = 0,
    chunk_size: int | None = None,
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


__all__ = [
    "load_model",
    "predict_positions",
    "predict_and_optionally_relax",
]
