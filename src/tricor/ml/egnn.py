"""E(n) equivariant graph neural network for atomic-position regression.

Implements the EGNN architecture from
Satorras, Hoogeboom & Welling, *E(n) Equivariant Graph Neural
Networks*, ICML 2021 (https://arxiv.org/abs/2102.09844).

The architecture is deliberately minimal: no spherical harmonics, no
high-order tensor products, no pretrained foundation model.  Each
EGNN layer simultaneously updates per-atom invariant features
``h`` (used for message passing) and equivariant positions ``x``.
Predictions are equivariant under global rotations / translations of
the input.

Adapted for tricor's setting:

* **Per-cell conditioning** on ``grain_size`` (a scalar regime
  parameter).  Broadcast to every atom and concatenated into the
  edge message MLP.
* **Species embedding.** Each atomic species (Si, O, ...) gets a
  learned ``species_embedding_dim`` vector; concatenated into the
  initial node feature ``h_0``.
* **Output:** per-atom position delta added to the input positions.
  Trained against the FIRE-quenched final positions via MSE on the
  displacement ``(x_target - x_input)``.

The graph is built externally (see :mod:`tricor.ml.dataset` for
training and :mod:`tricor.ml.inference` for runtime) — the model
consumes pre-computed ``edge_index`` + ``edge_vec`` tensors so the
PBC / cutoff logic lives in one place.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _silu_mlp(in_dim: int, hidden_dim: int, out_dim: int,
              n_layers: int = 2) -> nn.Sequential:
    """Small 2-layer SiLU MLP used throughout the EGNN."""
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(max(0, n_layers - 1)):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.SiLU())
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class EGNNLayer(nn.Module):
    """One E(n)-equivariant message-passing block.

    Inputs (per batch — batch dim flattened across cells):
        h          (N, F)    invariant features per atom
        x          (N, 3)    equivariant positions per atom
        edge_index (2, E)    src, dst indices (both into [0, N))
        edge_vec   (E, 3)    PBC-corrected ``x_j - x_i``
        edge_cond  (E, C)    per-edge conditioning (e.g. grain_size
                              broadcast from each cell to its edges).

    Output:
        h_new      (N, F)
        x_new      (N, 3)
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int = 0,
        update_positions: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.update_positions = update_positions
        # Edge message: takes (h_i, h_j, |r_ij|^2, cond) -> message
        edge_in = 2 * hidden_dim + 1 + cond_dim
        self.edge_mlp = _silu_mlp(edge_in, hidden_dim, hidden_dim)
        # Node update: takes (h_i, sum_j m_ij) -> h_new
        self.node_mlp = _silu_mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        # Position scalar: maps each message to a scalar coefficient.
        # ``tanh`` keeps the per-step position update bounded; this
        # was the trick used in the original EGNN paper to stabilise
        # training when atom counts are large.
        if update_positions:
            self.coord_mlp = nn.Sequential(
                _silu_mlp(hidden_dim, hidden_dim, 1),
                nn.Tanh(),
            )
        # Optional residual normalization on h
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index  # (E,)

        # |r_ij|^2 — invariant edge feature
        r2 = (edge_vec * edge_vec).sum(dim=-1, keepdim=True)  # (E, 1)

        # Edge message input
        msg_in = [h[src], h[dst], r2]
        if edge_cond is not None and self.cond_dim > 0:
            msg_in.append(edge_cond)
        m = self.edge_mlp(torch.cat(msg_in, dim=-1))  # (E, F)

        # Aggregate messages per destination node
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        # Node feature update with residual
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h_new)

        # Equivariant position update
        if self.update_positions:
            coord_weight = self.coord_mlp(m)  # (E, 1)
            # x_i <- x_i + sum_j (x_i - x_j) * w_ij
            # Note: edge_vec = x_dst - x_src, and we sum "messages into
            # dst", so the equivariant update reads x_dst -= edge_vec * w.
            pos_msg = edge_vec * coord_weight  # (E, 3)
            pos_agg = torch.zeros_like(x)
            pos_agg.index_add_(0, dst, pos_msg)
            # Divide by neighbour count per atom to stabilise large graphs
            counts = torch.zeros(x.shape[0], device=x.device).index_add_(
                0, dst, torch.ones_like(dst, dtype=x.dtype)
            ).clamp(min=1.0).unsqueeze(-1)
            x_new = x - pos_agg / counts
        else:
            x_new = x

        return h_new, x_new


class EGNN(nn.Module):
    """Stacked EGNN for per-atom displacement regression.

    Parameters
    ----------
    n_species : int
        Number of distinct atomic species the model is trained on.
        Each species gets a learned embedding of dimension
        ``species_embedding_dim``.
    hidden_dim : int
        Width of the invariant feature vector ``h``.
    n_layers : int
        Number of stacked EGNN layers.  Effective receptive field is
        ``n_layers × r_cut``.
    species_embedding_dim : int
        Size of each species embedding (added to initial ``h``).
    cond_dim : int
        Size of the per-cell conditioning vector AFTER embedding
        (e.g. for a single ``grain_size`` scalar passed through a
        small MLP, set ``cond_dim`` to whatever that MLP outputs).
        Set to 0 to disable conditioning.
    cond_input_dim : int
        Raw dimensionality of the conditioning input.  Default 1
        (just ``grain_size``).  Set higher when feeding the full
        regime parameter vector.

    Inputs (model.forward):
        positions     (N, 3)   atom Cartesian positions
        species_idx   (N,)     int64, indices into species embedding
        edge_index    (2, E)   src, dst
        edge_vec      (E, 3)   PBC-corrected r_dst - r_src
        cond          (B, K)   per-cell conditioning (B = number of cells in batch)
        batch         (N,)     int64, mapping each atom to its cell index in [0, B)

    Returns:
        predicted_positions  (N, 3)
    """

    def __init__(
        self,
        n_species: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        species_embedding_dim: int = 16,
        cond_dim: int = 8,
        cond_input_dim: int = 1,
    ) -> None:
        super().__init__()
        self.n_species = n_species
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        self.species_embedding = nn.Embedding(n_species, species_embedding_dim)
        # Project species embedding into the hidden dim space
        self.input_proj = nn.Linear(species_embedding_dim, hidden_dim)

        # Per-cell conditioning encoder
        if cond_dim > 0:
            self.cond_encoder = _silu_mlp(cond_input_dim, cond_dim, cond_dim)
        else:
            self.cond_encoder = None

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, cond_dim=cond_dim)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        positions: torch.Tensor,        # (N, 3)
        species_idx: torch.Tensor,      # (N,)
        edge_index: torch.Tensor,       # (2, E)
        edge_vec: torch.Tensor,         # (E, 3)
        cond: torch.Tensor | None = None,   # (B, cond_input_dim)
        batch: torch.Tensor | None = None,  # (N,) int64
    ) -> torch.Tensor:
        # Initial features from species embedding
        h = self.input_proj(self.species_embedding(species_idx))  # (N, F)
        x = positions

        # Encode + broadcast conditioning to every edge
        edge_cond = None
        if cond is not None and self.cond_encoder is not None:
            c = self.cond_encoder(cond)                       # (B, cond_dim)
            if batch is None:
                # Single-cell batch
                edge_cond = c.expand(edge_vec.shape[0], -1)
            else:
                # Each edge's cond comes from the src atom's batch
                src = edge_index[0]
                edge_cond = c[batch[src]]                     # (E, cond_dim)

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_vec, edge_cond)

        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["EGNN", "EGNNLayer"]
