"""Step-by-step relaxation surrogate model.

MeshGraphNets backbone (reusing ``graphite``'s conv primitives) with a
separate encoder for the global weight-parameter vector.  Output is a
per-atom 3D displacement: given the current positions + weights, predict
where each atom moves over ``k`` tricor relaxation steps.

Multi-species: nodes are embedded by atomic number Z via nn.Embedding,
and edges receive an explicit (src_species, dst_species) feature so the
network can learn pair-specific physics (Si–Si vs Si–C vs C–C).
"""

from __future__ import annotations

from typing import Tuple

import lightning as L
import torch
from torch import Tensor, nn

from graphite.nn import MLP
from graphite.nn.convs.mgn import MeshGraphNetsConv
from graphite.nn.models.mgn import Decoder

from .data import NUM_WEIGHT_FEATURES


# Periodic table covers Z = 1..118; sized to 120 gives a small safety margin.
DEFAULT_MAX_Z: int = 120


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────


class NodeEncoder(nn.Module):
    """Atomic number Z -> node embedding via lookup table."""

    def __init__(self, max_z: int, node_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_z, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, z: Tensor) -> Tensor:
        return self.norm(self.embed(z))


class EdgeEncoder(nn.Module):
    """[dx, dy, dz, r, src_pair_emb, dst_pair_emb] -> edge embedding.

    The geometric features `[dx, dy, dz, r]` are concatenated with a
    small (`species_pair_dim`) embedding of the source and destination
    atomic numbers so the network sees pair identity directly.
    """

    def __init__(
        self,
        edge_dim: int,
        species_pair_dim: int,
        init_geom_dim: int = 4,
    ) -> None:
        super().__init__()
        in_dim = init_geom_dim + 2 * species_pair_dim
        self.embed = nn.Sequential(
            MLP([in_dim, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )

    def forward(
        self, edge_attr: Tensor, src_pair: Tensor, dst_pair: Tensor,
    ) -> Tensor:
        return self.embed(torch.cat([edge_attr, src_pair, dst_pair], dim=-1))


class WeightEncoder(nn.Module):
    """Global weight-parameter vector (NUM_WEIGHT_FEATURES,) -> node-space embedding.

    Called once per graph.  The result is broadcast over every atom in
    that graph and added to the node embeddings before message passing.
    """

    def __init__(
        self,
        node_dim: int,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            MLP([num_weight_features, hidden_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )

    def forward(self, w: Tensor) -> Tensor:
        """
        Args:
            w: (B, num_weight_features) — one row per graph in the batch.

        Returns:
            (B, node_dim) — one embedding per graph.
        """
        return self.embed(w)


class Processor(nn.Module):
    """N MeshGraphNetsConv layers with residual + LayerNorm."""

    def __init__(self, num_convs: int, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            MeshGraphNetsConv(node_dim, edge_dim) for _ in range(num_convs)
        ])
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_convs)
        ])
        self.edge_norms = nn.ModuleList([
            nn.LayerNorm(edge_dim) for _ in range(num_convs)
        ])

    def forward(
        self, h_node: Tensor, edge_index: Tensor, h_edge: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        for conv, n_norm, e_norm in zip(self.convs, self.node_norms, self.edge_norms):
            dn, de = conv(h_node, edge_index, h_edge)
            h_node = n_norm(h_node + dn)
            h_edge = e_norm(h_edge + de)
        return h_node, h_edge


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────


class RelaxMLModel(nn.Module):
    """Step-by-step relaxation surrogate.

    Predicts per-atom min-image displacement over k tricor steps given
    the current positions, species, graph, and global weight parameters.

    Forward inputs
    --------------
    z           (N,)              long tensor of atomic numbers
    edge_index  (2, E)            graph connectivity
    edge_attr   (E, 4)            [dx, dy, dz, r]
    w           (B, num_weight_features)  per-graph weight vector
    batch       (N,)              torch_geometric node-to-graph index

    Returns
    -------
    displacement  (N, 3)
    """

    def __init__(
        self,
        max_z: int = DEFAULT_MAX_Z,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_convs: int = 4,
        weight_encoder_hidden: int = 64,
        species_pair_dim: int = 16,
    ) -> None:
        super().__init__()
        self.max_z = int(max_z)
        self.node_encoder = NodeEncoder(max_z, node_dim)
        # Separate small embedding fed into the edge encoder so pair
        # identity is explicit on every edge.  Independent of the node
        # embedding so the edge MLP doesn't grow with node_dim.
        self.pair_species_embed = nn.Embedding(max_z, species_pair_dim)
        self.edge_encoder = EdgeEncoder(
            edge_dim=edge_dim, species_pair_dim=species_pair_dim,
        )
        self.weight_encoder = WeightEncoder(
            node_dim=node_dim,
            num_weight_features=num_weight_features,
            hidden_dim=weight_encoder_hidden,
        )
        self.processor = Processor(num_convs, node_dim, edge_dim)
        self.decoder = Decoder(node_dim, 3)

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        w: Tensor,
        batch: Tensor,
    ) -> Tensor:
        h_node = self.node_encoder(z)

        # Edge features get explicit src/dst species identity.
        pair_emb = self.pair_species_embed(z)               # (N, species_pair_dim)
        src, dst = edge_index[0], edge_index[1]
        h_edge = self.edge_encoder(edge_attr, pair_emb[src], pair_emb[dst])

        # Add per-graph weight embedding to each node in that graph.
        w_emb = self.weight_encoder(w)                      # (B, node_dim)
        h_node = h_node + w_emb[batch]

        h_node, _ = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning training module
# ──────────────────────────────────────────────────────────────────────────────


class LitRelaxML(L.LightningModule):
    """Lightning training wrapper for RelaxMLModel.

    Hyperparameters (sensible defaults for multi-species training):
      - max_z: 120 (full periodic table + safety margin)
      - num_convs: 4
      - node_dim / edge_dim: 128
      - species_pair_dim: 16 (size of edge-side species embedding)
      - weight_encoder_hidden: 64
      - ema_decay: 0.9999
      - learn_rate: 1e-3
    """

    def __init__(
        self,
        max_z: int = DEFAULT_MAX_Z,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_convs: int = 4,
        weight_encoder_hidden: int = 64,
        species_pair_dim: int = 16,
        ema_decay: float = 0.9999,
        learn_rate: float = 1e-3,
        lr_schedule: str = "cosine",    # "none" | "cosine"
        lr_min_ratio: float = 0.01,
        warmup_steps: int = 500,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = RelaxMLModel(
            max_z=max_z,
            num_weight_features=num_weight_features,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_convs=num_convs,
            weight_encoder_hidden=weight_encoder_hidden,
            species_pair_dim=species_pair_dim,
        )

        # EMA weights — standard trick from the flowmatch training loop.
        _ema = float(ema_decay)
        ema_avg = lambda avg_p, p, num_avg: _ema * avg_p + (1 - _ema) * p
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, avg_fn=ema_avg,
        )

        self.learn_rate = learn_rate
        self.lr_schedule = lr_schedule
        self.lr_min_ratio = lr_min_ratio
        self.warmup_steps = warmup_steps

    def _shared_step(self, batch) -> tuple[Tensor, Tensor]:
        """Returns (loss, zero_baseline).  Zero-baseline = the loss the
        same batch would score if the model output zero everywhere; it's
        the natural comparison target for "is the model actually doing
        anything?" since per-step displacements are tiny."""
        pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
        )
        loss = (pred - batch.target_displacement).pow(2).sum(dim=-1).mean()
        zero_baseline = batch.target_displacement.pow(2).sum(dim=-1).mean()
        return loss, zero_baseline

    def training_step(self, batch, batch_idx):
        loss, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        # relative_loss = 1.0 means model predicts zero (baseline);
        # 0.0 means perfect.  Much more interpretable than raw MSE.
        rel = loss / zero_baseline.clamp(min=1e-12)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log("train_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, zero_baseline = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        rel = loss / zero_baseline.clamp(min=1e-12)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        self.log("val_relative_loss", rel, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        if self.lr_schedule == "none":
            return opt

        total_steps = int(self.trainer.estimated_stepping_batches)
        eta_min = self.learn_rate * self.lr_min_ratio

        if self.lr_schedule == "cosine":
            cosine_steps = max(1, total_steps - self.warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cosine_steps, eta_min=eta_min,
            )
            if self.warmup_steps > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0,
                    total_iters=self.warmup_steps,
                )
                sched = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup, cosine],
                    milestones=[self.warmup_steps],
                )
            else:
                sched = cosine
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
