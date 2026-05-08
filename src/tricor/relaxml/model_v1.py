"""Step-by-step relaxation surrogate model.

MeshGraphNets backbone (reusing ``graphite``'s conv primitives) with a
separate encoder for the global weight-parameter vector.  Output is a
per-atom 3D displacement: given the current positions + weights, predict
where each atom moves over ``k`` tricor relaxation steps.
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


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────


class NodeEncoder(nn.Module):
    """One-hot species -> node embedding."""

    def __init__(self, num_species: int, node_dim: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            MLP([num_species, node_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.embed(z)


class EdgeEncoder(nn.Module):
    """[dx, dy, dz, r] -> edge embedding."""

    def __init__(self, edge_dim: int, init_edge_dim: int = 4) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        return self.embed(edge_attr)


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
    z           (N, num_species)  one-hot species
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
        num_species: int = 1,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_convs: int = 4,
        weight_encoder_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.node_encoder = NodeEncoder(num_species, node_dim)
        self.edge_encoder = EdgeEncoder(edge_dim)
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
        h_edge = self.edge_encoder(edge_attr)

        # Add per-graph weight embedding to each node in that graph.
        w_emb = self.weight_encoder(w)         # (B, node_dim)
        h_node = h_node + w_emb[batch]

        h_node, _ = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning training module
# ──────────────────────────────────────────────────────────────────────────────


class LitRelaxML(L.LightningModule):
    """Lightning training wrapper for RelaxMLModel.

    Hyperparameters (sensible defaults for Si at 50 Å cells):
      - num_species: 1 (Si)
      - num_convs: 4
      - node_dim / edge_dim: 128
      - weight_encoder_hidden: 64
      - ema_decay: 0.9999
      - learn_rate: 1e-3
    """

    def __init__(
        self,
        num_species: int = 1,
        num_weight_features: int = NUM_WEIGHT_FEATURES,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_convs: int = 4,
        weight_encoder_hidden: int = 64,
        ema_decay: float = 0.9999,
        learn_rate: float = 1e-3,
        lr_schedule: str = "cosine",    # "none" | "cosine"
        lr_min_ratio: float = 0.01,
        warmup_steps: int = 500,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = RelaxMLModel(
            num_species=num_species,
            num_weight_features=num_weight_features,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_convs=num_convs,
            weight_encoder_hidden=weight_encoder_hidden,
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

    def _shared_step(self, batch) -> Tensor:
        pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr,
            batch.w, batch.batch,
        )
        # Sum squared error per atom, mean over atoms in the batch.
        loss = (pred - batch.target_displacement).pow(2).sum(dim=-1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log(
            "train_loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, batch_size=bs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log(
            "val_loss", loss,
            on_step=False, on_epoch=True, prog_bar=True, batch_size=bs,
        )
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
