"""Unconditional velocity model for flow matching.

Same MeshGraphNets encoder-processor-decoder as GLASS, but predicts
per-atom velocities instead of scores. No conditioning modules — the
target g2/ADF is injected at inference time via gradient guidance,
exactly like GLASS does with its score model.

Training loss (Lipman et al., 2023, Eq. 14):
    L = E_{t, x0, x1} || v_theta(x_t, t) - (x1 - x0) ||^2

The only architectural difference from GLASS's ScoreModel is the
absence of the 1/sigma output scaling.
"""

import torch
from torch import Tensor, nn
from typing import Tuple

from graphite.nn import MLP
from graphite.nn.basis import GaussianRandomFourierFeatures
from graphite.nn.models.mgn import Processor, Decoder


class Encoder_fm(nn.Module):
    """Time-conditioned encoder. Identical to GLASS's Encoder_dpm."""

    def __init__(
        self,
        init_node_dim: int,
        init_edge_dim: int,
        node_dim: int,
        edge_dim: int,
    ) -> None:
        super().__init__()
        self.embed_node = nn.Sequential(
            MLP([init_node_dim, node_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )
        self.embed_edge = nn.Sequential(
            MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(node_dim, input_dim=1),
            MLP([node_dim, node_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )

    def forward(self, x: Tensor, edge_attr: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        h_node = h_node + self.embed_time(t)
        return h_node, h_edge


class UncondVelocityModel(nn.Module):
    """Unconditional velocity network for flow matching.

    Predicts per-atom velocity v_theta(x_t, t). No conditioning input —
    spectroscopic guidance is applied externally during inference.
    """

    def __init__(self, encoder: Encoder_fm, processor: Processor, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (N, num_species) one-hot species per atom.
            edge_index: (2, E) graph connectivity.
            edge_attr: (E, 4) edge features [dx, dy, dz, r].
            t: (N, 1) flow time per atom.

        Returns:
            velocity: (N, 3) predicted velocity per atom.
        """
        h_node, h_edge = self.encoder(x, edge_attr, t)
        h_node, h_edge = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


# ─── Lightning training module ───────────────────────────────────────────────

import lightning as L


class LitUncondFlowMatch(L.LightningModule):
    """Lightning module for training the unconditional flow matching model.

    Hyperparameters follow GLASS conventions:
      - num_species: number of unique atomic species
      - num_convs: 5 message-passing layers (GLASS default)
      - dim: 200 hidden dimension (GLASS default)
      - ema_decay: 0.9999
      - learn_rate: 1e-3

    The batch must contain:
      - batch.z: (N, num_species) one-hot species
      - batch.edge_index: (2, E) graph connectivity
      - batch.edge_attr: (E, 4) edge features
      - batch.t: (N, 1) flow time
      - batch.target_velocity: (N, 3) the displacement x1 - x0
    """

    def __init__(
        self,
        num_species: int,
        num_convs: int = 5,
        dim: int = 200,
        ema_decay: float = 0.9999,
        learn_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = UncondVelocityModel(
            encoder=Encoder_fm(num_species, 3 + 1, dim, dim),
            processor=Processor(num_convs, dim, dim),
            decoder=Decoder(dim, 3),
        )

        ema_avg = lambda avg_p, p, num_avg: ema_decay * avg_p + (1 - ema_decay) * p
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, avg_fn=ema_avg
        )

        self.learn_rate = learn_rate

    def training_step(self, batch, batch_idx):
        v_pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr, batch.t,
        )
        loss = (v_pred - batch.target_velocity).pow(2).sum(dim=-1).mean()
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        v_pred = self.model(
            batch.z, batch.edge_index, batch.edge_attr, batch.t,
        )
        loss = (v_pred - batch.target_velocity).pow(2).sum(dim=-1).mean()
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
