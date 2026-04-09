"""Score model for GLASS.

MeshGraphNets-style encoder-processor-decoder GNN that predicts per-atom
score vectors for denoising score matching, following the architecture in
Guo & Schwalbe-Koda (arXiv:2603.23210) and Hsu et al. (graphite package).

The model predicts s_hat(x_tilde, t) / sigma, matching the parameterization
in the GLASS paper (Methods, p.11). The training loss is:

    L = E[ || sigma * s_hat(x_tilde, t) + epsilon ||^2 ]

Architecture details (from GLASS Sec. S1.4):
  - Node features: one-hot species (N_species channels)
  - Edge features: (dx, dy, dz, r) — 4D displacement + distance
  - Encoder: 2-layer MLPs with SiLU + LayerNorm for nodes, edges, and time
  - Time embedding: Gaussian random Fourier features -> MLP -> LayerNorm
  - Processor: N_conv=5 MeshGraphNetsConv layers, hidden dim d=200
  - Decoder: MLP -> 3D per-atom score vector, scaled by 1/sigma
  - EMA decay: 0.9999
"""

import torch
from torch import Tensor, nn
from typing import Tuple

from graphite.nn import MLP
from graphite.nn.basis import GaussianRandomFourierFeatures
from graphite.nn.models.mgn import Processor, Decoder


class Encoder_dpm(nn.Module):
    """Time-conditioned encoder for the score model.

    Embeds node features (one-hot species), edge features (displacement + distance),
    and diffusion time via Gaussian random Fourier features. The time embedding
    is added to the node embedding.

    Reproduces the encoder from graphite/notebooks/amorph-gen/lit/modules/prior.py.
    """

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


class ScoreModel(nn.Module):
    """Score network s_hat(x_tilde, t) / sigma.

    Chains encoder -> processor -> decoder, then divides the output by the
    noise level sigma to match GLASS's parameterization.
    """

    def __init__(self, encoder: Encoder_dpm, processor: Processor, decoder: Decoder) -> None:
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
        sigma: Tensor,
    ) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr, t)
        h_node, h_edge = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node) / sigma


# ─── Lightning training module ───────────────────────────────────────────────


import lightning as L


class LitScoreNet(L.LightningModule):
    """PyTorch Lightning module for training the GLASS score model.

    Hyperparameters follow GLASS Sec. S1.4:
      - num_species: number of unique atomic species
      - num_convs: 5 message-passing layers
      - dim: 200 hidden dimension
      - ema_decay: 0.9999
      - learn_rate: 1e-3

    The batch must contain:
      - batch.z: (N, num_species) one-hot species encoding
      - batch.edge_index: (2, E) graph connectivity
      - batch.edge_attr: (E, 4) edge features [dx, dy, dz, r]
      - batch.t: (N, 1) diffusion timestep per atom
      - batch.sigma_r: (N, 1) noise level sigma = k*t
      - batch.eps_r: (N, 3) injected noise epsilon
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

        self.model = ScoreModel(
            encoder=Encoder_dpm(num_species, 3 + 1, dim, dim),
            processor=Processor(num_convs, dim, dim),
            decoder=Decoder(dim, 3),
        )

        ema_avg = lambda avg_p, p, num_avg: ema_decay * avg_p + (1 - ema_decay) * p
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, avg_fn=ema_avg
        )

        self.learn_rate = learn_rate

    def training_step(self, batch, batch_idx):
        score = self.model(
            batch.z, batch.edge_index, batch.edge_attr, batch.t, batch.sigma_r
        )
        # Denoising score matching loss: L = E[||sigma*s + eps||^2]
        loss = (score * batch.sigma_r + batch.eps_r).pow(2).sum(dim=-1).mean()
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log(
            "train_loss", loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=bs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        score = self.model(
            batch.z, batch.edge_index, batch.edge_attr, batch.t, batch.sigma_r
        )
        loss = (score * batch.sigma_r + batch.eps_r).pow(2).sum(dim=-1).mean()
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log(
            "val_loss", loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=bs,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
