"""Conditional velocity model for flow matching.

MeshGraphNets encoder-processor-decoder with FiLM conditioning from
target g2/ADF and composition. Predicts a per-atom velocity field
v(x_t, t, c) for ODE-based structure generation.

The loss is the standard conditional flow matching objective:
    L = E_{t, x0, x1} || v_theta(x_t, t, c) - (x1 - x0) ||^2

where x_t = x0 + t*(x1-x0) is the periodic interpolation and
c = (g2_target, adf_target, composition) is the conditioning signal.
"""

import torch
from torch import Tensor, nn
from typing import Tuple

from graphite.nn import MLP
from graphite.nn.basis import GaussianRandomFourierFeatures
from graphite.nn.models.mgn import Decoder
from graphite.nn.convs.mgn import MeshGraphNetsConv

from .conditioning import SpectralEncoder, CompositionEncoder, FiLMLayer


class Encoder_cfm(nn.Module):
    """Time-conditioned encoder for the flow matching velocity model.

    Same structure as GLASS's Encoder_dpm: embeds node features, edge
    features, and time via Gaussian random Fourier features.
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


class ConditionalProcessor(nn.Module):
    """MeshGraphNets processor with FiLM conditioning after each layer.

    Wraps N MeshGraphNetsConv layers from graphite. After each conv,
    applies FiLM modulation from the conditioning vector.

    Args:
        num_convs: Number of message-passing layers.
        node_dim: Node feature dimension.
        edge_dim: Edge feature dimension.
        cond_dim: Conditioning vector dimension.
    """

    def __init__(
        self, num_convs: int, node_dim: int, edge_dim: int, cond_dim: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [MeshGraphNetsConv(node_dim, edge_dim) for _ in range(num_convs)]
        )
        self.films = nn.ModuleList(
            [FiLMLayer(cond_dim, node_dim) for _ in range(num_convs)]
        )

    def forward(
        self,
        h_node: Tensor,
        edge_index: Tensor,
        h_edge: Tensor,
        c: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        for conv, film in zip(self.convs, self.films):
            h_node, h_edge = conv(h_node, edge_index, h_edge)
            h_node = film(h_node, c)
        return h_node, h_edge


class VelocityModel(nn.Module):
    """Conditional velocity network for flow matching.

    Chains encoder -> conditional processor -> decoder to predict
    per-atom velocity vectors. Conditioned on target g2/ADF and
    composition via FiLM modulation.
    """

    def __init__(
        self,
        encoder: Encoder_cfm,
        processor: ConditionalProcessor,
        decoder: Decoder,
        spectral_encoder: SpectralEncoder,
        composition_encoder: CompositionEncoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.spectral_encoder = spectral_encoder
        self.composition_encoder = composition_encoder

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        t: Tensor,
        g2_target: Tensor,
        adf_target: Tensor,
        comp_frac: Tensor,
        batch_idx: Tensor,
    ) -> Tensor:
        """
        Args:
            z: (N, num_species) one-hot species per atom.
            edge_index: (2, E) graph connectivity.
            edge_attr: (E, 4) edge features [dx, dy, dz, r].
            t: (N, 1) flow time per atom.
            g2_target: (B, num_species, num_species, num_r) target g2 per graph.
            adf_target: (B, num_triplets, phi_num_bins) target ADF per graph.
            comp_frac: (B, num_species) composition fractions per graph.
            batch_idx: (N,) graph index per atom.

        Returns:
            velocity: (N, 3) predicted velocity per atom.
        """
        # Encode conditioning (per-graph)
        c_spec = self.spectral_encoder(g2_target, adf_target)  # (B, cond_dim)
        c_comp = self.composition_encoder(comp_frac)             # (B, cond_dim)
        c_graph = c_spec + c_comp                                # (B, cond_dim)

        # Broadcast to per-node
        c_nodes = c_graph[batch_idx]  # (N, cond_dim)

        # Encoder
        h_node, h_edge = self.encoder(z, edge_attr, t)

        # Processor with FiLM conditioning
        h_node, h_edge = self.processor(h_node, edge_index, h_edge, c_nodes)

        # Decoder -> velocity
        return self.decoder(h_node)


# ─── Lightning training module ───────────────────────────────────────────────

import lightning as L


class LitFlowMatch(L.LightningModule):
    """PyTorch Lightning module for conditional flow matching training.

    Args:
        num_species: Number of unique atomic species.
        num_convs: Message-passing layers (default 6).
        dim: Hidden dimension (default 256).
        cond_dim: Conditioning vector dimension (default 128).
        num_r: Number of radial bins in g2 (must match DifferentiablePDFADF).
        num_phi: Number of angular bins in ADF.
        ema_decay: EMA decay rate (default 0.9999).
        learn_rate: Learning rate (default 5e-4).
    """

    def __init__(
        self,
        num_species: int,
        num_convs: int = 6,
        dim: int = 256,
        cond_dim: int = 128,
        num_r: int = 200,
        num_phi: int = 90,
        ema_decay: float = 0.9999,
        learn_rate: float = 5e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_g2_channels = num_species * num_species
        # num_triplets = num_species^2 * (num_species+1) / 2
        num_triplets = 0
        for c in range(num_species):
            for n1 in range(num_species):
                for n2 in range(n1, num_species):
                    num_triplets += 1

        self.velocity_model = VelocityModel(
            encoder=Encoder_cfm(num_species, 3 + 1, dim, dim),
            processor=ConditionalProcessor(num_convs, dim, dim, cond_dim),
            decoder=Decoder(dim, 3),
            spectral_encoder=SpectralEncoder(
                num_g2_channels, num_r, num_triplets, num_phi, cond_dim,
            ),
            composition_encoder=CompositionEncoder(num_species, cond_dim),
        )

        ema_avg = lambda avg_p, p, num_avg: ema_decay * avg_p + (1 - ema_decay) * p
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.velocity_model, avg_fn=ema_avg
        )

        self.learn_rate = learn_rate

    def training_step(self, batch, batch_idx):
        v_pred = self.velocity_model(
            batch.z, batch.edge_index, batch.edge_attr, batch.t,
            batch.g2_target, batch.adf_target, batch.comp_frac, batch.batch,
        )
        loss = (v_pred - batch.target_velocity).pow(2).sum(dim=-1).mean()
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        v_pred = self.velocity_model(
            batch.z, batch.edge_index, batch.edge_attr, batch.t,
            batch.g2_target, batch.adf_target, batch.comp_frac, batch.batch,
        )
        loss = (v_pred - batch.target_velocity).pow(2).sum(dim=-1).mean()
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.velocity_model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.velocity_model)
