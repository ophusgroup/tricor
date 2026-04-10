"""Conditioning modules for the flow matching velocity model.

Encodes target g2(r), ADF(phi), and composition into conditioning vectors
that modulate the GNN via FiLM (Feature-wise Linear Modulation).

The spectral encoder compresses the multi-channel 1D signals (g2 and ADF)
into a fixed-size vector using a small CNN. The composition encoder embeds
the stoichiometric fractions via an MLP. Both are combined and injected
into every message-passing layer through learned affine transformations.
"""

import torch
from torch import Tensor, nn


class SpectralEncoder(nn.Module):
    """Encodes g2(r) and ADF(phi) signals into a conditioning vector.

    Uses 1D convolutions to process the multi-channel spectral signals,
    followed by global average pooling and a linear projection.

    Args:
        num_g2_channels: Number of g2 channels (num_species * num_species).
        num_r: Number of radial bins in g2.
        num_adf_channels: Number of ADF triplet channels.
        num_phi: Number of angular bins in ADF.
        cond_dim: Output conditioning dimension.
    """

    def __init__(
        self,
        num_g2_channels: int,
        num_r: int,
        num_adf_channels: int,
        num_phi: int,
        cond_dim: int,
    ) -> None:
        super().__init__()

        # g2 encoder: (batch, num_g2_channels, num_r) -> (batch, cond_dim)
        self.g2_cnn = nn.Sequential(
            nn.Conv1d(num_g2_channels, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 64, 1)
        )
        self.g2_proj = nn.Linear(64, cond_dim)

        # ADF encoder: (batch, num_adf_channels, num_phi) -> (batch, cond_dim)
        self.adf_cnn = nn.Sequential(
            nn.Conv1d(num_adf_channels, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.adf_proj = nn.Linear(64, cond_dim)

    def forward(self, g2: Tensor, adf: Tensor) -> Tensor:
        """
        Args:
            g2: (B, num_species, num_species, num_r) — one per graph in the batch.
            adf: (B, num_triplets, num_phi) — one per graph in the batch.

        Returns:
            c_spec: (B, cond_dim) spectral conditioning vector.
        """
        B = g2.shape[0]
        # Flatten species dimensions for g2: (B, ns*ns, num_r)
        g2_flat = g2.reshape(B, -1, g2.shape[-1])
        adf_flat = adf  # already (B, num_triplets, num_phi)

        h_g2 = self.g2_cnn(g2_flat).squeeze(-1)    # (B, 64)
        h_adf = self.adf_cnn(adf_flat).squeeze(-1)  # (B, 64)

        return self.g2_proj(h_g2) + self.adf_proj(h_adf)  # (B, cond_dim)


class CompositionEncoder(nn.Module):
    """Encodes composition fractions into a conditioning vector.

    Args:
        num_species: Number of possible atomic species.
        cond_dim: Output conditioning dimension.
    """

    def __init__(self, num_species: int, cond_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_species, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, comp_frac: Tensor) -> Tensor:
        """
        Args:
            comp_frac: (B, num_species) composition fractions per graph.

        Returns:
            c_comp: (B, cond_dim) composition conditioning vector.
        """
        return self.mlp(comp_frac)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Applies an affine transformation to node features conditioned on a
    global vector: h' = gamma * h + beta, where gamma and beta are
    learned functions of the conditioning vector.

    Args:
        cond_dim: Dimension of the conditioning vector.
        hidden_dim: Dimension of the node features to modulate.
    """

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, hidden_dim)
        self.beta_net = nn.Linear(cond_dim, hidden_dim)

        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.beta_net.bias)
        nn.init.zeros_(self.beta_net.weight)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            h: (N, hidden_dim) node features.
            c: (N, cond_dim) conditioning vector (broadcast from per-graph to per-node).

        Returns:
            Modulated features (N, hidden_dim).
        """
        gamma = self.gamma_net(c)
        beta = self.beta_net(c)
        return gamma * h + beta
