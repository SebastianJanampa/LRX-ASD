"""
Modified from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.

Audio-visual fusion and temporal detector of LRX-ASD.
"""

import torch
from torch import nn

from pyasd.models.lrxasd.rope import RoPE1D
from pyasd.models.lrxasd.transformer import Block


class Fusion2(nn.Module):
    """Fuse audio and visual embeddings by concatenating them and projecting with 1x1 Conv-BN-ReLU.

    Args:
        in_ch (int): Channels of the concatenated input (audio + visual).
        out_ch (int): Channels of the fused output.
    """

    def __init__(self, in_ch, out_ch):
        super(Fusion2, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm1d(out_ch, momentum = 0.01, eps = 0.001)

    def forward(self, x1, x2):
        """Fuse two per-frame embedding sequences.

        Args:
            x1 (torch.Tensor): Audio embeddings of shape (B, T, C1).
            x2 (torch.Tensor): Visual embeddings of shape (B, T, C2), with
                ``C1 + C2 == in_ch``.

        Returns:
            torch.Tensor: Fused embeddings of shape (B, T, out_ch).
        """
        x = torch.cat((x1, x2), 2).transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        return x.transpose(1, 2)


class DetectorAttention(nn.Module):
    """Temporal detector: a stack of bidirectional (unmasked) transformer blocks with 1D RoPE.

    Replaces the bidirectional GRU of LR-ASD.

    Args:
        nhead (int): Number of attention heads.
        d_model (int): Embedding dimension.
        n_layers (int): Number of transformer blocks.
    """

    def __init__(self,
                 nhead=8,
                 d_model=128,
                 n_layers=1,
                 ):
        super().__init__()
        self.attn_layers_forward = nn.ModuleList([
            Block(d_model, nhead, drop_path=0.2) for _ in range(n_layers)])

        self.n_layers = n_layers

        self.rope_embedding = RoPE1D(d_model // nhead)

    def forward(self, x):
        """Model temporal context over the fused sequence.

        Args:
            x (torch.Tensor): Fused embeddings of shape (B, T, d_model).

        Returns:
            torch.Tensor: Contextualized embeddings of shape (B, T, d_model).
        """
        rope = self.rope_embedding(x)
        for layer_idx in range(self.n_layers):
            x = self.attn_layers_forward[layer_idx](x, rope=rope)
        return x
