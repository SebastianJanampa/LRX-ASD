"""1D Rotary Position Embeddings (RoPE) used by the LRX-ASD temporal detector."""

import torch
import torch.nn as nn


def rope_rotate_half(x):
    """Rotate the last dimension by half: ``[x1, x2] -> [-x2, x1]``.

    Args:
        x (torch.Tensor): Tensor of shape (..., D) with an even D.

    Returns:
        torch.Tensor: Tensor with the same shape as ``x``.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x, sin, cos):
    """Apply the rotary embedding to a query or key tensor.

    Args:
        x (torch.Tensor): Queries or keys of shape (B, heads, N, head_dim).
        sin (torch.Tensor): Sine table of shape (N, head_dim).
        cos (torch.Tensor): Cosine table of shape (N, head_dim).

    Returns:
        torch.Tensor: Rotated tensor with the same shape as ``x``.
    """
    return (x * cos) + (rope_rotate_half(x) * sin)


class RoPE1D(nn.Module):
    """Computes the sine/cosine tables of a 1D rotary position embedding.

    Args:
        dim (int): Dimension of each attention head (``d_model // nhead``).
        base (int): Base of the geometric progression of frequencies.
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """Build the RoPE tables for the sequence length of ``x``.

        Args:
            x (torch.Tensor): Input sequence of shape (B, N, C). Only its length
                N and device are used.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(sin, cos)``, each of shape
            (N, dim).
        """
        seq_len = x.shape[1]
        device = x.device
        # Position indices [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)

        # Rotation angles, shape: [seq_len, dim // 2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Repeat to match the full head dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.sin(), emb.cos())
