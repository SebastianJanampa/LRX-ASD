# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Transformer building blocks (MLP, RoPE attention, pre-norm block) for LRX-ASD."""

from typing import Optional, Type

import torch
import torch.nn as nn

from pyasd.models.lrxasd.rope import rope_apply


class Mlp(nn.Module):
    """Two-layer MLP as used in Vision Transformers.

    Args:
        in_features (int): Input feature dimension.
        hidden_features (int, optional): Hidden dimension. Defaults to ``in_features``.
        out_features (int, optional): Output dimension. Defaults to ``in_features``.
        act_layer (Type[nn.Module]): Activation layer class.
        norm_layer (Type[nn.Module], optional): Normalization applied after the
            activation. ``None`` disables it.
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=True)

    def forward(self, x):
        """Apply the MLP.

        Args:
            x (torch.Tensor): Input of shape (..., in_features).

        Returns:
            torch.Tensor: Output of shape (..., out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional rotary position embeddings.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None, mask=None):
        """Compute self-attention.

        Args:
            x (torch.Tensor): Input sequence of shape (B, N, C).
            rope (tuple[torch.Tensor, torch.Tensor], optional): ``(sin, cos)``
                tables from :class:`RoPE1D`, each of shape (N, head_dim).
            mask (torch.Tensor, optional): Boolean mask where True blocks
                attention. Either an attention mask of shape (N, N) or a padding
                mask of shape (B, N).

        Returns:
            torch.Tensor: Output sequence of shape (B, N, C).
        """
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            # Attention mask [N, N]
            if mask.shape[-1] == N and mask.shape[-2] == N:
                attn.masked_fill_(mask, float('-inf'))
            # Padding mask [B, N]
            elif mask.shape[-1] == N and mask.shape[0] == B:
                attn.masked_fill_(mask.reshape(B, 1, 1, N).expand_as(attn), float('-inf'))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        return x

    def apply_rope(self, q, k, rope):
        """Rotate queries and keys with RoPE.

        Any leading prefix tokens (beyond the length of the RoPE tables) are
        left untouched.

        Args:
            q (torch.Tensor): Queries of shape (B, heads, N, head_dim).
            k (torch.Tensor): Keys of shape (B, heads, N, head_dim).
            rope (tuple[torch.Tensor, torch.Tensor]): ``(sin, cos)`` tables.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotated ``(q, k)`` in their
            original dtypes.
        """
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k


class Block(nn.Module):
    """Pre-norm transformer block: ``x + Attn(LN(x))`` followed by ``x + MLP(LN(x))``.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        drop_path (float): Dropout rate applied to each residual branch.
        norm_layer (Type[nn.Module]): Normalization layer.
        act_layer (Type[nn.Module]): Activation layer.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x, rope=None, mask=None):
        """Apply the transformer block.

        Args:
            x (torch.Tensor): Input sequence of shape (B, N, C).
            rope (tuple[torch.Tensor, torch.Tensor], optional): RoPE tables.
            mask (torch.Tensor, optional): Attention or padding mask, see
                :meth:`Attention.forward`.

        Returns:
            torch.Tensor: Output sequence of shape (B, N, C).
        """
        shortcut = x

        x = self.norm1(x)
        x = self.attn(x, rope=rope, mask=mask)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
