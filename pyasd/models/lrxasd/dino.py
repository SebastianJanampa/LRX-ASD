# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm

# Modified from DINOv2 (https://github.com/facebookresearch/dinov2).


class DINOHead(nn.Module):
    """DINO projection head: MLP -> L2-normalize -> weight-normalized linear layer.

    LRX-ASD uses it for audio-visual -> visual self-distillation: the teacher is
    the fused audio-visual embedding and the student is the visual-only embedding.

    Args:
        in_dim (int): Input embedding dimension.
        out_dim (int): Number of prototypes (output dimension).
        use_bn (bool): Use BatchNorm in the MLP.
        nlayers (int): Number of MLP layers (at least 1).
        hidden_dim (int): Hidden dimension of the MLP.
        bottleneck_dim (int): Output dimension of the MLP (input of the last layer).
        mlp_bias (bool): Use bias in the MLP linear layers.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, m):
        """Truncated-normal init (std 0.02) for linear layers and zero bias."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Project embeddings onto the prototypes.

        Args:
            x (torch.Tensor): Embeddings of shape (..., in_dim).

        Returns:
            torch.Tensor: Prototype scores (logits) of shape (..., out_dim).
        """
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    """Build the DINO head MLP: Linear(-BN)-GELU blocks ending in a Linear to ``bottleneck_dim``.

    Args:
        nlayers (int): Number of linear layers.
        in_dim (int): Input dimension.
        bottleneck_dim (int): Output dimension.
        hidden_dim (int, optional): Hidden dimension (needed when ``nlayers > 1``).
        use_bn (bool): Insert BatchNorm1d after each hidden linear layer.
        bias (bool): Use bias in the linear layers.

    Returns:
        nn.Module: The MLP.
    """
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)