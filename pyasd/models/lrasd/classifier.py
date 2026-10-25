"""
Copied from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.
"""

import torch
from torch import nn


class Fusion(nn.Module):
    """Gated fusion of two feature sequences.

    Concatenates the inputs along the feature axis and scales every element with a
    sigmoid gate computed by a 1x1 conv and BatchNorm.

    Adapted from LR-ASD (https://github.com/Junhua-Liao/LR-ASD).
    """
    def __init__(self, channel):
        """Initialise the gate.

        Args:
            channel (int): Feature dimension of the concatenated input.
        """
        super(Fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size = 1, padding = 0, bias = False)
        self.bn = nn.BatchNorm1d(channel, momentum = 0.01, eps = 0.001)
    
    def forward(self, x1, x2):
        """Concatenate and gate two sequences.

        Args:
            x1 (torch.Tensor): First sequence of shape (B, T, C1).
            x2 (torch.Tensor): Second sequence of shape (B, T, C2), with
                C1 + C2 == channel.

        Returns:
            torch.Tensor: Gated features of shape (B, T, channel).
        """
        x = torch.cat((x1, x2),2)
        identity = x.transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        x = (identity * w).transpose(1, 2) 
        return x


class Detector(nn.Module):
    """Temporal detector: forward and backward GRUs fused with ``Fusion``.

    Both GRUs (hidden size channel // 4) read the dropout-ed input, the backward
    one in reversed time; their outputs are concatenated and gated.

    Adapted from LR-ASD (https://github.com/Junhua-Liao/LR-ASD).
    """
    def __init__(self, channel):
        """Initialise the GRUs, dropout and fusion.

        Args:
            channel (int): Input feature dimension. The output dimension is
                channel // 2.
        """
        super(Detector, self).__init__()

        self.gru_forward = nn.GRU(input_size = channel, hidden_size = channel//4, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.gru_backward = nn.GRU(input_size = channel, hidden_size = channel//4, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.drop = nn.Dropout(0.5)
        self.attention = Fusion(channel//2)
        self.__init_weight()

    def forward(self, x):
        """Apply the detector.

        Args:
            x (torch.Tensor): Input sequence of shape (B, T, channel).

        Returns:
            torch.Tensor: Output sequence of shape (B, T, channel // 2).
        """
        x1, _ = self.gru_forward(self.drop(x))
        x = torch.flip(x, dims=[1])
        x2, _ = self.gru_backward(self.drop(x))
        x2 = torch.flip(x2, dims=[1])
        x = self.attention(x1, x2)

        return x

    def __init_weight(self):
        """Kaiming-normal init for GRU weights; GRU biases set to 0."""
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()