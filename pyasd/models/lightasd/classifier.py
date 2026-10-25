"""
Copied from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.
"""

import torch
from torch import nn


class BGRU(nn.Module):
    """Bidirectional GRU built from two unidirectional GRUs applied in sequence.

    The forward GRU runs over the input; the backward GRU then runs over the
    time-reversed forward outputs (not the raw input). GELU follows each GRU.

    Adapted from Light-ASD (https://github.com/Junhua-Liao/Light-ASD).
    """
    def __init__(self, channel):
        """Initialise the two GRUs.

        Args:
            channel (int): Input and hidden size of both GRUs.
        """
        super(BGRU, self).__init__()

        self.gru_forward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.gru_backward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        
        self.gelu = nn.GELU()
        self.__init_weight()

    def forward(self, x):
        """Apply the forward then backward GRU.

        Args:
            x (torch.Tensor): Input sequence of shape (B, T, channel).

        Returns:
            torch.Tensor: Output sequence of shape (B, T, channel).
        """
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, dims=[1])
        x = self.gelu(x)

        return x

    def __init_weight(self):
        """Kaiming-normal init for GRU weights; GRU biases set to 0."""
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()