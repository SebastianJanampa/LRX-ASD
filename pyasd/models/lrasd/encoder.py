"""
Copied from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.
"""

import torch
import torch.nn as nn


class Audio_Block(nn.Module):
    """LR-ASD audio block.

    Two stacked convs along the MFCC axis (kernel_1 then kernel_2; out_channels // 2
    then out_channels), followed by two stacked convs along time. Every conv is
    followed by BatchNorm and ReLU.

    Adapted from LR-ASD (https://github.com/Junhua-Liao/LR-ASD).
    """
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        """Initialise the block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_1 (int): Odd kernel size of the first conv on each axis.
            kernel_2 (int): Odd kernel size of the second conv on each axis.
        """
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1-1)/2)
        self.padding_2 = int((kernel_2-1)/2)

        self.m_1 = nn.Conv2d(in_channels, out_channels//2, kernel_size = (kernel_1, 1), padding = (self.padding_1, 0), bias = False)
        self.m_norm_1 = nn.BatchNorm2d(out_channels//2, momentum = 0.01, eps = 0.001)
        self.m_2 = nn.Conv2d(out_channels//2, out_channels, kernel_size = (kernel_2, 1), padding = (self.padding_2, 0), bias = False)
        self.m_norm_2 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.t_1 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, kernel_1), padding = (0, self.padding_1), bias = False)
        self.t_norm_1 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_2 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, kernel_2), padding = (0, self.padding_2), bias = False)
        self.t_norm_2 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):
        """Apply the block. Spatial size is preserved.

        Args:
            x (torch.Tensor): Input of shape (B, in_channels, M, L), with M the MFCC
                axis and L the audio-frame axis.

        Returns:
            torch.Tensor: Output of shape (B, out_channels, M, L).
        """

        x = self.relu(self.m_norm_1(self.m_1(x)))
        x = self.relu(self.m_norm_2(self.m_2(x)))

        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))

        return x


class Visual_Block(nn.Module):
    """LR-ASD visual block.

    Two stacked spatial convs (1 x k x k, kernel_1 then kernel_2; out_channels // 2
    then out_channels), followed by two stacked temporal convs (k x 1 x 1). Every
    conv is followed by BatchNorm and ReLU.

    Adapted from LR-ASD (https://github.com/Junhua-Liao/LR-ASD).
    """
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down = False):
        """Initialise the block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_1 (int): Odd kernel size of the first spatial and temporal convs.
            kernel_2 (int): Odd kernel size of the second spatial and temporal convs.
            is_down (bool): If True, the first spatial conv uses stride 2, halving H
                and W. Defaults to False.
        """
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1-1)/2)
        self.padding_2 = int((kernel_2-1)/2)

        if is_down:
            self.s_1 = nn.Conv3d(in_channels, out_channels//2, kernel_size = (1, kernel_1, kernel_1), stride = (1, 2, 2), padding = (0, self.padding_1, self.padding_1), bias = False)
        else:
            self.s_1 = nn.Conv3d(in_channels, out_channels//2, kernel_size = (1, kernel_1, kernel_1), padding = (0, self.padding_1, self.padding_1), bias = False)
        
        self.s_norm_1 = nn.BatchNorm3d(out_channels//2, momentum = 0.01, eps = 0.001)
        
        self.s_2 = nn.Conv3d(out_channels//2, out_channels, kernel_size = (1, kernel_2, kernel_2), padding = (0, self.padding_2, self.padding_2), bias = False)
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

        self.t_1 = nn.Conv3d(out_channels, out_channels, kernel_size = (kernel_1, 1, 1), padding = (self.padding_1, 0, 0), bias = False)
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_2 = nn.Conv3d(out_channels, out_channels, kernel_size = (kernel_2, 1, 1), padding = (self.padding_2, 0, 0), bias = False)
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):
        """Apply the block. The temporal length is preserved.

        Args:
            x (torch.Tensor): Input of shape (B, in_channels, T, H, W).

        Returns:
            torch.Tensor: Output of shape (B, out_channels, T, H', W'), where H', W' are
            H / 2, W / 2 if ``is_down`` else H, W.
        """

        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))
        
        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))

        return x


class visual_encoder(nn.Module):
    """Visual encoder giving one 128-dim embedding per frame.

    Three ``Visual_Block``s (32, 64, 128 channels) with spatial max pooling,
    followed by global spatial max pooling.

    Adapted from LR-ASD (https://github.com/Junhua-Liao/LR-ASD).
    """
    def __init__(self):
        """Build the blocks and pooling layers. Takes no arguments."""
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, 32, 5, 3, is_down = True)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))

        self.block2 = Visual_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        
        self.block3 = Visual_Block(64, 128, 5, 3)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()     

    def forward(self, x):
        """Encode normalised face crops.

        Spatial size goes 112 -> 56 (block1) -> 28 (pool1) -> 14 (pool2), then is
        max-pooled to 1x1.

        Args:
            x (torch.Tensor): Normalised video of shape (B, 1, T, 112, 112).

        Returns:
            torch.Tensor: Visual embeddings of shape (B, T, 128).
        """

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.reshape(B*T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):
        """Kaiming-normal init for Conv3d weights; BatchNorm3d weight 1, bias 0."""

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    """Audio encoder giving one 128-dim embedding per video frame.

    Three ``Audio_Block``s (32, 64, 128 channels) with max pooling along time
    (4*T -> 2*T -> T), then averaging over the MFCC axis.

    Adapted from LR-ASD (https://github.com/Junhua-Liao/LR-ASD).
    """
    def __init__(self):
        """Build the blocks and pooling layers. Takes no arguments."""
        super(audio_encoder, self).__init__()
        
        self.block1 = Audio_Block(1, 32, 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block2 = Audio_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        
        self.block3 = Audio_Block(64, 128, 5, 3)

        self.__init_weight()
            
    def forward(self, x):
        """Encode MFCC features.

        ``pool1``/``pool2`` are ``MaxPool3d`` applied to 4D tensors, which PyTorch
        treats as unbatched (C, D, H, W); with kernel (1, 1, 3) and stride (1, 1, 2)
        this halves the time axis only.

        Args:
            x (torch.Tensor): MFCC features of shape (B, 1, 13, 4*T).

        Returns:
            torch.Tensor: Audio embeddings of shape (B, T, 128).
        """

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim = 2, keepdim = True)
        x = x.squeeze(2).transpose(1, 2)
        
        return x

    def __init_weight(self):
        """Kaiming-normal init for Conv2d weights; BatchNorm2d weight 1, bias 0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()