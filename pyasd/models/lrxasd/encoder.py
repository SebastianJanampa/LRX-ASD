"""
Modified from LR-ASD (https://github.com/Junhua-Liao/LR-ASD)
Copyright(c) 2024 LR-ASD. All Rights Reserved.

Visual and audio encoders of LRX-ASD. Both follow the LR-ASD design: spatial
(or frequency) convolutions followed by temporal convolutions. Every convolution
is a re-parameterizable :class:`~pyasd.models.lrxasd.vggconv.RepVGGBlock`.
"""

import torch
import torch.nn as nn

from pyasd.models.lrxasd.vggconv import RepVGGBlock


class Audio_Block(nn.Module):
    """Audio block: two frequency convolutions, then two temporal convolutions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_1 (int): Unused, kept for compatibility with the LR-ASD API.
        kernel_2 (int): Unused, kept for compatibility with the LR-ASD API.
    """

    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        super(Audio_Block, self).__init__()

        self.m1 = RepVGGBlock(
            dim=2, ch_in=in_channels, ch_out=out_channels//2, kernel_sizes=[(5, 1)],
            stride=1, bias=False, groups=in_channels, act='relu'
            )
        self.m2 = RepVGGBlock(
            dim=2, ch_in=out_channels//2, ch_out=out_channels, kernel_sizes=[(3, 1)],
            stride=1, bias=False, groups=1, act='relu'
            )

        self.t1 = RepVGGBlock(
            dim=2, ch_in=out_channels, ch_out=out_channels, kernel_sizes=[(1, 5)],
            stride=1, bias=False, groups=out_channels, act='relu'
            )
        self.t2 = RepVGGBlock(
            dim=2, ch_in=out_channels, ch_out=out_channels, kernel_sizes=[(1, 3)],
            stride=1, bias=False, groups=1, act='relu'
            )

    def forward(self, x):
        """Apply the block.

        Args:
            x (torch.Tensor): MFCC features of shape (B, in_channels, F, T_a).

        Returns:
            torch.Tensor: Features of shape (B, out_channels, F, T_a).
        """
        x = self.m1(x)
        x = self.m2(x)

        x = self.t1(x)
        x = self.t2(x)

        return x


class Visual_Block(nn.Module):
    """Visual block: two spatial convolutions, then two temporal convolutions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_1 (int): Unused, kept for compatibility with the LR-ASD API.
        kernel_2 (int): Unused, kept for compatibility with the LR-ASD API.
        is_down (bool): If True, the first spatial conv uses stride 2 in H and W.
    """

    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down = False):
        super(Visual_Block, self).__init__()

        self.s1 = RepVGGBlock(
            dim=3, ch_in=in_channels, ch_out=out_channels//2, kernel_sizes=[(1, 5, 5)],
            stride=(1, 2, 2) if is_down else 1, bias=False, groups=in_channels, act='relu'
            )
        self.s2 = RepVGGBlock(
            dim=3, ch_in=out_channels//2, ch_out=out_channels, kernel_sizes=[(1, 3, 3)],
            stride=1, bias=False, groups=1, act='relu'
            )

        self.t1 = RepVGGBlock(
            dim=3, ch_in=out_channels, ch_out=out_channels, kernel_sizes=[(5, 1, 1)],
            stride=1, bias=False, groups=out_channels, act='relu'
            )
        self.t2 = RepVGGBlock(
            dim=3, ch_in=out_channels, ch_out=out_channels, kernel_sizes=[(3, 1, 1)],
            stride=1, bias=False, groups=1, act='relu'
            )

    def forward(self, x):
        """Apply the block.

        Args:
            x (torch.Tensor): Face crops of shape (B, in_channels, T, H, W).

        Returns:
            torch.Tensor: Features of shape (B, out_channels, T, H', W'), where
            H' = H / 2 when ``is_down`` is True, and H' = H otherwise.
        """
        x = self.s1(x)
        x = self.s2(x)

        x = self.t1(x)
        x = self.t2(x)

        return x


class visual_encoder(nn.Module):
    """Visual encoder: three visual blocks and spatial max-pooling to one vector per frame.

    Args:
        dims (list[int]): Output channels of the three blocks. The last value is
            the embedding dimension.
    """

    def __init__(self, dims=[32, 64, 128]):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, dims[0], 5, 3, is_down = True)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))

        self.block2 = Visual_Block(dims[0], dims[1], 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))

        self.block3 = Visual_Block(dims[1], dims[2], 5, 3)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()

    def forward(self, x):
        """Encode a face track.

        Args:
            x (torch.Tensor): Normalized grayscale faces of shape (B, 1, T, 112, 112).

        Returns:
            torch.Tensor: Per-frame visual embeddings of shape (B, T, dims[-1]).
        """
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = x.transpose(1,2)
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)

        x = self.maxpool(x)

        x = x.view(B, T, C)

        return x

    def __init_weight(self):
        """Kaiming-normal init for Conv3d layers and identity init for BatchNorm3d."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    """Audio encoder: three audio blocks with 4x temporal downsampling to the video frame rate.

    Args:
        dims (list[int]): Output channels of the three blocks. The last value is
            the embedding dimension.
    """

    def __init__(self, dims=[32, 64, 128]):
        super(audio_encoder, self).__init__()

        self.block1 = Audio_Block(1, dims[0], 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block2 = Audio_Block(dims[0], dims[1], 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block3 = Audio_Block(dims[1], dims[2], 5, 3)

        self.__init_weight()

    def forward(self, x):
        """Encode an MFCC sequence.

        Args:
            x (torch.Tensor): MFCCs of shape (B, 1, 13, 4*T).

        Returns:
            torch.Tensor: Per-video-frame audio embeddings of shape (B, T, dims[-1]).
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
        """Kaiming-normal init for Conv2d layers and identity init for BatchNorm2d."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
