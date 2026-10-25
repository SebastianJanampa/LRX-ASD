##
# ResNet18 Pretrained network to extract lip embedding
# This code is modified based on https://github.com/lordmartian/deep_avsr
##

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """

    def __init__(self, inplanes, outplanes, stride):
        """Initialise the layer.

        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            stride (int): Stride of the first convolution and of the downsampling
                shortcut (used only when ``stride != 1``).
        """
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        """Apply the layer.

        Args:
            inputBatch (torch.Tensor): Input of shape (N, inplanes, H, W).

        Returns:
            torch.Tensor: Output of shape (N, outplanes, H / stride, W / stride).
        """
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """

    def __init__(self):
        """Build the four ResNet layers and the final 4x4 average pool.

        Takes no arguments.
        """
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        
        return


    def forward(self, inputBatch):
        """Apply the ResNet.

        Args:
            inputBatch (torch.Tensor): Feature maps of shape (N, 64, 28, 28).

        Returns:
            torch.Tensor: Features of shape (N, 512, 1, 1).
        """
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class GlobalLayerNorm(nn.Module):
    """Global layer normalisation (gLN).

    Normalises over both the channel and time axes, with a learnable per-channel
    scale (gamma) and shift (beta).

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self, channel_size):
        """Initialise the layer.

        Args:
            channel_size (int): Number of channels N.
        """
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        """Reset gamma to 1 and beta to 0."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Normalise ``y``.

        Args:
            y (torch.Tensor): Input of shape (B, N, L).

        Returns:
            torch.Tensor: Normalised input of shape (B, N, L).
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

class visualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """

    def __init__(self):
        """Build the 3D-conv block and the ResNet.

        Takes no arguments.
        """
        super(visualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet()
        return


    def forward(self, inputBatch):
        """Extract a 512-dim feature per frame.

        The input is in (time, batch, channel, height, width) order. TalkNet passes all
        B*T frames as a single sequence of shape (B*T, 1, 1, 112, 112), so the temporal
        kernel of the 3D conv also spans neighbouring clips in the batch. The 3D conv
        and max pool reduce 112x112 to 28x28 before the ResNet.

        Args:
            inputBatch (torch.Tensor): Frames of shape (L, N, 1, 112, 112).

        Returns:
            torch.Tensor: Per-frame features of shape (L, N, 512).
        """
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1 ,2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch

class DSConv1d(nn.Module):
    """Residual depthwise-separable temporal convolution block on 512 channels.

    ReLU, BatchNorm, depthwise conv (kernel 3), PReLU, gLN and a pointwise conv,
    added to the input.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self):
        """Build the block. Takes no arguments."""
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(512),
            nn.Conv1d(512, 512, 1, bias=False),
            )

    def forward(self, x):
        """Apply the block.

        Args:
            x (torch.Tensor): Input of shape (B, 512, T).

        Returns:
            torch.Tensor: Output of shape (B, 512, T).
        """
        out = self.net(x)
        return out + x

class visualTCN(nn.Module):
    """Visual temporal network (V-TCN): a stack of five ``DSConv1d`` blocks.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self):
        """Build the five blocks. Takes no arguments."""
        super(visualTCN, self).__init__()
        stacks = []        
        for x in range(5):
            stacks += [DSConv1d()]
        self.net = nn.Sequential(*stacks) # Visual Temporal Network V-TCN

    def forward(self, x):
        """Apply the temporal network.

        Args:
            x (torch.Tensor): Visual features of shape (B, 512, T).

        Returns:
            torch.Tensor: Visual features of shape (B, 512, T).
        """
        out = self.net(x)
        return out

class visualConv1D(nn.Module):
    """Projects 512-dim visual features to 128 dims.

    A temporal conv (kernel 5, 512 -> 256) with BatchNorm and ReLU, followed by a
    1x1 conv (256 -> 128).

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self):
        """Build the projection. Takes no arguments."""
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            )

    def forward(self, x):
        """Apply the projection.

        Args:
            x (torch.Tensor): Visual features of shape (B, 512, T).

        Returns:
            torch.Tensor: Visual features of shape (B, 128, T).
        """
        out = self.net(x)
        return out