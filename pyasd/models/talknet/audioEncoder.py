import torch
import torch.nn as nn

class SEBasicBlock(nn.Module):
    """ResNet basic block with a squeeze-and-excitation (SE) layer.

    Two 3x3 convolutions with batch norm, SE channel re-weighting and a residual
    connection.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        """Initialise the block.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int or tuple): Stride of the first convolution. Defaults to 1.
            downsample (nn.Module, optional): Module applied to the input so the
                residual matches the output shape. None means identity.
            reduction (int): Channel reduction ratio of the SE layer. Defaults to 8.
        """
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Apply the block.

        Args:
            x (torch.Tensor): Input of shape (B, inplanes, H, W).

        Returns:
            torch.Tensor: Output of shape (B, planes, H', W'), where H' and W' are H and
            W reduced by ``stride``.
        """
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    """Squeeze-and-excitation layer.

    Global average pooling followed by a two-layer MLP with sigmoid output that
    re-weights the channels of the input.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self, channel, reduction=8):
        """Initialise the layer.

        Args:
            channel (int): Number of input/output channels.
            reduction (int): Reduction ratio of the hidden MLP layer. Defaults to 8.
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        """Re-weight the channels of ``x``.

        Args:
            x (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            torch.Tensor: Channel-scaled input of shape (B, C, H, W).
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class audioEncoder(nn.Module):
    """SE-ResNet audio encoder over MFCC features.

    A strided stem and four SE-ResNet stages reduce the time axis by 4 (one
    embedding per video frame); the MFCC axis is then averaged out.

    Adapted from TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD).
    """
    def __init__(self, layers, num_filters, **kwargs):
        """Initialise the encoder.

        Args:
            layers (list[int]): Number of SE blocks in each of the four stages.
            num_filters (list[int]): Number of channels of each of the four stages. The
                output dimension is ``num_filters[3]``.
            **kwargs: Unused.
        """
        super(audioEncoder, self).__init__()
        block = SEBasicBlock
        self.inplanes   = num_filters[0]

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build one stage of residual blocks.

        Also updates ``self.inplanes`` to the output channels of the stage.

        Args:
            block (type): Residual block class (``SEBasicBlock``).
            planes (int): Number of output channels of the stage.
            blocks (int): Number of blocks in the stage.
            stride (int or tuple): Stride of the first block. Defaults to 1.

        Returns:
            nn.Sequential: The stage.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Encode MFCC features into per-video-frame embeddings.

        The stem (stride (2, 1)) and stages 2 and 3 (stride (2, 2)) map the
        (13, 4T) MFCC map to (2, T), which is then averaged over the MFCC axis.

        Args:
            x (torch.Tensor): MFCC features of shape (B, 1, 13, 4T).

        Returns:
            torch.Tensor: Audio embeddings of shape (B, T, num_filters[3]).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=True)
        x = x.view((x.size()[0], x.size()[1], -1))
        x = x.transpose(1, 2)

        return x