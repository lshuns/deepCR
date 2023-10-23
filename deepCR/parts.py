import torch
import torch.nn as nn


class double_conv(nn.Module):
    """
    A double convolution layers: Batch normalisation and ReLU activation
    """
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = nn.Sequential(
            # 3x3 convolution
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # Batch normalisation
            nn.BatchNorm2d(out_ch, momentum=0.005),
            # ReLU activation
            nn.ReLU(inplace=True),
            # 3x3 convolution
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # Batch normalisation
            nn.BatchNorm2d(out_ch, momentum=0.005),
            # ReLU activation
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    """
    The initial convolutional operation
    """
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """
    The downsampling block (encoding part)
    """
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.mpconv = nn.Sequential(
            # Max pooling with a 2x2 kernel
            nn.MaxPool2d(2),
            # Call the double convolution block
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    """
    The upsampling block (decoding part)
    """
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        # Transposed convolution
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        # Call the double convolution block
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate feature maps
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    """
    The final convolution block
    """
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        # 1x1 convolution
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x