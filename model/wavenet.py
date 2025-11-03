import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int,
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., : -self.pad]
        return x


class WaveNetResidualBlock(nn.Module):
    def __init__(
        self, n_channel, n_mul, frames, kernel_size, dilation_rate, n_groups=1
    ):
        super(WaveNetResidualBlock, self).__init__()
        self.sigmoid_conv = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            CausalConv1d(
                n_channel * n_mul,
                n_channel * n_mul,
                kernel_size,
                dilation_rate,
                n_groups,
            ),
            nn.Sigmoid(),
        )
        self.tanh_conv = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            CausalConv1d(
                n_channel * n_mul,
                n_channel * n_mul,
                kernel_size,
                dilation_rate,
                n_groups,
            ),
            nn.Tanh(),
        )
        self.skip_connection = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            nn.Conv1d(n_channel * n_mul, n_channel, (1,), groups=n_groups),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            nn.Conv1d(n_channel * n_mul, n_channel * n_mul, (1,), groups=n_groups),
        )

    def forward(self, x):
        sigmoid_conv = self.sigmoid_conv(x)
        tanh_conv = self.tanh_conv(x)
        mul = torch.mul(sigmoid_conv, tanh_conv)
        skip = self.skip_connection(mul)
        residual = self.residual(mul)
        return skip, residual + x


class WaveNet(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups):
        super(WaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, (1,), groups=n_groups),
        )
        self.blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(
                    n_channel, n_mul, frames, kernel_size, 2**i, n_groups
                )
                for i in range(n_blocks)
            ]
        )
        self.skip_connection = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, (1,), groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, (1,), groups=n_groups),
        )

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[1]

        x = self.feature_layer(x)
        skips = []
        for idx, block in enumerate(self.blocks):
            skip, x = block(x)
            skips.append(skip)
        skips = torch.stack(skips).sum(0)
        output = self.skip_connection(skips)
        return output[..., self.get_receptive_field() - 1 : -1]
