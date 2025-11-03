import math

import torch
from torch import nn

from model import wavenet
from util.register import register_node


@register_node("tgramnet")
class TgramNet(nn.Module):
    def __init__(
        self,
        num_layer: int = 3,
        mel_bins: int = 128,
        win_len: int = 1024,
        hop_len: int = 512,
        n_frames: int = 313,
    ) -> None:
        super().__init__()
        self.conv_extrctor = nn.Conv1d(
            1, mel_bins, win_len, hop_len, win_len // 2, bias=False
        )
        self.conv_encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LayerNorm(n_frames),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
                )
                for idx in range(num_layer)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out


class Bottleneck(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: tuple[int, int] | int, expansion: int
    ):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                inp * expansion,
                inp * expansion,
                3,
                stride,
                1,
                groups=inp * expansion,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: int,
        padding: int,
        dw: bool = False,
        linear: bool = False,
    ):
        super().__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        self.bn = nn.BatchNorm2d(out_channels)
        if not linear:
            self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        expension: int,
        out_channels: int,
        n_blocks: int,
        stride: tuple[int, int] | int,
    ):
        super().__init__()

        input_channel = ch_in
        layers = []
        for i in range(n_blocks):
            if i == 0:
                layers.append(
                    Bottleneck(input_channel, out_channels, stride, expension)
                )
            else:
                layers.append(Bottleneck(input_channel, out_channels, 1, expension))
            input_channel = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


@register_node("tp_mfn_encoder")
class TPMFNEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride_1: tuple[int, int] | int = (2, 1),
        stride_2: tuple[int, int] | int = (2, 1),
        stride_3: tuple[int, int] | int = (2, 2),
    ):
        super().__init__()

        self.conv1 = ConvBlock(c_in, c_out, 3, 2, 1)
        self.dw_conv1 = ConvBlock(c_out, c_out, 3, 1, 1, dw=True)

        self.encoder1 = EncoderBlock(c_out, 2, 2 * c_out, 2, stride_1)
        self.encoder2 = EncoderBlock(2 * c_out, 4, 2 * c_out, 2, stride_2)
        self.encoder3 = EncoderBlock(2 * c_out, 4, 2 * c_out, 2, stride_3)

    def forward(self, x):
        h = self.conv1(x)
        h = self.dw_conv1(h)

        h = self.encoder1(h)
        h = self.encoder2(h)
        out = self.encoder3(h)

        return out


@register_node("tp_mfn_latent_extractor")
class TPMFNLatentExtractor(nn.Module):
    def __init__(
        self,
        c_in: int = 2,
        c_out: int = 64,
        stride_1: tuple[int, int] | int = (2, 1),
        stride_2: tuple[int, int] | int = (2, 1),
        stride_3: tuple[int, int] | int = (2, 2),
        feature1: int = 512,
        feature2: int = 128,
        height: int = 8,
        return_feature: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBlock(c_in, c_out, 3, 2, 1)
        self.dw_conv1 = ConvBlock(c_out, c_out, 3, 1, 1, dw=True)
        self.encoder1 = EncoderBlock(c_out, 2, 2 * c_out, 2, stride_1)
        self.encoder2 = EncoderBlock(2 * c_out, 4, 2 * c_out, 2, stride_2)
        self.encoder3 = EncoderBlock(2 * c_out, 4, 2 * c_out, 2, stride_3)

        self.ch_conv1 = ConvBlock(2 * c_out, feature1, 1, 1, 0)
        self.ch_conv2 = ConvBlock(
            feature1, feature1, (height, 1), 1, 0, dw=True, linear=True
        )
        self.ch_conv3 = ConvBlock(feature1, feature2, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.return_feature = return_feature

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h0 = self.dw_conv1(h)
        h1 = self.encoder1(h0)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)

        h = self.ch_conv1(h3)
        h = self.ch_conv2(h)
        out = self.ch_conv3(h)

        if self.return_feature:
            out0 = h0.mean(dim=-1)
            out0 = out0.view(out0.shape[0], -1)

            out1 = h1.mean(dim=-1)
            out1 = out1.view(out1.shape[0], -1)

            out2 = h2.mean(dim=-1)
            out2 = out2.view(out2.shape[0], -1)

            out3 = h3.mean(dim=-1)
            out3 = out3.view(out3.shape[0], -1)
            return out, out0, out1, out2, out3
        else:
            return out


@register_node("tp_mfn_projector")
class TPMFNProjector(nn.Module):
    def __init__(self, n_features: int = 128, frames: int = 79):
        super().__init__()
        self.projector = ConvBlock(
            n_features, n_features, (1, frames), 1, 0, dw=True, linear=True
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.projector(x)
        feature = h.view(h.size(0), -1)
        return feature


@register_node("ltsp")
class LTSP(nn.Module):
    def __init__(
        self,
        n_channel: int = 128,
        frames: int = 313,
        n_blocks: int = 3,
        n_mul: int = 3,
        kernel_size: int = 3,
        n_groups: int = 1,
    ):
        super().__init__()

        self.pr = wavenet.WaveNet(
            n_blocks=n_blocks,
            n_channel=n_channel,
            n_mul=n_mul,
            frames=frames,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.pr(x)  # (Batch, Channel, Time)

        return out
