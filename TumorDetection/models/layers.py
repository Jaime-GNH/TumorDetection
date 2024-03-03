from typing import Union, Tuple, List, Optional

import torch
import torch.nn.functional as tfn


class ConvBlock(torch.nn.Module):
    """
    Convolutional + BatchNorm + Activation Block.
    """
    def __init__(self, in_channels: Optional[int], out_channels: Optional[int],
                 kernel_size: Optional[Union[Tuple[int, int], int]],
                 stride:  Optional[Union[Tuple[int, int], int]],
                 dilation: Optional[int], padding: Optional[Union[str, int]],
                 bias: Optional[bool], device: str,
                 groups: Optional[int] = 1, apply_conv: bool = True, use_batchnorm: bool = True,
                 use_act: bool = True, transpose: bool = False):
        """
        Block Constructor
        :param in_channels: input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: output channels (C) in BxCxHxW batch image tensor.
        :param kernel_size: kernel convolution size.
        :param stride: convolution stride factor.
        :param dilation: convolution dilation rate.
        :param padding: convolution padding.
        :param bias: use bias or not.
        :param device: device to use for computation.
        :param groups: convolution groups.
        :param apply_conv: Apply convolution in convolution block or not.
        :param use_batchnorm: Use batch normalization in convolution block.
        :param use_act: Use activation prelu in convolution block.
        :param transpose: Use transposed convolution.
        """
        super().__init__()
        layers = []
        if apply_conv:
            if transpose:
                layers.append(torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                                       dilation=dilation, bias=bias, groups=groups, device=device))
            else:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, groups=groups, device=device))
        if use_batchnorm:
            layers.append(torch.nn.BatchNorm2d(out_channels, device=device))
        if use_act:
            layers.append(torch.nn.PReLU(device=device))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        return self.layers(x)


class SeparableConv2d(torch.nn.Module):
    """
    Separable DepthWise Conv.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Optional[Union[Tuple[int, int], int]],
                 bias: bool, device: str):
        """
        Layer constructor
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param kernel_size: Kernel convolution size.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                         groups=in_channels, bias=bias, padding=1, device=device)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels,
                                         kernel_size=1, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel Shuffle -> torch.nn.Channel Shuffle: `RuntimeError: derivative for channel_shuffle is not implemented`
    Forward pass of a Channel Shuffle Layer.
    :param x: incoming_tensor
    :param groups: number of channel groups to be shuffled.
    """
    batch, in_channels, height, width = x.shape
    channels_per_group = in_channels // groups
    x = torch.reshape(x, [batch, groups, channels_per_group, height, width])
    x = torch.permute(x, (0, 2, 1, 3, 4))
    x = torch.reshape(x, (batch, in_channels, height, width))
    return x


class InitialBlock(torch.nn.Module):
    """
    Initial Block of EFSNet.
    """
    def __init__(self, in_channels: int, dr_rate: float, bias: bool, device: str):
        """
        Inital Block.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=16-in_channels, kernel_size=(3, 3),
                              stride=2, padding=1, dilation=1, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)
        self.mxp21 = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        x1 = self.cb11(x)
        x1 = self.spr11(x1)
        x2 = self.mxp21(x)
        return torch.cat([x1, x2], dim=1)


class DownsamplingBlock(torch.nn.Module):
    """
    Downsampling block of EFSNet
    """
    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, bias: bool, device: str):
        """
        DownSampling Block for dimension reduction.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.mxp11 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.bn11 = torch.nn.BatchNorm2d(in_channels, device=device)
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device, use_act=False)

        self.cb21 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=(2, 2),
                              padding=0, stride=2, dilation=1, bias=bias, device=device)
        self.cb22 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3, 3),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.cb23 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=(1, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.spr21 = torch.nn.Dropout2d(p=dr_rate)

        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # SHORTCUT
        x1 = self.mxp11(x)
        x1 = self.bn11(x1)
        x1 = self.cb11(x1)

        # PATH
        x2 = self.cb21(x)
        x2 = self.cb22(x2)
        x2 = self.cb23(x2)
        x2 = self.spr21(x2)

        # COMBINED
        return self.act_f(x1+x2)


class FactorizedBlock(torch.nn.Module):
    """
    Downsampling block of EFSNet
    """
    def __init__(self, in_channels, out_channels, dr_rate, bias, device):
        """
        Factorized Block layer constructor.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=(1, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.cb12 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(1, 3),
                              padding='same', stride=1, dilation=1, bias=bias, device=device, use_act=False)

        self.cb13 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)

        self.cb14 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=(1, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)

        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # PATH
        x1 = self.cb11(x)
        x1 = self.cb12(x1)
        x1 = self.cb13(x1)
        x1 = self.cb14(x1)
        x1 = self.spr11(x1)

        # COMBINED
        return self.act_f(x + x1)


class SDCBlock(torch.nn.Module):
    """
    SDC Block of EFSNet
    """
    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, bias: bool,
                 groups: int, dilation: int, device: str):
        """
        Shuffle Dilated Convolution -SDC- Block layer constructor
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param dilation: convolution dilation rate.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.groups = groups
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=(1, 1),
                              stride=1, groups=groups, padding='same', dilation=1, bias=bias, device=device)

        self.cb12 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3, 3),
                              stride=1, padding='same', dilation=dilation, bias=bias, device=device,
                              use_act=False)

        self.cb13 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, groups=groups, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # PATH
        x1 = self.cb11(x)
        x1 = channel_shuffle(x1, self.groups)
        x1 = self.cb12(x1)
        x1 = self.cb13(x1)
        x1 = self.spr11(x1)

        # COMBINED
        return self.act_f(x + x1)


class SuperSDCBlock(torch.nn.Module):
    """
    Super SDC Block of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float,
                 bias: bool, groups: int, device: str, num_sdc: int = 4):
        """
        Super SDC Block Layer constructor.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param device: Device to use for computation.
        :param num_sdc: Number of sdc blocks to stack.
        """
        super().__init__()
        self.super_sdc = torch.nn.Sequential(*[
            SDCBlock(in_channels=in_channels, out_channels=out_channels, dr_rate=dr_rate,
                     bias=bias, groups=groups, dilation=2 ** k,
                     device=device)
            for k in range(num_sdc)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        return self.super_sdc(x)


class UpsamplingBlock(torch.nn.Module):
    """
    Upsampling Block of EFSNet
    """
    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, bias: bool, device: str):
        """
        Upsample Module for enlarging image.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device,
                              use_act=False)
        self.bnact11 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=None,
                                 stride=None, padding=None, dilation=None, bias=None, device=device,
                                 apply_conv=False)

        self.cb21 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device)
        self.ct21 = ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=(2, 2),
                              stride=2, padding=0, dilation=1, bias=bias, device=device,
                              transpose=True)
        self.spr21 = torch.nn.Dropout2d(p=dr_rate)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x1: incoming tensor 1
        :param x2: incoming tensor 2
        :return: outgoing tensor
        """
        # PATH1
        x1 = self.cb11(x1)
        x1 = tfn.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.bnact11(x1)

        # PATH2
        x2 = self.cb21(x2)
        x2 = self.ct21(x2)
        x2 = self.spr21(x2)

        return self.act_f(torch.cat([x1, x2], dim=1))


class ShuffleNet(torch.nn.Module):
    """
    ShuffleNet of EFSNet
    """
    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, bias: bool, groups: int, device: str):
        """
        Shuffle Net layer constructor
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.groups = groups
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, groups=groups, bias=bias, device=device,
                              use_act=False)
        self.act11 = torch.nn.ReLU()
        self.scb11 = SeparableConv2d(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3, 3),
                                     bias=False, device=device)
        self.bn11 = torch.nn.BatchNorm2d(out_channels // 4, device=device)
        self.cb12 = ConvBlock(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)
        self.act_f = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # PATH
        x1 = self.cb11(x)
        x1 = self.act11(x1)
        x1 = channel_shuffle(x1, self.groups)
        x1 = self.scb11(x1)
        x1 = self.bn11(x1)
        x1 = self.cb12(x1)
        x1 = self.spr11(x1)

        # COMBINED
        return self.act_f(x+x1)


class Encoder(torch.nn.Module):
    """
    Encoder of EFSNet
    """
    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, bias: bool, groups: int,
                 num_factorized_blocks: int, num_super_sdc_blocks: int,
                 num_sdc_per_supersdc: int, device: str):
        """
        EFSNet Encoder Layer constructor.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param num_factorized_blocks: Num for factorized blocks.
        :param num_super_sdc_blocks: Num of super SDC Blocks.
        :param num_sdc_per_supersdc: Num of SDC Blocks per superSDC Block.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.initial_block = InitialBlock(in_channels=in_channels, dr_rate=dr_rate, bias=bias, device=device)
        self.downsampling_block1 = DownsamplingBlock(in_channels=16, out_channels=out_channels // 2, dr_rate=dr_rate,
                                                     bias=bias, device=device)
        self.factorized_blocks = torch.nn.Sequential(*[FactorizedBlock(
            in_channels=out_channels // 2, out_channels=out_channels // 2, dr_rate=dr_rate, bias=bias, device=device
        ) for _ in range(num_factorized_blocks)])
        self.downsampling_block2 = DownsamplingBlock(in_channels=out_channels // 2, out_channels=out_channels,
                                                     dr_rate=dr_rate, bias=bias, device=device)
        self.super_sdc_blocks = torch.nn.Sequential(*[
            SuperSDCBlock(in_channels=out_channels, out_channels=out_channels, dr_rate=dr_rate, bias=bias,
                          groups=groups, device=device,
                          num_sdc=num_sdc_per_supersdc)
            for _ in range(num_super_sdc_blocks)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        outputs = []
        x1 = self.initial_block(x)
        x1 = self.downsampling_block1(x1)
        outputs.append(x1)

        x2 = self.factorized_blocks(x1)
        x2 = self.downsampling_block2(x2)
        outputs.append(x2)

        x3 = self.super_sdc_blocks(x2)
        outputs.append(x3)
        return outputs


class Decoder(torch.nn.Module):
    """
    Decoder of EFSNet
    """
    def __init__(self, in_channels: int, dr_rate: float, bias: bool, groups: int,
                 num_shufflenet: int, device: str):
        """
        EFSNet Decoder layer constructor
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param num_shufflenet: Num of shuffle nets per upsampling step.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.upsample_module1 = UpsamplingBlock(in_channels=in_channels, out_channels=in_channels // 4, dr_rate=dr_rate,
                                                bias=bias,  device=device)
        self.shufflenet1 = torch.nn.Sequential(*[
            ShuffleNet(in_channels=in_channels // 2, out_channels=in_channels // 2, dr_rate=dr_rate,
                       bias=bias, groups=groups, device=device)
            for _ in range(num_shufflenet)
        ])

        self.upsample_module2 = UpsamplingBlock(in_channels=in_channels // 2, out_channels=8,
                                                dr_rate=dr_rate, bias=bias, device=device)
        self.shufflenet2 = torch.nn.Sequential(*[
            ShuffleNet(in_channels=16, out_channels=16, dr_rate=dr_rate,
                       bias=bias, groups=groups // 2, device=device)
            for _ in range(num_shufflenet)
        ])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x1: incoming tensor 1 from EFSNet Encoder.
        :param x2: incoming tensor 2 from EFSNet Encoder.
        :param x3: incoming tensor 3 from EFSNet Encoder.
        :return: outgoing tensor
        """

        x = self.upsample_module1(x3, x2)
        x = self.shufflenet1(x)
        x = self.upsample_module2(x, x1)
        x = self.shufflenet2(x)

        return x
