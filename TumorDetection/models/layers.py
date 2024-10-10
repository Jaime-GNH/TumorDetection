from typing import Union, Tuple, List, Optional
import torch
import torch.nn.functional as tfn


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


class ChannelShuffle(torch.nn.Module):
    """
    Channel Shuffle Layer.
    """

    def __init__(self, groups: int):
        """
        Layer constructor
        :param groups: Number of grups to shuffle.
        """
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x:
        :return:
        """
        batch, in_channels, height, width = x.shape
        channels_per_group = in_channels // self.groups
        x = torch.reshape(x, [batch, self.groups, channels_per_group, height, width])
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = torch.reshape(x, (batch, in_channels, height, width))
        return x


class InterpolateBilinear(torch.nn.Module):
    """
    Interpolation Layer.
    """

    def __init__(self, stride: int):
        """
        Layer constructor
        :param stride: Augment factor.
        """
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x:
        :return:
        """
        return tfn.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)


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


class DisaggregatedConvBlock(torch.nn.Module):
    """
    Desaggregated Convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, reduction: int,
                 kernel_size: int, stride: int, dilation: int, groups: int,
                 bias: bool, factorize: bool, separable: bool, device: str):
        super().__init__()
        assert not (factorize and separable), f'You must choose between Separable and Factorized.'
        layers = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction,
                                  kernel_size=(stride, stride), stride=stride, padding=0,
                                  dilation=1, bias=bias, groups=groups, device=device),
                  torch.nn.BatchNorm2d(out_channels // reduction, device=device),
                  torch.nn.PReLU(device=device)]

        if groups > 1:
            layers.append(ChannelShuffle(groups=groups))

        if factorize:
            layers += [torch.nn.Conv2d(in_channels=out_channels // reduction,
                                       out_channels=out_channels // reduction,
                                       kernel_size=(kernel_size, 1), stride=1, padding='same',
                                       dilation=dilation, bias=bias, groups=1, device=device),
                       torch.nn.Conv2d(in_channels=out_channels // reduction,
                                       out_channels=out_channels // reduction,
                                       kernel_size=(1, kernel_size), stride=1, padding='same',
                                       dilation=dilation, bias=bias, groups=1, device=device)]
        elif separable:
            layers.append(SeparableConv2d(in_channels=out_channels // reduction,
                                          out_channels=out_channels // reduction,
                                          kernel_size=(kernel_size, kernel_size), bias=bias, device=device))
        else:
            layers.append(torch.nn.Conv2d(in_channels=out_channels // reduction,
                                          out_channels=out_channels // reduction,
                                          kernel_size=(kernel_size, kernel_size), stride=1, padding='same',
                                          dilation=dilation, bias=bias, groups=1, device=device))

        layers += [torch.nn.BatchNorm2d(out_channels // reduction, device=device),
                   torch.nn.PReLU(device=device),
                   torch.nn.Conv2d(in_channels=out_channels // reduction,
                                   out_channels=out_channels,
                                   kernel_size=(1, 1), stride=1, padding='same',
                                   dilation=dilation, bias=bias, groups=groups, device=device),
                   torch.nn.BatchNorm2d(out_channels, device=device)]
        if dr_rate > 0.:
            layers.append(torch.nn.Dropout2d(p=dr_rate))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x: tensor to be calculated.
        :return: tensor
        """
        return self.layers(x)


class InitialBlock(torch.nn.Module):
    """
    Initial Block of EFSNet.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dr_rate: float, reduction: int, kernel_size: int, stride: int,
                 bias: bool, device: str):
        """
        Inital Block.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Out channels (C) in BxCxHxW batch image tensor
        :param reduction: Reduction in desagregated block.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.non_linear_branch = DisaggregatedConvBlock(in_channels=in_channels,
                                                        out_channels=out_channels - in_channels,
                                                        dr_rate=dr_rate, reduction=reduction, kernel_size=kernel_size,
                                                        stride=stride, dilation=1, groups=1, bias=bias, factorize=False,
                                                        separable=False, device=device)
        self.linear_branch = torch.nn.MaxPool2d(kernel_size=(stride, stride), stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # NON-LINEAR BRANCH
        x1 = self.non_linear_branch(x)
        # LINEAR BRANCH
        x2 = self.linear_branch(x)
        return torch.cat([x1, x2], dim=1)


class DownsamplingBlock(torch.nn.Module):
    """
    Downsampling block of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, reduction: int, kernel_size: int,
                 stride: int, bias: bool, device: str):
        """
        DownSampling Block for dimension reduction.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        mxp11 = torch.nn.MaxPool2d(kernel_size=(stride, stride), stride=stride)
        cb11 = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1), stride=1, padding=0,
                               dilation=1, bias=bias, groups=1, device=device)
        bn11 = torch.nn.BatchNorm2d(out_channels, device=device)
        self.linear_branch = torch.nn.Sequential(mxp11, cb11, bn11)

        self.non_linear_branch = DisaggregatedConvBlock(in_channels=in_channels, out_channels=out_channels,
                                                        dr_rate=dr_rate, reduction=reduction,
                                                        kernel_size=kernel_size, stride=stride,
                                                        dilation=1, groups=1, bias=bias, factorize=False,
                                                        separable=False, device=device)

        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # NON-LINEAR BRANCH
        x2 = self.non_linear_branch(x)
        # LINEAR BRANCH
        x1 = self.linear_branch(x)
        # COMBINED
        return self.act_f(x1 + x2)


class FactorizedBlock(torch.nn.Module):
    """
    Downsampling block of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, reduction: int,
                 kernel_size: int, bias: bool, device: str):
        """
        Factorized Block layer constructor.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param reduction: Channel reduction in dissagregated block
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.non_linear_branch = DisaggregatedConvBlock(in_channels=in_channels, out_channels=out_channels,
                                                        dr_rate=dr_rate, reduction=reduction,
                                                        kernel_size=kernel_size, stride=1,
                                                        dilation=1, groups=1, bias=bias, factorize=True,
                                                        separable=False, device=device)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # NON-LINEAR BRANCH
        x1 = self.non_linear_branch(x)

        # COMBINED
        return self.act_f(x + x1)


class SDCBlock(torch.nn.Module):
    """
    SDC Block of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float, reduction: int,
                 kernel_size: int, bias: bool, groups: int, dilation: int, device: str):
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
        self.non_linear_branch = DisaggregatedConvBlock(in_channels=in_channels, out_channels=out_channels,
                                                        dr_rate=dr_rate, reduction=reduction,
                                                        kernel_size=kernel_size, stride=1,
                                                        dilation=dilation, groups=groups,
                                                        bias=bias, factorize=False,
                                                        separable=False, device=device)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # NON-LINEAR BRANCH
        x1 = self.non_linear_branch(x)

        # COMBINED
        return self.act_f(x + x1)


class CSDCBlock(torch.nn.Module):
    """
    Super SDC Block of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float,
                 reduction: int, kernel_size: int,
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
                     reduction=reduction, kernel_size=kernel_size,
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
    def __init__(self, in_channels: int, out_channels: int, dr_rate: float,
                 stride: int, bias: bool, device: str):
        """
        Upsample Module for enlarging image.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param device: Device to use for computation.
        """
        super().__init__()
        main = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2,
                                kernel_size=(1, 1), stride=1, padding='same',
                                dilation=1, bias=bias, device=device),
                torch.nn.BatchNorm2d(out_channels//2, device=device),
                InterpolateBilinear(stride=stride)]
        self.main_path = torch.nn.Sequential(*main)

        skip = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2,
                                kernel_size=(1, 1), stride=1, padding='same',
                                dilation=1, bias=bias, device=device),
                torch.nn.ConvTranspose2d(in_channels=out_channels//2, out_channels=out_channels//2,
                                         kernel_size=(stride, stride), stride=stride, padding=0,
                                         dilation=1, bias=bias, device=device),
                torch.nn.BatchNorm2d(out_channels//2, device=device)
                ]
        if dr_rate > 0:
            skip += [torch.nn.Dropout2d(dr_rate)]
        self.skip_path = torch.nn.Sequential(*skip)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x1: incoming tensor 1
        :param x2: incoming tensor 2
        :return: outgoing tensor
        """
        # MAIN PATH
        x1 = self.main_path(x1)
        # SKIP PATH
        x2 = self.skip_path(x2)
        return self.act_f(torch.cat([x1, x2], dim=1))


class ShuffleNet(torch.nn.Module):
    """
    ShuffleNet of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, dr_rate: float,
                 reduction: int, kernel_size: int,
                 bias: bool, groups: int, device: str):
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
        self.non_linear_branch = DisaggregatedConvBlock(in_channels=in_channels, out_channels=out_channels,
                                                        dr_rate=dr_rate, reduction=reduction, kernel_size=kernel_size,
                                                        stride=1, dilation=1, groups=groups, bias=bias,
                                                        factorize=False, separable=True, device=device)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        # PATH
        x1 = self.non_linear_branch(x)
        # COMBINED
        return self.act_f(x + x1)


class Encoder(torch.nn.Module):
    """
    Encoder of EFSNet
    """

    def __init__(self, in_channels: int, out_channels: int, initial_channels: int, dr_rate: float,
                 reduction: int, kernel_size: int, stride: int, bias: bool, groups: int,
                 num_factorized_blocks: int, num_csdc_blocks: int, num_sdc_per_csdc: int, device: str):
        """
        EFSNet Encoder Layer constructor.
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor
        :param initial_channels: Channels for the initial block.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param num_factorized_blocks: Num for factorized blocks.
        :param num_csdc_blocks: Num of super SDC Blocks.
        :param num_sdc_per_csdc: Num of SDC Blocks per superSDC Block.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.initial_block = InitialBlock(in_channels=in_channels, out_channels=initial_channels, dr_rate=dr_rate,
                                          reduction=reduction, kernel_size=kernel_size, stride=stride,
                                          bias=bias, device=device)
        self.downsampling_block1 = DownsamplingBlock(in_channels=initial_channels, out_channels=out_channels // 2,
                                                     dr_rate=dr_rate, reduction=reduction, kernel_size=kernel_size,
                                                     stride=stride, bias=bias, device=device)
        self.factorized_blocks = torch.nn.Sequential(*[FactorizedBlock(
            in_channels=out_channels // 2, out_channels=out_channels // 2, dr_rate=dr_rate,
            reduction=reduction, kernel_size=kernel_size,
            bias=bias, device=device
        ) for _ in range(num_factorized_blocks)])
        self.downsampling_block2 = DownsamplingBlock(in_channels=out_channels // 2, out_channels=out_channels,
                                                     dr_rate=dr_rate, reduction=reduction, kernel_size=kernel_size,
                                                     stride=stride, bias=bias, device=device)
        self.super_sdc_blocks = torch.nn.Sequential(*[
            CSDCBlock(in_channels=out_channels, out_channels=out_channels, dr_rate=dr_rate,
                      reduction=reduction, kernel_size=kernel_size, bias=bias,
                      groups=groups, device=device, num_sdc=num_sdc_per_csdc)
            for _ in range(num_csdc_blocks)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
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

    def __init__(self, in_channels: int, out_channels: int,
                 dr_rate: float, reduction: int, kernel_size: int,
                 stride: int, bias: bool, groups: int,
                 num_shufflenet: int, device: str):
        """
        EFSNet Decoder layer constructor
        :param in_channels: Input channels (C) in BxCxHxW batch image tensor.
        :param out_channels: Output channels (C) in BxCxHxW batch image tensor.
        :param dr_rate: Spatial Dropout rate applied channel wise.
        :param bias: Use bias or not.
        :param groups: convolution groups.
        :param num_shufflenet: Num of shuffle nets per upsampling step.
        :param device: Device to use for computation.
        """
        super().__init__()
        self.upsample_module1 = UpsamplingBlock(in_channels=in_channels, out_channels=in_channels // 2, dr_rate=dr_rate,
                                                stride=stride, bias=bias, device=device)
        self.shufflenet1 = torch.nn.Sequential(*[
            ShuffleNet(in_channels=in_channels // 2, out_channels=in_channels // 2, dr_rate=dr_rate,
                       reduction=reduction, kernel_size=kernel_size, bias=bias, groups=groups, device=device)
            for _ in range(num_shufflenet)
        ])

        self.upsample_module2 = UpsamplingBlock(in_channels=in_channels // 2, out_channels=out_channels,
                                                stride=stride, dr_rate=dr_rate, bias=bias, device=device)
        self.shufflenet2 = torch.nn.Sequential(*[
            ShuffleNet(in_channels=out_channels, out_channels=out_channels, dr_rate=dr_rate,
                       reduction=reduction, kernel_size=kernel_size, bias=bias, groups=groups // 2, device=device)
            for _ in range(num_shufflenet)
        ])

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """
        Forward call
        :param x1: incoming tensor 1 from EFSNet Encoder (first downsampling skip connection).
        :param x2: incoming tensor 2 from EFSNet Encoder (second downsampling skip connection).
        :param x3: incoming tensor 3 from EFSNet Encoder (main path).
        :return: outgoing tensor
        """

        x = self.upsample_module1(x3, x2)
        x = self.shufflenet1(x)
        x = self.upsample_module2(x, x1)
        x = self.shufflenet2(x)

        return x
