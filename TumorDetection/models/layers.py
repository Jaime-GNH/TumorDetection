import torch
import torch.nn.functional as tfn


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, device,
                 groups=1, apply_conv=True, use_batchnorm=True, use_act=True, transpose=False):
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

    def forward(self, x):
        return self.layers(x)


class SeparableConv2d(torch.nn.Module):
    """
    Separable DepthWise Conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias, device):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                         groups=in_channels, bias=bias, padding=1, device=device)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels,
                                         kernel_size=1, bias=bias, device=device)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def channel_shuffle(x, groups):
    """
    Channel Shuffle -> torch.nn.Channel Shuffle: `RuntimeError: derivative for channel_shuffle is not implemented`
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
    def __init__(self, in_channels, dr_rate, bias, name, device):
        super().__init__()
        self.name = name
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=16-in_channels, kernel_size=(3, 3),
                              stride=2, padding=1, dilation=1, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)
        self.mxp21 = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x1 = self.cb11(x)
        x1 = self.spr11(x1)
        x2 = self.mxp21(x)
        return torch.cat([x1, x2], dim=1)


class DownsamplingBlock(torch.nn.Module):
    """
    Downsampling block of EFSNet
    """
    def __init__(self, in_channels, filters, dr_rate, bias, name, device):
        super().__init__()
        self.name = name
        self.mxp11 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.bn11 = torch.nn.BatchNorm2d(in_channels, device=device)
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device, use_act=False)

        self.cb21 = ConvBlock(in_channels=in_channels, out_channels=filters // 4, kernel_size=(2, 2),
                              padding=0, stride=2, dilation=1, bias=bias, device=device)
        self.cb22 = ConvBlock(in_channels=filters // 4, out_channels=filters // 4, kernel_size=(3, 3),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.cb23 = ConvBlock(in_channels=filters // 4, out_channels=filters, kernel_size=(1, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.spr21 = torch.nn.Dropout2d(p=dr_rate)

        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x):
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
    def __init__(self, in_channels, filters, dr_rate, bias, name, device):
        super().__init__()
        self.name = name
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=filters // 4, kernel_size=(1, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.cb12 = ConvBlock(in_channels=filters // 4, out_channels=filters // 4, kernel_size=(1, 3),
                              padding='same', stride=1, dilation=1, bias=bias, device=device, use_act=False)

        self.cb13 = ConvBlock(in_channels=filters // 4, out_channels=filters // 4, kernel_size=(3, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)

        self.cb14 = ConvBlock(in_channels=filters // 4, out_channels=filters, kernel_size=(1, 1),
                              padding='same', stride=1, dilation=1, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)

        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x):
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
    def __init__(self, in_channels, filters, dr_rate, bias, groups, dilation, name, device):
        super().__init__()
        self.name = name
        self.groups = groups
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=filters // 4, kernel_size=(1, 1),
                              stride=1, groups=groups, padding='same', dilation=1, bias=bias, device=device)
        # self.chs11 = torch.nn.ChannelShuffle(groups=groups)

        self.cb12 = ConvBlock(in_channels=filters // 4, out_channels=filters // 4, kernel_size=(3, 3),
                              stride=1, padding='same', dilation=dilation, bias=bias, device=device,
                              use_act=False)

        self.cb13 = ConvBlock(in_channels=filters // 4, out_channels=filters, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, groups=groups, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x):
        # PATH
        x1 = self.cb11(x)
        # x1 = self.chs11(x1)
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

    def __init__(self, in_channels, filters, dr_rate, bias, groups, name, device, num_sdc=4):
        super().__init__()
        self.name = name
        self.super_sdc = torch.nn.Sequential(*[
            SDCBlock(in_channels=in_channels, filters=filters, dr_rate=dr_rate,
                     bias=bias, groups=groups, dilation=2 ** k, name=self.name+f'_{k+1}',
                     device=device)
            for k in range(num_sdc)])

    def forward(self, x):
        return self.super_sdc(x)


class UpsamplingBlock(torch.nn.Module):
    """
    Upsampling Block of EFSNet
    """

    def __init__(self, in_channels, filters, dr_rate, bias, name, device):
        super().__init__()
        self.name = name
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device,
                              use_act=False)
        self.bnact11 = ConvBlock(in_channels=in_channels, out_channels=filters, kernel_size=None,
                                 stride=None, padding=None, dilation=None, bias=None, device=device,
                                 apply_conv=False)

        self.cb21 = ConvBlock(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device)
        self.ct21 = ConvBlock(in_channels=filters, out_channels=filters, kernel_size=(2, 2),
                              stride=2, padding=0, dilation=1, bias=bias, device=device,
                              transpose=True)
        self.spr21 = torch.nn.Dropout2d(p=dr_rate)
        self.act_f = torch.nn.PReLU(device=device)

    def forward(self, x1, x2):
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
    def __init__(self, in_channels, filters, dr_rate, bias, groups, name, device):
        super().__init__()
        self.name = name
        self.groups = groups
        self.cb11 = ConvBlock(in_channels=in_channels, out_channels=filters // 4, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, groups=groups, bias=bias, device=device,
                              use_act=False)
        self.act11 = torch.nn.ReLU()
        # self.chs11 = torch.nn.ChannelShuffle(groups)
        self.scb11 = SeparableConv2d(in_channels=filters // 4, out_channels=filters // 4, kernel_size=(3, 3),
                                     bias=False, device=device)
        self.bn11 = torch.nn.BatchNorm2d(filters // 4, device=device)
        self.cb12 = ConvBlock(in_channels=filters // 4, out_channels=filters, kernel_size=(1, 1),
                              stride=1, padding='same', dilation=1, bias=bias, device=device)
        self.spr11 = torch.nn.Dropout2d(p=dr_rate)
        self.act_f = torch.nn.ReLU()

    def forward(self, x):
        # PATH
        x1 = self.cb11(x)
        x1 = self.act11(x1)
        # x1 = self.chs11(x1)
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
    def __init__(self, in_channels, filters, dr_rate, bias, groups,
                 num_factorized_blocks, num_super_sdc_blocks,
                 num_sdc_per_supersdc,
                 name, device):
        super().__init__()
        # Num_SDC_per_supersdc = 4
        self.name = name
        self.initial_block = InitialBlock(in_channels=in_channels, dr_rate=dr_rate, bias=bias,
                                          name=self.name + '_IB', device=device)
        self.downsampling_block1 = DownsamplingBlock(in_channels=16, filters=filters // 2, dr_rate=dr_rate,
                                                     bias=bias, name=self.name+'_DS1', device=device)
        self.factorized_blocks = torch.nn.Sequential(*[FactorizedBlock(
            in_channels=filters // 2, filters=filters // 2, dr_rate=dr_rate, bias=bias,
            name=self.name+f'_FB_{fb+1}', device=device
        ) for fb in range(num_factorized_blocks)])
        self.downsampling_block2 = DownsamplingBlock(in_channels=filters // 2, filters=filters, dr_rate=dr_rate,
                                                     bias=bias, name=self.name + '_DS2', device=device)
        self.super_sdc_blocks = torch.nn.Sequential(*[
            SuperSDCBlock(in_channels=filters, filters=filters, dr_rate=dr_rate, bias=bias,
                          groups=groups, name=self.name + f'_SSDC_{sdc+1}', device=device,
                          num_sdc=num_sdc_per_supersdc)
            for sdc in range(num_super_sdc_blocks)
        ])

    def forward(self, x):
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

    def __init__(self, in_channels, dr_rate, bias, groups,
                 num_shufflenet,
                 name, device):
        super().__init__()
        # Num shufflenet == 2
        self.name = name
        self.upsample_module1 = UpsamplingBlock(in_channels=in_channels, filters=in_channels//4, dr_rate=dr_rate,
                                                bias=bias, name=self.name+'_US1', device=device)
        self.shufflenet1 = torch.nn.Sequential(*[
            ShuffleNet(in_channels=in_channels // 2, filters=in_channels // 2, dr_rate=dr_rate,
                       bias=bias, groups=groups, name=self.name+f'_SN1_{sh+1}', device=device)
            for sh in range(num_shufflenet)
        ])

        self.upsample_module2 = UpsamplingBlock(in_channels=in_channels // 2, filters=8,
                                                dr_rate=dr_rate, bias=bias, name=self.name + '_US2', device=device)
        self.shufflenet2 = torch.nn.Sequential(*[
            ShuffleNet(in_channels=16, filters=16, dr_rate=dr_rate,
                       bias=bias, groups=groups // 2, name=self.name + f'_SN2_{sh + 1}', device=device)
            for sh in range(num_shufflenet)
        ])

    def forward(self, x1, x2, x3):

        x = self.upsample_module1(x3, x2)
        x = self.shufflenet1(x)
        x = self.upsample_module2(x, x1)
        x = self.shufflenet2(x)

        return x
