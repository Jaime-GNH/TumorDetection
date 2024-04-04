from typing import Tuple, Optional
import torch

from TumorDetection.utils.base import BaseClass
from TumorDetection.models.layers import Encoder, Decoder, DownsamplingBlock
from TumorDetection.utils.dict_classes import Device, Verbosity


class EFSNetClfSeg(torch.nn.Module, BaseClass):
    """
    Efficient Segmentation Network from https://ieeexplore.ieee.org/document/9063469
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (1, 256, 256), num_classes: int = 3,
                 out_channels: int = 128, dr_rate: float = 0.2, groups: int = 2, bias: bool = False,
                 num_factorized_blocks: int = 4, num_super_sdc_blocks: int = 2, num_sdc_per_supersdc: int = 4,
                 num_shufflenet: int = 2,
                 device: str = Device.get('device'), verbose: int = Verbosity.get('verbose')):
        """
        EFSNet model initialization
        :param input_shape: Input tensor image shape as CxHxW
        :param num_classes: Number of classes to classify in the classifier output
        :param out_channels: Encoder output channels.
        :param dr_rate: Global spatial dropout channel wise.
        :param groups: Number of groups in grouped convolutions.
        :param bias: Whether to use bias in layers or not.
        :param num_factorized_blocks: Number of factorized blocks to use in the encoder.
        :param num_super_sdc_blocks: Number of super sdc blocks to use in the encoder.
        :param num_sdc_per_supersdc: Number of shuffle dilated convolution block per sdc block to use in the encoder.
        :param num_shufflenet: Number of shuffle net blocks per upsample step to use in the decoder.
        :param device: Device identifier for computing.
        :param verbose:
        """
        super().__init__()
        self.kwargs = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'out_channels': out_channels,
            'dr_rate': dr_rate,
            'groups': groups,
            'bias': bias,
            'num_factorized_blocks': num_factorized_blocks,
            'num_super_sdc_blocks': num_super_sdc_blocks,
            'num_sdc_per_supersdc': num_sdc_per_supersdc,
            'num_shufflenet': num_shufflenet,
            'device': device,
            'verbose': verbose
        }
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder = Encoder(in_channels=self.input_shape[0], out_channels=out_channels,
                               dr_rate=dr_rate, bias=bias, groups=groups,
                               num_factorized_blocks=num_factorized_blocks,
                               num_super_sdc_blocks=num_super_sdc_blocks,
                               num_sdc_per_supersdc=num_sdc_per_supersdc,
                               device=self.device)

        self.label_ds = DownsamplingBlock(in_channels=out_channels,
                                          out_channels=out_channels // 16,
                                          dr_rate=dr_rate, bias=bias, device=self.device)
        self.label_fl = torch.nn.Flatten()
        self.labeler = torch.nn.Linear(
            in_features=(out_channels // 16) * (self.input_shape[-2] // 16) * (
                        self.input_shape[-1] // 16),
            out_features=self.num_classes, device=self.device
        )

        self.decoder = Decoder(in_channels=out_channels, dr_rate=dr_rate,
                               bias=bias, groups=groups,
                               num_shufflenet=num_shufflenet,
                               device=self.device)
        self.segment = torch.nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                                kernel_size=(3, 3), stride=2, padding=1, output_padding=1,
                                                device=self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        x1, x2, x3 = self.encoder(x)
        xs = self.decoder(x1, x2, x3)
        xl = self.label_ds(x3)
        xl = self.label_fl(xl)
        label = self.labeler(xl)
        segment = self.segment(xs)

        return segment, label


class EFSNetSeg(torch.nn.Module, BaseClass):
    """
    Efficient Segmentation Network from https://ieeexplore.ieee.org/document/9063469
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (1, 256, 256), num_classes: int = 3,
                 out_channels: int = 128, dr_rate: float = 0.2, groups: int = 2, bias: bool = False,
                 num_factorized_blocks: int = 4, num_super_sdc_blocks: int = 2, num_sdc_per_supersdc: int = 4,
                 num_shufflenet: int = 2, clamp_output: Optional[float] = 10.,
                 device: str = Device.get('device'), verbose: int = Verbosity.get('verbose')):
        """
        EFSNet model initialization
        :param input_shape: Input tensor image shape as CxHxW
        :param num_classes: Number of classes to classify in the classifier output
        :param out_channels: Encoder output channels.
        :param dr_rate: Global spatial dropout channel wise.
        :param groups: Number of groups in grouped convolutions.
        :param bias: Whether to use bias in layers or not.
        :param num_factorized_blocks: Number of factorized blocks to use in the encoder.
        :param num_super_sdc_blocks: Number of super sdc blocks to use in the encoder.
        :param num_sdc_per_supersdc: Number of shuffle dilated convolution block per sdc block to use in the encoder.
        :param num_shufflenet: Number of shuffle net blocks per upsample step to use in the decoder.
        :param clamp_output: Value for clamping values in segmentation (exploding gradients).
        :param device: Device identifier for computing.
        :param verbose: Verbosity level.
        """
        super().__init__()
        self.kwargs = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'out_channels': out_channels,
            'dr_rate': dr_rate,
            'groups': groups,
            'bias': bias,
            'num_factorized_blocks': num_factorized_blocks,
            'num_super_sdc_blocks': num_super_sdc_blocks,
            'num_sdc_per_supersdc': num_sdc_per_supersdc,
            'num_shufflenet': num_shufflenet,
            'clamp_output': clamp_output,
            'device': device,
            'verbose': verbose
        }
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder = Encoder(in_channels=self.input_shape[0], out_channels=out_channels,
                               dr_rate=dr_rate, bias=bias, groups=groups,
                               num_factorized_blocks=num_factorized_blocks,
                               num_super_sdc_blocks=num_super_sdc_blocks,
                               num_sdc_per_supersdc=num_sdc_per_supersdc,
                               device=self.device)

        self.decoder = Decoder(in_channels=out_channels, dr_rate=dr_rate,
                               bias=bias, groups=groups,
                               num_shufflenet=num_shufflenet,
                               device=self.device)
        self.segment = torch.nn.ConvTranspose2d(in_channels=16, out_channels=self.num_classes,
                                                kernel_size=(3, 3), stride=2, padding=1, output_padding=1,
                                                device=self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        x1, x2, x3 = self.encoder(x)
        xs = self.decoder(x1, x2, x3)
        segment = self.segment(xs)

        if (clamp_val := self.kwargs.get('clamp_output')) is not None:
            return segment.clamp(min=-clamp_val, max=clamp_val)
        else:
            return segment
