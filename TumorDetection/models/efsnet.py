from typing import Tuple, Optional
import torch

from TumorDetection.utils.base import BaseClass
from TumorDetection.models.layers import Encoder, Decoder
from TumorDetection.utils.dict_classes import Device, Verbosity


class EFSNet(torch.nn.Module, BaseClass):
    """
    Efficient Segmentation Network from https://ieeexplore.ieee.org/document/9063469
    with changes.
    """
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 256, 256), num_classes: int = 3,
                 latent_size: int = 128, initial_channels: int = 8, dr_rate: float = 0.2,
                 reduction: int = 2, kernel_size: int = 3, stride: int = 2, groups: int = 2, bias: bool = False,
                 num_factorized_blocks: int = 4, num_csdc_blocks: int = 2, num_sdc_per_csdc: int = 4,
                 num_shufflenet: int = 2, clamp_output: Optional[float] = 10.,
                 device: str = Device.get('device'), verbose: int = Verbosity.get('verbose')):
        """
        EFSNet model initialization
        :param input_shape: Input tensor image shape as CxHxW
        :param num_classes: Number of classes to classify in the classifier output
        :param latent_size: Encoder output channels.
        :param initial_channels: Output channels in InitialBlock
        :param dr_rate: Global spatial dropout channel wise.
        :param reduction: Global Channel reduction in Dissagregated Blocks
        :param kernel_size: Global kernel size
        :param stride: Global stride for downsampling/upsampling.
        :param groups: Number of groups in grouped convolutions.
        :param bias: Whether to use bias in layers or not.
        :param num_factorized_blocks: Number of factorized blocks to use in the encoder.
        :param num_csdc_blocks: Number of super sdc blocks to use in the encoder.
        :param num_sdc_per_csdc: Number of shuffle dilated convolution block per sdc block to use in the encoder.
        :param num_shufflenet: Number of shuffle net blocks per upsample step to use in the decoder.
        :param clamp_output: Value for clamping values in segmentation (exploding gradients).
        :param device: Device identifier for computing.
        :param verbose: Verbosity level.
        """
        super().__init__()
        self.kwargs = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'latent_size': latent_size,
            'initial_channels': initial_channels,
            'dr_rate': dr_rate,
            'reduction': reduction,
            'kernel_size': kernel_size,
            'stride': stride,
            'groups': groups,
            'bias': bias,
            'num_factorized_blocks': num_factorized_blocks,
            'num_csdc_blocks': num_csdc_blocks,
            'num_sdc_per_csdc': num_sdc_per_csdc,
            'num_shufflenet': num_shufflenet,
            'clamp_output': clamp_output,
            'device': device,
            'verbose': verbose
        }
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder = Encoder(in_channels=self.input_shape[0], out_channels=latent_size,
                               initial_channels=initial_channels, dr_rate=dr_rate, reduction=reduction,
                               kernel_size=kernel_size, stride=stride, bias=bias, groups=groups,
                               num_factorized_blocks=num_factorized_blocks, num_csdc_blocks=num_csdc_blocks,
                               num_sdc_per_csdc=num_sdc_per_csdc, device=self.device)

        self.decoder = Decoder(in_channels=latent_size, out_channels=latent_size // 8,
                               dr_rate=dr_rate, reduction=reduction, kernel_size=kernel_size, stride=stride,
                               bias=bias, groups=groups, num_shufflenet=num_shufflenet, device=self.device)
        self.segment = torch.nn.ConvTranspose2d(in_channels=latent_size // 8, out_channels=self.num_classes,
                                                kernel_size=(kernel_size, kernel_size), stride=stride,
                                                padding=1, output_padding=1, device=self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward call
        :param x: incoming tensor
        :return: outgoing tensor
        """
        x1, x2, x3 = self.encoder(x)
        xs = self.decoder(x1, x2, x3)
        segment = self.segment(xs)

        if clamp_val := self.kwargs.get('clamp_output'):
            return segment.clamp(min=-clamp_val, max=clamp_val)
        else:
            return segment
