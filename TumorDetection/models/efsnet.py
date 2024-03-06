from typing import Tuple
import torch

from TumorDetection.utils.base import BaseClass
from TumorDetection.models.layers import Encoder, Decoder, DownsamplingBlock
from TumorDetection.utils.dict_classes import EFSNetInit


class EFSNet(torch.nn.Module, BaseClass):
    """
    Efficient Segmentation Network from https://ieeexplore.ieee.org/document/9063469
    """

    def __init__(self, **kwargs):
        """
        EFSNet model definition.
        :keyword device: (str)
            Device identifier for computing.
        :keyword input_shape: (Tuple[int, int, int])
            Input tensor image shape as CxHxW
        :keyword num_classes: (int)
            Number of classes to classify in the classifier output
        :keyword out_channels: (int)
            Encoder output channels.
        :keyword dr_rate: (float)
            Global spatial dropout channel wise.
        :keyword bias: (bool)
            Whether to use bias in layers or not.
        :keyword groups: (int)
            Number of groups in grouped convolutions.
        :keyword num_factorized_blocks: (int)
            Number of factorized blocks to use in the encoder.
        :keyword num_super_sdc_blocks: (int)
            Number of super sdc blocks to use in the encoder.
        :keyword num_sdc_per_supersdc: (int)
            Number of shuffle dilated convolution block per sdc block to use in the encoder.
        :keyword num_shufflenet: (int)
            Number of shuffle net blocks per upsample step to use in the decoder.
        """
        super().__init__()
        kwargs = self._default_config(EFSNetInit, **kwargs)
        self.kwargs = kwargs
        self.device = kwargs.get('device')
        self.input_shape = kwargs.get('input_shape')
        self.num_classes = kwargs.get('num_classes')
        self.encoder = Encoder(in_channels=self.input_shape[0], out_channels=kwargs.get('out_channels'),
                               dr_rate=kwargs.get('dr_rate'), bias=kwargs.get('bias'), groups=kwargs.get('groups'),
                               num_factorized_blocks=kwargs.get('num_factorized_blocks'),
                               num_super_sdc_blocks=kwargs.get('num_super_sdc_blocks', 2),
                               num_sdc_per_supersdc=kwargs.get('num_sdc_per_supersdc', 4),
                               device=self.device)

        self.label_ds = DownsamplingBlock(in_channels=kwargs.get('out_channels'),
                                          out_channels=kwargs.get('out_channels') // 16,
                                          dr_rate=kwargs.get('dr_rate'), bias=kwargs.get('bias'), device=self.device)
        self.label_fl = torch.nn.Flatten()
        self.labeler = torch.nn.Linear(
            in_features=(kwargs.get('out_channels') // 16) * (self.input_shape[-2] // 16) * (
                        self.input_shape[-1] // 16),
            out_features=self.num_classes, device=self.device
        )

        self.decoder = Decoder(in_channels=kwargs.get('out_channels'), dr_rate=kwargs.get('dr_rate'),
                               bias=kwargs.get('bias'), groups=kwargs.get('groups'),
                               num_shufflenet=kwargs.get('num_shufflenet', 2),
                               device=self.device)
        self.segment = torch.nn.ConvTranspose2d(in_channels=16, out_channels=2,
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
