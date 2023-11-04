import torch_geometric as tg
import torch

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import HyperGNNInit


class HyperGNN(torch.nn.Module, BaseClass):
    """

    """
    def __init__(self, **kwargs):
        """
        GNN Constructor.
        Layer definitions
        """
        super().__init__()
        kwargs = self._default_config(HyperGNNInit)
        h_dims = kwargs.get('h_dims')

        self.convs = torch.nn.ModuleList([
            # TODO: Fill with layers
        ])

        self.out = tg.nn.Linear(h_dims[-1], h_dims).relu_()

    def forward(self, batch):
        """

        :param batch:
        :return:
        """
        for conv in self.convs:
            batch.x_dict = conv(batch.x_dict, batch.edge_index_dict)

        return self.out(batch.x_dict)
