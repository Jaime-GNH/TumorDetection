import torch_geometric as tg
import torch

from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import BaseGNNInit, ConvLayerKwargs, HyperConvLayerKwargs


class BaseGNN(torch.nn.Module, BaseClass):
    """
    Base GNN for graph and hypergraph convolutional netwroks
    """
    def __init__(self, conv_layer_kwargs=None, **kwargs):
        """
        GNN Constructor.
        Layer definitions
        :param conv_layer_kwargs: (dict)
        """
        super().__init__()
        kwargs = self._default_config(BaseGNNInit, **kwargs)
        self.num_classes = kwargs.get('num_classes')
        h_dims = kwargs.get('h_size')
        conv_layers = kwargs.get('conv_layer_type')
        if not isinstance(conv_layers, list):
            conv_layers = [conv_layers]
        if len(conv_layers) < len(h_dims):
            conv_layers = [conv_layers[0]]*len(h_dims)

        if conv_layer_kwargs is None:
            conv_layer_kwargs = {}
        if issubclass(conv_layers[0], tg.nn.HypergraphConv):
            conv_layer_kwargs = self._default_config(HyperConvLayerKwargs, **conv_layer_kwargs)
            self.convs = torch.nn.ModuleList([
                *[cl(in_channels=-1, out_channels=h_dims[i],
                     **conv_layer_kwargs)
                  for i, cl in enumerate(conv_layers)]
            ])
        else:
            conv_layer_kwargs = self._default_config(ConvLayerKwargs, **conv_layer_kwargs)
            self.convs = torch.nn.ModuleList([
                *[cl(in_channels=-1, out_channels=h_dims[i],
                     hidden_channels=h_dims[i],
                     **conv_layer_kwargs)
                  for i, cl in enumerate(conv_layers)]
            ])
        self.out = tg.nn.Linear(h_dims[-1], self.num_classes)

    def forward(self, batch):
        """

        :param batch:
        :return:
        """
        for i, conv in enumerate(self.convs):
            batch.x = conv(batch.x, batch.edge_index)

        return self.out(batch.x).clamp(-4, 4)
