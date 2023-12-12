import torch_geometric as tg
import torch

from TumorDetection.Utils.Utils import tup2hypergraph, apply_function2list
from TumorDetection.Utils.BaseClass import BaseClass
from TumorDetection.Utils.DictClasses import BaseGNNInit, ImageGNNInit, ConvLayerKwargs


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
        h_dims = kwargs.get('h_size')
        conv_layers = kwargs.get('conv_layer_type')
        if not isinstance(conv_layers, list):
            conv_layers = [conv_layers]
        if len(conv_layers) < len(h_dims):
            conv_layers = [conv_layers[0]]*len(h_dims)

        if conv_layer_kwargs is None:
            conv_layer_kwargs = {}
        conv_layer_kwargs = self._default_config(ConvLayerKwargs, **conv_layer_kwargs)
        self.convs = torch.nn.ModuleList([
            *[cl(in_channels=-1, out_channels=h_dims[i],
                 hidden_channels=h_dims[i],
                 **conv_layer_kwargs)
              for i, cl in enumerate(conv_layers)]
        ])
        self.out = tg.nn.Linear(h_dims[-1], 1)  # Binary classification

    def forward(self, batch):
        """

        :param batch:
        :return:
        """
        x, edge_index = batch.x, batch.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        return self.out(x).clamp(-4, 4)


class ImageGNN(torch.nn.Module, BaseClass):
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
        kwargs = self._default_config(ImageGNNInit, **kwargs)
        self.hypernode_patch_dims = kwargs.get('hypernode_patch_dims')
        self.kernels = [kwargs.get('kernel_kind')]*len(self.hypernode_patch_dims) + [kwargs.get('last_kernel')]
        h_dims = kwargs.get('h_size')
        conv_layers = kwargs.get('conv_layer_type')
        if not isinstance(conv_layers, list):
            conv_layers = [conv_layers]
        if len(conv_layers) < len(h_dims):
            conv_layers = [conv_layers[0]]*len(h_dims)
        if conv_layer_kwargs is None:
            conv_layer_kwargs = {}
        conv_layer_kwargs = self._default_config(ConvLayerKwargs, **conv_layer_kwargs)
        self.convs = torch.nn.ModuleList([
            torch.nn.ModuleList([
                cl(in_channels=-1, out_channels=h_dims[i],
                   hidden_channels=h_dims[i],
                   **conv_layer_kwargs)
                for i, cl in enumerate(conv_layers)
            ])
            for _ in range(len(self.hypernode_patch_dims) + 1)
        ])
        self.projections = torch.nn.ModuleList([
            tg.nn.Linear(-1, patch**2)
            for patch in self.hypernode_patch_dims
        ])
        self.solutions = torch.nn.ModuleList([
            tg.nn.Linear(-1, 1)
            for _ in range(len(self.hypernode_patch_dims))
        ])  # Binary classification

        self.out = tg.nn.Linear(-1, 1)

    def forward(self, batch):
        """
        batch of images.
        :param batch:
        :return:
        """
        batch_mask = batch.get('mask')
        solutions = []
        node_masks = []
        for i, patch in enumerate(self.hypernode_patch_dims):
            batch, node_mask = self.image2graph(batch, patch, self.kernels[i])
            node_masks.append(node_mask)
            x = self.conv_block(self.convs[i], batch.x, batch.edge_index)
            x = self.projections[i](x)
            batch.x = x
            solutions.append(self.solutions[i](batch.x).clamp(-4, 4))
            batch = self.graph2image(batch, batch_mask, node_mask, solutions[-1])
        return solutions, batch, node_masks
        # if len(batch.x) > 0:
        #     x_conv = self.conv_block(self.convs[i], batch.x, batch.edge_index)
        #     x = self.projections[i](x_conv)
        #     if i < len(self.hypernode_patch_dims) - 1:
        #         batch.x = x  # torch.add(batch.x, x)
        #         solution = self.solutions[i](batch.x)
        #         batch = self.graph2image(batch, batch_mask, node_mask, solution)
        #     else:
        #         return self.out(batch.x).clamp(-4, 4), batch, node_mask
        # else:
        #     batch = self.graph2image(batch, batch_mask, node_mask, None)
        #     batch, node_mask = self.image2graph(batch, self.hypernode_patch_dims[-1], self.kernels[i])
        #     return self.out(batch.x), batch, node_mask
        # TODO:
        #  2. Use random crop to resize images to (256x256). -> DONE
        #       Use only tumor images (some-patch will be similar to normal). -> NOT DONE
        #  3. Try without parts of preprocessing. Inverse-color is mandatory. -> DOING
        #  5. Predict tumor/no-tumor in betweeen. Predict benign-malignant on last
        #
        #     batch = self.graph2image(batch, batch_mask, node_mask, solution)
        #
        # batch, _ = self.image2graph(batch, 1, self.kernels[-1])
        # batch.x = self.conv_block(self.convs[-1], batch.x, batch.edge_index)
        # return self.out(batch.x).clamp(-4, 4)

    @staticmethod
    def conv_block(convs, x, edge_index):
        """
        Applies a convolutional block to nodes
        :param convs:
        :param x:
        :param edge_index:
        :return:
        """
        for conv in convs:
            x = conv(x, edge_index)
        return x.relu()

    @staticmethod
    def image2graph(batch, patch_dim, kernel_kind):
        """
        Converts a batch of images to a Data Batch
        :param batch: tup
            (img, mask, current_solution)
        :param patch_dim: (int)
        :param kernel_kind: (str)
        :return:
        """
        new_batch = list((i, m, cs) for i, m, cs in zip(batch.get('image'),
                                                        batch.get('mask'),
                                                        batch.get('current_solution', [None]*len(batch['image']))
                                                        )
                         )
        graph, mask = tuple(zip(
            *apply_function2list(
                new_batch,
                tup2hypergraph,
                image_idx=0,
                mask_idx=1,
                current_solution_idx=2,
                hypernode_patch_dim=patch_dim,
                kernel_kind=kernel_kind
            )
        ))
        return tg.data.Batch.from_data_list(graph), torch.stack(mask).flatten()

    @staticmethod
    def graph2image(batch, batch_mask, node_mask, solution=None):
        """
        Converts a DataBatch to a batch of images, masks and current_solutions
        :param batch:
        :param batch_mask:
        :param node_mask:
        :param solution:
        :return:
        """
        zeros = torch.zeros((node_mask.size(0), batch.x.size(1)),
                            device=node_mask.device,
                            dtype=batch.x.dtype)
        zeros[node_mask > 0] = batch.x
        batch_imgs = zeros.reshape(len(batch.original_shape), *batch.original_shape[0])

        if solution is not None:
            zeros = torch.zeros((node_mask.size(0), solution.size(1)), device=node_mask.device)
            zeros[node_mask > 0] = solution.to(zeros.dtype)
            hypernode_dim = batch.hypernode_patch_dim[0].to(device=zeros.device)
            batch_solution = (
                torch.repeat_interleave(
                    torch.repeat_interleave(zeros
                                            .reshape(len(batch.original_shape),
                                                     batch.original_shape[0][0] // batch.hypernode_patch_dim[0],
                                                     batch.original_shape[0][0] // batch.hypernode_patch_dim[0]
                                                     ),
                                            hypernode_dim,
                                            dim=1
                                            ),
                    hypernode_dim,
                    dim=2)
            )
            return {'image': batch_imgs, 'mask': batch_mask, 'current_solution': batch_solution}
        else:
            return {'image': batch_imgs, 'mask': batch_mask}

