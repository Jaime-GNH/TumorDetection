import os
import cv2
import random

import torch
from time import perf_counter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from TumorDetection.Data.Loader import ImageLoader, DataPathLoader
from TumorDetection.Utils.DictClasses import (DataPathDir, ReportingPathDir,
                                              BaseClassMap, MappedClassValues,
                                              ImageToGraphCall, GraphDataLoaderCall)
from TumorDetection.Utils.Utils import calculate_dilations
from TumorDetection.Utils.Viewer import Viewer
from TumorDetection.Data.Preprocess import Preprocessor
from TumorDetection.Models.GNN.ImageToGraph import ImageToGraph
from TumorDetection.Data.GraphDataset import GraphDataset
from TumorDetection.Utils.Plotter import plot_graph

VIEW_IMAGES = False
VIEW_GRAPHS = False
REDUCE_DIM = False
CHECK_KERNEL_TIME = False
NORMALLOADER = True
CALCULATE_DILATIONS = True
CALCULATE_ALL_GRAPHS = False
CHECK_BATCHES = False
GRAPHDATALOADER = False
PICKLEDATALOADER = False

if __name__ == "__main__":
    print('Torch cuda:', torch.cuda.is_available())
    if NORMALLOADER:
        # 1. TEST DATA
        # 1.1 TEST DATAPATH -> DONE
        Dp = DataPathLoader(dir_path=DataPathDir.get('dir_path'))

        # 1.2 TEST IMAGE LOADER -> DONE
        paths_classes = Dp(map_classes=BaseClassMap.to_dict())
        result = ImageLoader()(paths_classes, class_values=MappedClassValues.to_dict())

        # 1.3 TEST PREPROCESSOR -> DONE
        prep_result = Preprocessor()([r[2] for r in result])
        result = [(r[0], r[1], pr, r[3]) for r, pr in zip(result, prep_result)]
        Ig = ImageToGraph()
        # 2. TEST UTILITIES
        # 2.1 TEST VIEWER -> DONE
        if VIEW_IMAGES:
            idx = random.choice(range(len(result)))
            graph = Ig(result[idx], dilations=1)
            print(f'Shape (height, width): {result[idx][2][:,:,0].shape}')
            img = graph.x[torch.where(graph.y == 1.)[0]]
            mask = graph.y[torch.where(graph.y == 1.)[0]]
            for k in range(result[idx][2].shape[-1]):
                Viewer.show_masked_image(result[idx][2][:, :, k], result[idx][3],
                                         win_title=f'Image: {os.path.splitext(os.path.basename(result[idx][0]))[0]}.'
                                                   f' Class: {result[idx][1][0]}.'
                                                   f' Preprocesado: {k}')

        if CALCULATE_DILATIONS:
            num_hops = 4
            dils = [list(calculate_dilations(r[2].shape, num_hops, 'star')) for r in result]
            print(dils)
        # 3. TEST MODELS
        # 3.1 GRAPH NEURAL NETWORK -> ON-GOING
        # 3.1.1 IMAGE TO HOMOGENENOUS GRAPH CONVERSION

        if VIEW_GRAPHS:
            idx = 0
            if REDUCE_DIM:
                # (64, 64) shape
                image2graph = [(result[idx][0],
                                result[idx][1],
                                cv2.resize(result[idx][2], (64, 64), interpolation=cv2.INTER_LINEAR),
                                cv2.resize(result[idx][3], (64, 64), interpolation=cv2.INTER_NEAREST_EXACT))]
                dilations = (1, 7, 31)
            else:
                # (471, 562) shape
                image2graph = [result[0]]
                dilations = ImageToGraphCall.get('dilations')
            print(f'Shape (height, width): {image2graph[0][2][:,:,0].shape}')
        elif CHECK_KERNEL_TIME:
            image2graph = [result[0]]
            dilations = 1
            for kernel_kind in ['corner', 'hex', 'cross', 'diagonal', 'star', 'square']:
                it = perf_counter()
                g = Ig(image2graph, dilations=dilations, kernel_kind=kernel_kind)
                print(f'Kernel: {kernel_kind}. Edge index dim: {g[0].edge_index.size(dim=1)}'
                      f' Time: {round(perf_counter() - it, 2)} s')

        else:
            image2graph = result
            dilations = ImageToGraphCall.get('dilations')
        if CHECK_BATCHES:
            graphs = Ig(image2graph[:4], dilations=3, kernel_kind='hex')
            print(graphs)
            batch = Batch.from_data_list(graphs)
            print(batch)
        elif CALCULATE_ALL_GRAPHS:
            it = perf_counter()
            graphs = Ig(image2graph, dilations=dilations)
            print(f'Elapsed time: {round(perf_counter()-it, 5)} s')
        else:
            graphs = Ig(image2graph[:16], dilations=dilations)
        if not isinstance(graphs, list):
            graphs = [graphs]
        if VIEW_GRAPHS:
            it = perf_counter()
            pos = graphs[0].pos
            plot_graph(graphs[0],
                       path=os.path.join(
                           ReportingPathDir.get('dir_path'),
                           f'graph_prueba_{image2graph[0][2][:,:,0].shape[0]}x{image2graph[0][2][:,:,0].shape[1]}'
                       ),
                       format='html',
                       show=False,
                       edges_show=[0, pos[:, 1].max()//2, pos[:, 1].max(),
                                   len(pos)//2, len(pos)//2 + (pos[:, 1].max()//2),
                                   len(pos)//2 + pos[:, 1].max(),
                                   len(pos)-pos[:, 1].max()-1,
                                   len(pos) - (pos[:, 1].max()//2) - 2, len(pos)-1])
            print(f'Elapsed time: {round(perf_counter()-it, 5)} s')

    # 3.1.2 DATALOADERS
    if GRAPHDATALOADER:
        train_dataloader = DataLoader(GraphDataset(DataPathDir.get('dir_path'),
                                                   train=True),
                                      **GraphDataLoaderCall.to_dict())
        test_dataloader = DataLoader(GraphDataset(DataPathDir.get('dir_path'),
                                                  train=False),
                                     **GraphDataLoaderCall.to_dict())
        print('dataloaders done!')
        it = perf_counter()
        next_iter_tr = next(iter(train_dataloader))
        print(next_iter_tr)
        print(f'Iter', round(perf_counter()-it, 5), 's')
        next_iter_ts = next(iter(test_dataloader))
        print(next_iter_ts)
        print(f'Equals: {next_iter_tr == next_iter_ts}')

    # 3.1.3 MODEL

    # 3.1.4 TRAIN PERFORMANCE

    # 3.1.5 VALIDATION PERFORMANCE

        # ----


