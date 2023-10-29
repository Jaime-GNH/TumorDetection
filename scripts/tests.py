import os
import cv2
import random
from time import perf_counter

from TumorDetection.Data.Loader import ImageLoader, DataPathLoader
from TumorDetection.Utils.DictClasses import DataPathDir, ReportingPathDir
from TumorDetection.Utils.DictClasses import BaseClassMap, MappedClassValues
from TumorDetection.Utils.Viewer import Viewer
from TumorDetection.Data.Preprocess import Preprocessor
from TumorDetection.Models.GNN.ImageToGraph import ImageToGraph
from TumorDetection.Utils.Plotter import plot_graph

VIEW_IMAGES = False
VIEW_GRAPHS = False

# 1. TEST DATA
# 1.1 TEST DATAPATH -> DONE
Dp = DataPathLoader(dir_path=DataPathDir.get('dir_path'))

# 1.2 TEST IMAGE LOADER -> DONE
paths_classes = Dp(map_classes=BaseClassMap.to_dict())
result = ImageLoader()(paths_classes, class_values=MappedClassValues.to_dict())

# 1.3 TEST PREPROCESSOR -> DONE
prep_result = Preprocessor()([r[2] for r in result])
result = [(r[0], r[1], pr, r[3]) for r, pr in zip(result, prep_result)]

# 2. TEST UTILITIES
# 2.1 TEST VIEWER -> DONE
if VIEW_IMAGES:
    idx = random.choice(range(len(result)))
    print(f'Shape (height, width): {result[idx][2][:,:,0].shape}')
    for k in range(result[idx][2].shape[-1]):
        Viewer.show_masked_image(result[idx][2][:, :, k], result[idx][3],
                                 win_title=f'Image: {os.path.splitext(os.path.basename(result[idx][0]))[0]}.'
                                           f' Class: {result[idx][1][0]}.'
                                           f' Preprocesado: {k}')

# 3. TEST MODELS
# 3.1 GRAPH NEURAL NETWORK -> ON-GOING
# 3.1.1 IMAGE TO HOMOGENENOUS GRAPH CONVERSION
Ig = ImageToGraph()

if VIEW_GRAPHS:
    idx = 0
    image2graph = [(result[idx][0],
                    result[idx][1],
                    cv2.resize(result[idx][2], (64, 64), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(result[idx][3], (64, 64), interpolation=cv2.INTER_NEAREST_EXACT))]

    print(f'Shape (height, width): {image2graph[0][2][:,:,0].shape}')
else:
    image2graph = result

it = perf_counter()
graphs = Ig(image2graph, dilations=(1, 5, 11))
print(f'Elapsed time: {round(perf_counter()-it, 5)} s')


if VIEW_GRAPHS:
    it = perf_counter()
    pos = graphs[0].pos
    plot_graph(graphs[0],
               path=os.path.join(ReportingPathDir.get('dir_path'), 'graph_prueba'),
               format='html',
               show=False,
               edges_show=[0, pos.max()//2, pos.max(),
                           len(pos)//2, len(pos)//2 + (pos.max()//2),
                           len(pos)//2 + pos.max(),
                           len(pos)-pos.max()-1, len(pos) - (pos.max()//2) - 2, len(pos)-1])
    print(f'Elapsed time: {round(perf_counter()-it, 5)} s')

# 3.1.2 DATALOADERS

# 3.1.3 MODEL

# 3.1.4 TRAIN PERFORMANCE

# 3.1.5 VALIDATION PERFORMANCE

# ----
