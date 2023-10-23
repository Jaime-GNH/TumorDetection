import os
import random

from TumorDetection.Data.Loader import ImageLoader, DataPathLoader
from TumorDetection.Utils.DictClasses import DataPathDir
from TumorDetection.Utils.DictClasses import BaseClassMap, MappedClassValues
from TumorDetection.Utils.Viewer import Viewer
from TumorDetection.Data.Preprocess import Preprocessor
from TumorDetection.Models.GNN.ImageToGraph import ImageToGraph

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
idx = random.choice(range(len(result)))
for k in range(result[idx][2].shape[-1]):
    Viewer.show_masked_image(result[idx][2][:, :, k], result[idx][3],
                             win_title=f'Image: {os.path.splitext(os.path.basename(result[idx][0]))[0]}.'
                                       f' Class: {result[idx][1][0]}.'
                                       f' Preprocesado: {k}')

# 3. TEST MODELS
# 3.1 GRAPH NEURAL NETWORK -> ON-GOING
# 3.1.1 IMAGE TO HOMOGENENOUS GRAPH CONVERSION
Ig = ImageToGraph()
graphs = Ig(result)


# 3.1.2 DATALOADERS

# 3.1.3 TRAIN PERFORMANCE

# 3.1.4 VALIDATION PERFORMANCE
