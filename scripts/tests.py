import os
import random

from TumorDetection.Data.Loader import ImageLoader, DataPathLoader
from TumorDetection.Utils.DictClasses import DataPathDir
from TumorDetection.Utils.DictClasses import BaseClassMap, MappedClassValues
from TumorDetection.Utils.Viewer import Viewer
from TumorDetection.Data.Preprocess import Preprocessor

# 1. TEST DATA
# 1.1 TEST DATAPATH -> DONE
Dp = DataPathLoader(dir_path=DataPathDir.get('dir_path'))

# 1.2 TEST IMAGE LOADER -> DONE
paths_classes = Dp(map_classes=BaseClassMap.to_dict())
result = ImageLoader()(paths_classes, class_values=MappedClassValues.to_dict())

# 1.3 TEST PREPROCESSOR -> DONE
prep_results = Preprocessor()([r[2] for r in result])

# 2. TEST UTILITIES
# 2.1 TEST VIEWER -> DONE
idx = random.choice(range(len(result)))
print(result[idx][2].shape)
for k in prep_results:
    Viewer.show_masked_image(prep_results[k][idx], result[idx][3],
                             win_title=f'Image: {os.path.splitext(os.path.basename(result[idx][0]))[0]}.'
                                       f' Class: {result[idx][1][0]}.'
                                       f' Preprocesado: {k}')

# 3. TEST MODELS
# 3.1 GRAPH NEURAL NETWORK -> ON-GOING
# 3.1.1 IMAGE TO HOMOGENENOUS GRAPH CONVERSION

# 3.1.2 DATALOADERS

# 3.1.3 TRAIN PERFORMANCE

# 3.1.4 VALIDATION PERFORMANCE
