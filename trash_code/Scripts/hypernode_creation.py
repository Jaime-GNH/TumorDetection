import cv2

from TumorDetection.data.loader import ImageLoader, DataPathLoader
from trash_code.Data.Preprocess import Preprocessor
from trash_code.Models.GNN import ImageToGraph

from TumorDetection.utils.dict_classes import DataPathDir, BaseClassMap, MappedClassValues, PreprocessorCall

Dp = DataPathLoader(dir_path=DataPathDir.get('dir_path'))
# 1.2 TEST IMAGE LOADER -> DONE
paths_classes = Dp(map_classes=BaseClassMap.to_dict())


result = ImageLoader()(paths_classes, class_values=MappedClassValues.to_dict())

prep_result = Preprocessor()([r[2] for r in result])

# Preprocess + Initial current stage.
result = [(r[0], r[1], pr,
           Preprocessor().resize(
               r[3],
               PreprocessorCall.get('resize_dim'),
               cv2.INTER_NEAREST_EXACT) if PreprocessorCall.get('resize') else r[3],
           None)
          for r, pr in zip(result, prep_result)]

Ig = ImageToGraph()
# it = perf_counter()
graph, mask = Ig(result[0])
# print(f'Elapsed time: {round(perf_counter()-it, 5)} s')
print(graph)