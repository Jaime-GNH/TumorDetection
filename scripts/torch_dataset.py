from TumorDetection.Data.Loader import DataPathLoader
from TumorDetection.Data.Dataset import TorchDataset
from TumorDetection.Utils.DictClasses import DataPathDir

dp = DataPathLoader(DataPathDir.get('dir_path'))
paths = dp()
td = TorchDataset(paths)
a, b, c = next(iter(td))
print(b.max(), b.min(), b.sum(), c)