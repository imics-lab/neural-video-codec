import sys
from pathlib import Path
sys.path.insert(0, '.')
from training.train_restoration import RestorationDataset, _load_config
print("imports ok")
cfg = _load_config('configs/gpu/restoration.yaml')
print("config ok")
T_win = cfg.get('temporal_window', 3)
ds = RestorationDataset(Path('data/restoration_pairs'), T=T_win, patch_size=64, augment=False)
print(f"dataset len: {len(ds)}")
sample = ds[0]
print(f"sample shapes: {[x.shape for x in sample]}")
print("dataset ok")
