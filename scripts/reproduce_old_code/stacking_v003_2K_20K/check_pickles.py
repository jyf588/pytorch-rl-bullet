import os
import pickle
from tqdm import tqdm

from ns_vqa_dart.bullet import util


data_dir = "/home/mguo/data/stacking_v003_2K_20K/data"
for fname in tqdm(sorted(os.listdir(data_dir))):
    path = os.path.join(data_dir, fname)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except:
        print(f"Warning: EOF error when reading pickle file {path}.")
