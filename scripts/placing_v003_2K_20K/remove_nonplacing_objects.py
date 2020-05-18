import os
import shutil
from tqdm import tqdm

src_dir = "/home/mguo/data/placing_v003_2K_20K/data"
dst_dir = "/home/mguo/data/placing_v003_2K_20K/filtered_data"

# for fname in tqdm(sorted(os.listdir(src_dir))):
#     if fname.endswith("00.p"):
#         src_path = os.path.join(src_dir, fname)
#         dst_path = os.path.join(dst_dir, fname)
#         shutil.copyfile(src_path, dst_path)

scenes = set()
for fname in sorted(os.listdir(dst_dir)):
    scene_id = int(fname.split("_")[0])
    scenes.add(scene_id)
all_scenes = set(range(20000))
diff = all_scenes - scenes
print(diff)
