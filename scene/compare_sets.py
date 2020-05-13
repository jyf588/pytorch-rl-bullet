import os
import ns_vqa_dart.bullet.util as util

data_dir = "/home/mguo/data/dash/t1_small_range_no_zrot_no_sphere"
set1 = "stack_v1"
set2 = "stack"

dir1 = os.path.join(data_dir, set1, "scenes")
dir2 = os.path.join(data_dir, set2, "scenes")
for fname1 in os.listdir(dir1):
    path1 = os.path.join(dir1, fname1)
    path2 = os.path.join(dir2, fname1)

    x1 = util.load_pickle(path=path1)
    x2 = util.load_pickle(path=path2)

    for idx in range(len(x1)):
        for attr in ["shape", "color", "radius", "height", "position"]:
            assert x1[idx][attr] == x2[idx][attr]
