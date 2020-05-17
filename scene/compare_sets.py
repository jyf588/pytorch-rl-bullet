import os
import ns_vqa_dart.bullet.util as util

dir1 = "/home/mguo/data/dash/t1"
dir2 = "/home/mguo/data/dash/t1_repro_test"


for task in ["place", "stack"]:
    n_diff = 0
    t1 = os.path.join(dir1, task)
    t2 = os.path.join(dir2, task)
    for fname1 in os.listdir(t1):
        name, _ = fname1.split(".")
        path1 = os.path.join(t1, fname1)
        path2 = os.path.join(t2, fname1)

        x1 = util.load_json(path=path1)
        x2 = util.load_json(path=path2)

        if x1 != x2:
            n_diff += 1
    print(n_diff)

# for idx in range(len(x1)):
#     for attr in ["shape", "color", "radius", "height", "position", ""]:
#         assert x1[idx][attr] == x2[idx][attr]
