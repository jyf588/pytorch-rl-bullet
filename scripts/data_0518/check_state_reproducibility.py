import os

import numpy as np
from tqdm import tqdm
from ns_vqa_dart.bullet import util


def main():
    check_policy_states()


def compare_pickles(paths1, paths2):
    mismatch_01 = 0
    mismatch_02 = 0
    mismatch_03 = 0
    n = 0
    for idx in tqdm(range(len(paths1))):
        p1 = paths1[idx]
        p2 = paths2[idx]
        x1 = util.load_pickle(p1)["robot"]
        x2 = util.load_pickle(p2)["robot"]
        arr1 = []
        arr2 = []
        for j in x1.keys():
            a = x1[j]
            b = x2[j]
            arr1.append(a)
            arr2.append(b)
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        if not np.allclose(arr1, arr2, 1e-01, 1e-01):
            mismatch_01 += 1
        if not np.allclose(arr1, arr2, 1e-02, 1e-02):
            mismatch_02 += 1
        if not np.allclose(arr1, arr2, 1e-03, 1e-03):
            mismatch_03 += 1
        n += 1
    print(f"Mismatches: {mismatch_01}")
    print(f"Mismatches: {mismatch_02}")
    print(f"Mismatches: {mismatch_03}")
    print(f"Total: {n}")


def check_policy_states():
    dir1 = "/home/mguo/states/partial/stack_100K_0518"
    dir2 = "/home/mguo/states/policy/partial/stacking_v003_2K"

    paths1 = collect_trial_organized_paths(dir1)
    paths2 = collect_paths(dir2)

    compare_pickles(paths1, paths2)


def check_scene_states():
    dir1 = "/home/mguo/states/scenes/place_2K_20K_0517"
    dir2 = "/home/mguo/states/full/placing_v003_2K_20K"

    paths1 = collect_trial_organized_paths(dir1)
    paths2 = collect_paths(dir2)

    compare_pickles(paths1, paths2)


def collect_trial_organized_paths(states_dir):
    paths = []
    for t in sorted(os.listdir(states_dir)):
        if int(t) > 2000:
            continue
        t_dir = os.path.join(states_dir, t)
        for fname in sorted(os.listdir(t_dir)):
            path = os.path.join(t_dir, fname)
            paths.append(path)
    return paths


def collect_paths(states_dir):
    paths = []
    for fname in sorted(os.listdir(states_dir)):
        path = os.path.join(states_dir, fname)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
