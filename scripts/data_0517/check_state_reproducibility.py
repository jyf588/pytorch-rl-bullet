import os

from ns_vqa_dart.bullet import util


def main():
    check_policy_states()


def compare_pickles(paths1, paths2):
    assert len(paths1) == len(paths2)

    for p1, p2, in zip(paths1, paths2):
        x1 = util.load_pickle(p1)
        x2 = util.load_pickle(p2)
        if x1 != x2:
            debug = 0


def check_policy_states():
    dir1 = "/home/mguo/states/policy/place_100K_0517"
    dir2 = "/home/mguo/states/policy/partial/placing_v003_2K"

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
