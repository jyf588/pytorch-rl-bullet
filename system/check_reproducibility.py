import os
import numpy as np
from ns_vqa_dart.bullet import util

GT_RUN_NAMES = ["2020_05_15_22_47_48", "2020_05_15_22_50_35", "2020_05_15_23_06_50"]
GT_UNITY_RUN_NAMES = ["2020_05_15_23_31_33"]
VISION_SEG_RUN_NAMES = ["2020_05_16_01_41_05", "2020_05_16_01_41_33"]


def main():
    run_names = VISION_SEG_RUN_NAMES

    root_dir = "/home/mguo/outputs/system/t1/0404"

    # Initialize the dictionary we will store states in.
    run2task2id2ts = util.AutoVivification()

    # Populate the dictionary of states.
    examples = []
    for run_name in run_names:
        run_dir = os.path.join(root_dir, run_name)
        states_dir = os.path.join(run_dir, "states")
        for fname in sorted(os.listdir(states_dir)):
            task, scene_id, base = fname.split("_")
            ts_str, _ = base.split(".")
            ts = int(ts_str)
            path = os.path.join(states_dir, fname)
            state = util.load_pickle(path=path)
            if task == "stack" and int(scene_id) == 0 and ts < 300:
                run2task2id2ts[run_name][task][scene_id][ts] = state
                e = (task, scene_id, ts)
                if e not in examples:
                    examples.append((task, scene_id, ts))

    # Now, loop over the timesteps and compare across runs.
    for task, sid, ts in examples:
        states = []
        for run in run_names:
            state = run2task2id2ts[run][task][sid][ts]
            states.append(state)
        src_idx = 0
        src_state = states[src_idx]
        for idx, state in enumerate(states):
            if src_state != state:
                x1, x2 = extract_robot_states(src_state, state)
                if not np.allclose(x1, x2, 1e-07, 1e-07):
                    print(
                        f"Run {run_names[src_idx]} and {run_names[idx]} do not match for task {task}, sid {sid}, ts {ts}."
                    )
                    print(f"Diff: {(x1-x2)/x2}")


def extract_robot_states(s1, s2):
    x1, x2 = [], []
    for j in s1["robot"].keys():
        x1.append(s1["robot"][j])
        x2.append(s2["robot"][j])
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2


if __name__ == "__main__":
    main()
