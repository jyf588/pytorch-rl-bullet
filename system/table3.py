import os
import sys
import pprint
import shutil
import argparse
import collections
from tqdm import tqdm

import system.env
from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet.metrics import Metrics
from ns_vqa_dart.scene_parse.detectron2.dash import DASHSegModule, register_dataset


def main(args):
    run_dir = f"/home/mguo/outputs/system/t1/0404/{args.run_name}"
    pickle_dir = os.path.join(run_dir, "pickle")
    metrics_dir = os.path.join(run_dir, "metrics")
    successes_path = os.path.join(run_dir, "successes.json")

    # Initialize the metrics classes, and define paths.
    util.delete_and_create_dir(metrics_dir)
    denom2metrics = {}
    denom2path = {}
    for denom in ["all", "success", "or_success"]:
        name2metrics = {}
        name2path = {}
        for name in ["plan", "place", "stack"]:
            name2metrics[name] = Metrics()
            path = os.path.join(metrics_dir, f"{denom}_{name}.txt")
            assert not os.path.exists(path), path
            name2path[name] = path
        denom2metrics[denom] = name2metrics
        denom2path[denom] = name2path

    # Load successes.
    success_dict = util.load_json(path=successes_path)["scenes"]

    # Gather the pickle files containing gt and preds.
    task2sid2paths = util.AutoVivification()
    for fname in os.listdir(pickle_dir):
        path = os.path.join(pickle_dir, fname)
        task, sid, base_str = fname.split("_")
        ts_str, _ = base_str.split(".")
        ts = int(ts_str)
        if sid not in task2sid2paths[task]:
            task2sid2paths[task][sid] = []
        task2sid2paths[task][sid].append(path)

    # Loop over the scenes that have recorded successes.
    for task in success_dict.keys():
        print(f"Computing task {task}...")
        for sid in tqdm(success_dict[task].keys()):
            # Determine whether to include the example in each of the denominator types.
            # For `all`, we include all examples. For other denominator types, we only
            # include the example if the example is enabled for the tag.
            denoms = ["all"]
            for denom in ["or_success", "success"]:
                if success_dict[task][sid][denom]:
                    denoms += [denom]

            # Loop over all the frames for the task-scene trial.
            for frame_path in task2sid2paths[task][sid]:
                data = util.load_pickle(path=frame_path)

                scene_id = data["scene_id"]
                stage = data["stage"]
                task = data["task"]

                gt_odicts = list(data["gt"]["oid2odict"].values())
                pred_odicts = data["pred"]["odicts"]
                src_idx = data["src_idx"]
                dst_idx = data["dst_idx"]
                gt2pred_map = system.env.match_objects(
                    src_odicts=gt_odicts, dst_odicts=pred_odicts
                )
                for gt_idx, gt_odict in enumerate(gt_odicts):
                    pred_idx = gt2pred_map[gt_idx]
                    pred_odict = pred_odicts[pred_idx]

                    name = None
                    include_example = False
                    if stage == "plan":
                        name = stage
                        include_example = True
                    elif stage == "place":
                        name = task
                        if task == "place" and pred_idx == src_idx:
                            include_example = True
                        elif task == "stack" and pred_idx in [src_idx, dst_idx]:
                            include_example = True

                    if include_example:
                        for denom in denoms:
                            denom2metrics[denom][name].add_example(
                                gt_dict=gt_odict, pred_dict=pred_odict
                            )

    for denom, name2metrics in denom2metrics.items():
        for name, metrics in name2metrics.items():
            if metrics.n_total > 0:
                # Print to console.
                sys.stdout = sys.__stdout__
                metrics.print()

                # Print to file.
                sys.stdout = open(denom2path[denom][name], "wt")
                metrics.print()

    # place_metrics.plot(save_dir=run_dir)

    # print(metrics_path)
    # attr_metrics.print()

    # register_dataset(exp_name=system_exp_name)
    # seg_module = DASHSegModule(
    #     mode="eval",
    #     exp_name=system_exp_name,
    #     checkpoint_path=VISION_OPTIONS.seg_checkpoint_path,
    # )
    # res = seg_module.eval(compute_metrics=True)
    # print(f"Segmentation Performance:")
    # pprint.pprint(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="The name of the run dir.")
    args = parser.parse_args()
    main(args)
