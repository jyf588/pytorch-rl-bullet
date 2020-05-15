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


def main():
    run_dir = "/home/mguo/outputs/system/t1/0404/2020_05_15_12_51_33"
    pickle_dir = os.path.join(run_dir, "pickle")
    metrics_dir = os.path.join(run_dir, "metrics")
    successes_path = os.path.join(run_dir, "successes.json")

    # Initialize the metrics classes, and define paths.
    os.makedirs(metrics_dir)
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

    for fname in tqdm(os.listdir(pickle_dir)):
        path = os.path.join(pickle_dir, fname)
        data = util.load_pickle(path=path)

        scene_id = data["scene_id"]
        stage = data["stage"]
        task = data["task"]

        # Determine whether the task-scene succeeded.
        denoms = ["all"]
        for denom in ["or_success", "success"]:
            if success_dict[task][scene_id][denom]:
                denoms += [denom]

        # Compute attribute metrics.
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("exp_name", type=str, help="The name of the experiment to run.")
    # args = parser.parse_args()
    main()
