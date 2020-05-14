import os
import sys
import pprint
import shutil
import argparse
import collections

import exp.loader
import system.env
from system.options import VISION_OPTIONS
from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet.metrics import Metrics
from ns_vqa_dart.scene_parse.detectron2.dash import DASHSegModule, register_dataset


def main():
    plan_metrics = Metrics()
    place_metrics = Metrics()
    stack_metrics = Metrics()
    pickle_dir = "/home/mguo/outputs/system/t1/0404/2020_05_13_22_05_00/pickle"

    stage2count = collections.defaultdict(int)
    for fname in os.listdir(pickle_dir):
        path = os.path.join(pickle_dir, fname)
        data = util.load_pickle(path=path)

        scene_id = data["scene_id"]
        stage = data["stage"]
        task = data["task"]

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
            if stage == "plan":
                plan_metrics.add_example(gt_dict=gt_odict, pred_dict=pred_odict)
            elif stage == "place":
                if task == "place" and pred_idx == src_idx:
                    place_metrics.add_example(gt_dict=gt_odict, pred_dict=pred_odict)
                elif task == "stack" and pred_idx in [src_idx, dst_idx]:
                    stack_metrics.add_example(gt_dict=gt_odict, pred_dict=pred_odict)
    plan_metrics.print()
    place_metrics.print()
    # stack_metqrics.print()
    print(stage2count)

    # metrics_path = os.path.join(output_exp_dir, "metrics.txt")
    # assert not os.path.exists(metrics_path)
    # print(metrics_path)
    # sys.stdout = open(metrics_path, "wt")
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
