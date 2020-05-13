import os
import sys
import pprint
import argparse

import exp.loader
import system.env
from system.options import VISION_OPTIONS
from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet.metrics import Metrics
from ns_vqa_dart.scene_parse.detectron2.dash import DASHSegModule, register_dataset


def main(args: argparse.Namespace):
    attr_metrics = Metrics()
    system_exp_name = f"system_{args.exp_name}"
    output_exp_dir = os.path.join(
        "/home/mguo/outputs/system", args.exp_name, "2020_05_12_18_41_54"
    )

    # Loop over the predictions...
    for set_name in exp.loader.ExpLoader(exp_name=args.exp_name).set_names:
        system_set_loader = exp.loader.SetLoader(
            exp_name=system_exp_name, set_name=set_name
        )
        system_set_loader.scenes_dir
        scene_ids = exp.loader.SetLoader(
            exp_name=args.exp_name, set_name=set_name
        ).get_scene_ids()
        if not os.path.exists(system_set_loader.scenes_dir):
            system_set_loader.save_scenes(scenes=[[] * len(scene_ids)])
        for scene_id in scene_ids:
            output_scene_dir = os.path.join(output_exp_dir, set_name, scene_id)
            for fname in sorted(os.listdir(output_scene_dir)):
                path = os.path.join(output_scene_dir, fname)
                data = util.load_pickle(path=path)

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
                    if pred_idx in [src_idx, dst_idx]:
                        pred_odict = pred_odicts[pred_idx]
                        attr_metrics.add_example(gt_dict=gt_odict, pred_dict=pred_odict)

    metrics_path = os.path.join(output_exp_dir, "metrics.txt")
    assert not os.path.exists(metrics_path)
    print(metrics_path)
    sys.stdout = open(metrics_path, "wt")
    attr_metrics.print()

    register_dataset(exp_name=system_exp_name)
    seg_module = DASHSegModule(
        mode="eval",
        exp_name=system_exp_name,
        checkpoint_path=VISION_OPTIONS.seg_checkpoint_path,
    )
    res = seg_module.eval(compute_metrics=True)
    print(f"Segmentation Performance:")
    pprint.pprint(res)
    # util.save_json(path=os.path.join(output_exp_dir, "seg_metrics.json"), data=res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="The name of the experiment to run.")
    args = parser.parse_args()
    main(args)
