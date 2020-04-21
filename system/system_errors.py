"""Analyzes errors of the vision module for the DASH system.

Expects the following file structure:
<args.input_dir>/
    <sid:02>/
        <timestep:06>.p = {
            "gt": [
                {
                    <attr>: <value>,
                    ...
                },
                ...
            ],
            "pred": [
                {
                    <attr>: <value>,
                    ...
                },
                ...
            ],
        }
"""
import argparse
import os

from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet.metrics import Metrics


def main(args: argparse.Namespace):
    # Initialize the metrics class.
    plot_path = os.path.join(args.input_dir, "plot.png")
    planning_metrics = Metrics(plot_path=plot_path)
    stacking_metrics = Metrics(plot_path=plot_path)

    for scene_id in range(0, 100):
        scene_dir = os.path.join(args.input_dir, f"{scene_id:02}")
        if not os.path.isdir(scene_dir):
            continue

        ts2path = {}
        for fname in sorted(os.listdir(scene_dir)):
            ts = int(fname.split(".")[0])
            ts2path[ts] = os.path.join(scene_dir, fname)

        for ts, path in ts2path.items():
            data = util.load_pickle(path=path)
            gt_odicts = data["gt"]
            pred_odicts = data["pred"]

            for gt_odict, pred_odict in zip(gt_odicts, pred_odicts):
                if ts == 0:
                    planning_metrics.add_example(
                        gt_dict=gt_odict, pred_dict=pred_odict
                    )
                else:
                    stacking_metrics.add_example(
                        gt_dict=gt_odict, pred_dict=pred_odict
                    )

    planning_metrics.print()
    stacking_metrics.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/michelle/mguo/outputs/system",
        help="The input directory to read ground truth and predicted objects from.",
    )
    args = parser.parse_args()

    main(args=args)
