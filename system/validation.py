"""The purpose of the script is to run the validation set through the test code
from the system. We can use this script to determine whether the validation 
inference code is doing the same thing as the inference code of the system/test
set.

0.77: 28%
1.46: 57%
0.3: 20%
"""
from ns_vqa_dart.bullet import dash_object, gen_dataset, util
from ns_vqa_dart.bullet.metrics import Metrics
import os
from system.vision_module import VisionModule


CAMERA_CONTROL = "center"


def main():
    model = VisionModule(
        load_checkpoint_path="/home/michelle/mguo/outputs/planning_v003_20K/checkpoint_best.pt"
    )

    states_dir = os.path.join(
        "/home/michelle/mguo/data/states/full/planning_v003_20K"
    )

    root_dir = "/home/michelle/mguo/data/datasets/planning_v003_20K"
    dataset_dir = os.path.join(root_dir, "data")
    img_dir = os.path.join(root_dir, "unity_output/images")
    cam_dir = os.path.join(root_dir, "unity_output/json")

    metrics = Metrics(plot_path="/home/michelle/mguo/outputs/system/val.png")

    sid2fnames = {}
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".p"):
            continue
        sid = int(fname.split("_")[0])
        if sid not in sid2fnames:
            sid2fnames[sid] = []
        sid2fnames[sid].append(fname)

    for sid in range(16000, 16100):
        state = util.load_pickle(path=os.path.join(states_dir, f"{sid:06}.p"))
        oids = list(state["objects"].keys())

        for oid in oids:
            gt_dict = state["objects"][oid]

            rgb, seg_img = gen_dataset.load_rgb_and_seg_img(
                img_dir=img_dir,
                sid=sid,
                oid=oid,
                camera_control=CAMERA_CONTROL,
            )
            pred = model.predict(oid=oid, rgb=rgb, seg_img=seg_img)
            if pred is None:
                continue
            # path = os.path.join(dataset_dir, fname)
            # data = util.load_pickle(path=path)
            # pred = model.predict_from_data(data=X)
            cam_position, cam_orientation = gen_dataset.load_camera_pose(
                cam_dir=cam_dir,
                sid=sid,
                oid=None,
                camera_control=CAMERA_CONTROL,
            )
            pred_dict = dash_object.y_vec_to_dict(
                y=list(pred[0]),
                coordinate_frame="unity_camera",
                cam_position=cam_position,
                cam_orientation=cam_orientation,
            )
            # gt_dict = dash_object.y_vec_to_dict(
            #     y=y,
            #     coordinate_frame="unity_camera",
            #     cam_position=cam_position,
            #     cam_orientation=cam_orientation,
            # )
            gt_dict["up_vector"] = [0.0, 0.0, 1.0]
            metrics.add_example(gt_dict=gt_dict, pred_dict=pred_dict)
    metrics.print()


if __name__ == "__main__":
    main()
