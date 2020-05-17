import cv2
import pprint
import imageio
import numpy as np
from PIL import Image

from ns_vqa_dart.bullet import util, dash_object
from system.vision_module import VisionModule
import ns_vqa_dart.bullet.seg


def main():
    # visualize_inputs()

    debug_single_train_scene()


def debug_single_train_scene():
    # Our goal is to compare the images rendered using the training vs. testing
    # rendering pipeline. If the images are different, this means that the input to the
    # model is different and thus is giving us unexpected results. We will make the
    # comparison on a single example scene from the training set. The training image
    # has already been produced. All that remains is to render the same scene using the
    # testing pipeline.

    # First, we load the scene.
    path = "/home/mguo/states/full/placing_v003_2K_20K/000068.p"
    state = util.load_pickle(path)
    oid2odict = state["objects"]
    scene = list(oid2odict.values())
    print(f"Scene:")
    pprint.pprint(scene)
    """
    [{'color': 'green',
    'height': 0.17203193853442042,
    'last_orientation': (-0.04786798194793857,
                        0.02928311924830052,
                        -0.661528149578617,
                        0.7478179340898508),
    'last_position': (0.1508376098274795,
                        0.14199724698008834,
                        0.1580361648609928),
    'mass': 3.984239766811468,
    'mu': 1.1164703573372667,
    'orientation': [-0.07398998200024144,
                    0.0288114596492522,
                    -0.6505331146743686,
                    0.75531586046405],
    'position': [0.1546370704499176, 0.14178100838002647, 0.14640540703486363],
    'radius': 0.02534757332771756,
    'shape': 'box'},
    {'color': 'yellow',
    'height': 0.05713063913416408,
    'mass': 1.1819285826811763,
    'mu': 1.0,
    'orientation': [0.0, 0.0, 0.0, 1.0],
    'position': [-0.09643964030837852, 0.19411718734653902, 0.02856531956708204],
    'radius': 0.023430980219651523,
    'shape': 'cylinder'},
    {'color': 'blue',
    'height': 0.09086558266922806,
    'mass': 2.9844958049994443,
    'mu': 1.0,
    'orientation': [0.0, 0.0, 0.0, 1.0],
    'position': [0.033186927023374, 0.3215138203768405, 0.04543279133461403],
    'radius': 0.04543279133461403,
    'shape': 'sphere'}]
    """

    # We also want the robot arm state for rendering.
    print(f"Robot:")
    pprint.pprint(state["robot"])
    """
    Robot:
    {'r_elbow_flex_joint': -1.3390375025171732,
    'r_elbow_roll_joint': -0.6332177550309832,
    'r_shoulder_lift_joint': -0.34280413217417843,
    'r_shoulder_out_joint': -0.7649060740130982,
    'r_upper_arm_roll_joint': -0.13771305025626246,
    'r_wrist_roll_joint': 0.0,
    'rh_FFJ1': -0.03646373161276685,
    'rh_FFJ2': 0.03006739546566138,
    'rh_FFJ3': 0.28844378675722276,
    'rh_FFJ4': 0.008249924428430169,
    'rh_FFtip': 0.0,
    'rh_LFJ1': 0.1046723838034935,
    'rh_LFJ2': 0.870062334135498,
    'rh_LFJ3': 0.390097552844339,
    'rh_LFJ4': 0.01059673751129744,
    'rh_LFJ5': 0.0,
    'rh_LFtip': 0.0,
    'rh_MFJ1': 0.0055998471602668855,
    'rh_MFJ2': 0.6699840759332788,
    'rh_MFJ3': 0.29380181248089576,
    'rh_MFJ4': 0.002040712097830817,
    'rh_MFtip': 0.0,
    'rh_RFJ1': -0.0006214548816121941,
    'rh_RFJ2': 1.034483270259803,
    'rh_RFJ3': 0.3969398734378197,
    'rh_RFJ4': 0.00020752723033524588,
    'rh_RFtip': 0.0,
    'rh_THJ1': 0.002013363480970862,
    'rh_THJ2': -0.1946846632162728,
    'rh_THJ3': 0.07347760250744376,
    'rh_THJ4': 1.2215781011096742,
    'rh_THJ5': 0.6067656469340215,
    'rh_WRJ1': 0.14692470391586823,
    'rh_WRJ2': -0.13188266656856792,
    'rh_thtip': 0.0}
    """

    # Copy the printed scene and robot state and run it through the testing pipeline.
    # The resulting image returned by unity is what we will compare.
    # Save the image to ~/test_image.png.

    # Now, load the train and test images.
    old_train_path = "/home/mguo/debug_single_frame/train_image_final.png"
    new_train_v1_path = "/home/mguo/debug_single_frame/new_train_v1.png"
    new_train_v2_path = "/home/mguo/debug_single_frame/new_train_v2.png"
    test_incorrect_path = "/home/mguo/debug_single_frame/test_image_incorrect.png"
    test_correct_path = "/home/mguo/debug_single_frame/test_image_correct.png"

    compare_images(new_train_v1_path, test_correct_path)
    compare_images(new_train_v1_path, new_train_v2_path)


def compare_images(path1, path2):
    print(f"Comparing two images:")
    print(path1)
    print(path2)
    print()
    img1_uint8 = imageio.imread(path1)
    img2_uint8 = imageio.imread(path2)

    img1 = img1_uint8.copy().astype(np.float32)
    img2 = img2_uint8.copy().astype(np.float32)

    assert img1.shape == img2.shape

    H, W, C = img1.shape
    n_equal, n_diff = 0, 0
    diff_total = np.zeros((3,))
    diff_mask = np.full((H, W), False, dtype=np.bool)
    for i in range(H):
        for j in range(W):
            train_pixel = list(img1[i, j])
            test_pixel = list(img2[i, j])
            if train_pixel != test_pixel:
                n_diff += 1
                diff_total = diff_total + np.abs(
                    np.array(train_pixel) - np.array(test_pixel)
                )
                diff_mask[i, j] = True
            else:
                n_equal += 1
    diff = img1 - img2
    print(diff.sum())
    diff_img = diff.astype(np.uint8)
    diff_img = cv2.cvtColor(diff_img, cv2.COLOR_RGB2GRAY)

    masked_img = img1.copy()
    masked_img[diff_mask] = [0, 0, 0]
    imageio.imwrite("/home/mguo/masked.png", masked_img)
    imageio.imwrite("/home/mguo/diff.png", diff_img)

    diff_avg = diff_total / n_diff

    print(f"Same pixels: {n_equal}")
    print(f"Different pixels: {n_diff}")
    print(f"Diff total: {diff_total}")
    print(f"Diff avg: {diff_avg}")

    # Next, blend them together.
    blend_images(path1=path1, path2=path2)

    # We also verify that the camera poses match.
    # camera_path = "/home/mguo/data/placing_v003_2K_20K/unity_output/json/000068.json"
    # train_cam_dict = util.load_json(camera_path)["0"]
    # test_cam_dict = {
    #     "camera_position": [-4.93443, 1.555091, -2.752659],
    #     "camera_orientation": [0.04981242, -0.9406025, 0.1561613, 0.2973206],
    # }
    # Train:
    # {'camera_orientation': [0.04981242, -0.9406025, 0.1561613, 0.2973206], 'camera_position': [-4.93443, 1.555091, -2.752659]}
    # Test (incorrect):
    # [-4.938934, 1.535119, -2.759074], [0.06319467, -0.9325694, 0.1985249, 0.2948031]
    # Test (correct):
    # [-4.93443, 1.555091, -2.752659], [0.04981242, -0.9406025, 0.1561613, 0.2973206]

    # Next, we feed the images through the vision module.
    # First, we need to load the mask image.
    # seg_img = imageio.imread("/home/mguo/debug_single_frame/train_masks.png")
    # masks, oids = ns_vqa_dart.bullet.seg.seg_img_to_map(seg_img)
    # model = VisionModule(
    #     load_checkpoint_path="/home/mguo/outputs/placing_v003_2K_20K/2020_04_22_04_35/checkpoint_best.pt",
    # )
    # train_preds, train_input = model.predict(rgb=train_img_original, masks=[masks[0]])
    # # test_preds, test_input = model.predict(rgb=test_img_original, masks=[masks[0]])

    # train_dict_local = dash_object.y_vec_to_dict(
    #     y=train_preds[0], coordinate_frame="world"
    # )
    # test_dict_local = dash_object.y_vec_to_dict(
    #     y=test_preds[0], coordinate_frame="world"
    # )
    # print(train_dict_local)
    # print(test_dict_local)

    # train_dict_world = dash_object.y_vec_to_dict(
    #     y=train_preds[0],
    #     coordinate_frame="unity_camera",
    #     cam_position=train_cam_dict["camera_position"],
    #     cam_orientation=train_cam_dict["camera_orientation"],
    # )
    # test_dict_world = dash_object.y_vec_to_dict(
    #     y=test_preds[0],
    #     coordinate_frame="unity_camera",
    #     cam_position=test_cam_dict["camera_position"],
    #     cam_orientation=test_cam_dict["camera_orientation"],
    # )
    # print(train_dict_world)
    # print(test_dict_world)


def blend_images(path1: str, path2: str):
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")

    new_img = Image.blend(img1, img2, 0.5)

    blend_path = "/home/mguo/blend.png"
    new_img.save(blend_path, "PNG")
    print(f"Saved blend image to: {blend_path}")


def visualize_inputs():
    test_path = "/home/mguo/outputs/system/t1/0404/2020_05_14_14_54_45/pickle/place_0000_000646.p"

    test_data = util.load_pickle(test_path)
    test_input_img = test_data["vision_inputs"]  # RGB

    train_path = "/home/mguo/data/placing_v003_2K_20K/data/000000_00.p"
    X_train = util.load_pickle(train_path)[0]  # RGB
    train_input_img = np.hstack([X_train[:, :, :3], X_train[:, :, 3:6]])

    # RGB to BGR
    test_input_img = test_input_img[:, :, ::-1]
    train_input_img = train_input_img[:, :, ::-1]

    # cv2.imshow("test input", test_input_img)
    # cv2.imshow("train input", train_input_img)
    # cv2.waitKey(0)

    cv2.imwrite("/home/mguo/test_input.png", test_input_img)
    cv2.imwrite("/home/mguo/train_input.png", train_input_img)


if __name__ == "__main__":
    main()
