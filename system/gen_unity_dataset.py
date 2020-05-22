"""A script that sends states to Unity and receives Unity images in return."""
import argparse
import copy
import os
import pprint
import sys
import time
import numpy as np
from typing import *

import bullet2unity.interface as interface
import bullet2unity.states
from system.dataset_loader import DatasetLoader
from system.unity_saver import UnitySaver
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet import util

global args


def main():
    # Start the python server.
    interface.run_server(hostname=args.hostname, port=args.port, handler=send_to_client)


async def send_to_client(websocket, path):
    global args

    np.random.seed(args.seed)

    # Ensure that the unity dir exists but the captures dir does not.
    captures_dir = os.path.join(args.unity_dir, "Captures")
    assert os.path.exists(args.unity_dir)
    util.delete_and_create_dir(captures_dir)

    # Initialize a dataset loader.
    loader = DatasetLoader(
        stage=args.stage,
        states_dir=args.states_dir,
        start_trial_incl=args.start_trial_incl,
        end_trial_incl=args.end_trial_incl,
    )
    saver = UnitySaver(
        cam_dir=args.cam_dir, save_keys=["camera_position", "camera_orientation"],
    )
    # cam_target_position = [0.075, 0.2, 0.155]
    # cam_target_position = [-0.06, 0.3, 0.0]

    start = time.time()
    n_iter = 0
    last_trial = None
    while 1:
        example_id, bullet_state = loader.get_next_state()

        # No more states, so we are done.
        if bullet_state is None:
            break

        if args.cam_version == "v1":
            # We update every frame if trial info is not provided.
            if args.missing_trial_info:
                update_cam_target = True
            else:
                # This is the first frame of the trial if the trial of the current
                # state is different from the last trial.
                trial = bullet_state["trial"]
                update_cam_target = trial != last_trial

            if update_cam_target:
                # GT index of target object is always 0.
                target_oidx = 0
                odicts = bullet_state["objects"]
                # We don't need the image sent but we need unity save the first person
                # images it generates.
                bullet_camera_targets = bullet2unity.states.compute_bullet_camera_targets(
                    version=args.cam_version,
                    send_image=False,
                    save_image=True,
                    stage=args.stage,
                    odicts=odicts,
                    oidx=target_oidx,
                )
        elif args.cam_version == "v2":
            if args.stage == "place":
                tx_act = bullet_state["tx_act"]
                ty_act = bullet_state["ty_act"]

                tx = utils.perturb_scalar(np.random, tx_act, 0.02)
                ty = utils.perturb_scalar(np.random, ty_act, 0.02)
            else:
                tx, ty = None, None

            # We don't need the image sent but we need unity save the first person
            # images it generates.
            bullet_camera_targets = bullet2unity.states.compute_bullet_camera_targets(
                version=args.cam_version,
                send_image=False,
                save_image=True,
                stage=args.stage,
                tx=tx,
                ty=ty,
            )

        bullet_state["objects"] = assign_ids_to_objects(bullet_state["objects"])

        if args.visualize_cam_debug:
            bullet_state = add_cam_target_visual(
                bullet_state, bullet_camera_targets[0]["position"]
            )

        # Encode, send, receive, and decode.
        message = interface.encode(
            state_id=example_id,
            bullet_state=bullet_state,
            bullet_animation_target=None,
            bullet_cam_targets=bullet_camera_targets,
        )
        await websocket.send(message)
        reply = await websocket.recv()
        data = interface.decode(
            example_id, reply, bullet_cam_targets=bullet_camera_targets
        )
        saver.save(example_id, data)

        n_iter += 1
        avg_iter_time = (time.time() - start) / n_iter
        print(f"Average iteration time: {avg_iter_time:.2f}")

        # Store the current trial as the "last trial".
        if args.cam_version == "v1":
            if args.missing_trial_info:
                pass
            else:
                last_trial = trial
    print(f"Time elapsed: {time.time() - start}")
    sys.exit(0)


def assign_ids_to_objects(odicts):
    id2odict = {idx: odict for idx, odict in enumerate(odicts)}
    return id2odict


# Create a state with extreme object locations and sides.
# extreme_xy = [
#     (utils.TX_MIN, utils.TY_MIN),
#     (utils.TX_MIN + 0.15, utils.TY_MAX - 0.15),
#     (utils.TX_MAX, utils.TY_MIN),
#     (utils.TX_MAX, utils.TY_MAX),
# ]
# base_odict = {
#     "radius": utils.HALF_W_MAX,
#     "height": utils.H_MAX,
#     "orientation": [0.0, 0.0, 0.0, 1.0],
# }
# colors = ["red", "green", "blue", "yellow"]
# shapes = ["cylinder", "box", "cylinder", "box"]
# bullet_state = {"objects": {}}
# for odict_idx, (x, y) in enumerate(extreme_xy):
#     extreme_odict = copy.deepcopy(base_odict)
#     extreme_odict["shape"] = shapes[odict_idx]
#     extreme_odict["color"] = colors[odict_idx]
#     extreme_odict["position"] = [x, y, utils.H_MAX / 2]
#     bullet_state["objects"][odict_idx] = extreme_odict


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname", type=str, default="127.0.0.1", help="The hostname of the server.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port of the server."
    )
    parser.add_argument(
        "--seed", required=True, type=int, help="The random seed.",
    )
    parser.add_argument(
        "--unity_dir",
        required=True,
        type=str,
        help="The directory of the unity executable.",
    )
    parser.add_argument(
        "--states_dir",
        required=True,
        type=str,
        help="The directory of states to read from and send to client.",
    )
    parser.add_argument(
        "--cam_dir",
        required=True,
        type=str,
        help="The output directory to save client data to.",
    )
    parser.add_argument(
        "--stage", required=True, type=str, help="The stage of the dataset.",
    )
    parser.add_argument(
        "--start_trial_incl", type=int, help="The state ID to start generation at.",
    )
    parser.add_argument(
        "--end_trial_incl", type=int, help="The state ID to end generation at.",
    )
    parser.add_argument(
        "--cam_version",
        required=True,
        type=str,
        help="The version of camera target algorithm to use.",
    )
    parser.add_argument(
        "--visualize_cam_debug",
        action="store_true",
        help="Whether to visualize cam target for debugging.",
    )
    parser.add_argument(
        "--missing_trial_info",
        action="store_true",
        help="Whether it's missing trial info in the states, so we recompute / update cam target every frame.",
    )
    args = parser.parse_args()
    main()
