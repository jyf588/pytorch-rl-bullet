"""A script that sends states to Unity and receives Unity images in return."""
import argparse
import copy
import os
import pprint
import sys
import time
from typing import *

import bullet2unity.interface as interface
import bullet2unity.states
from system.dataset_loader import DatasetLoader
from system.unity_saver import UnitySaver
import my_pybullet_envs.utils as utils

global args


def main():
    # Start the python server.
    interface.run_server(hostname=args.hostname, port=args.port, handler=send_to_client)


async def send_to_client(websocket, path):
    global args

    # Ensure that the unity dir exists but the captures dir does not.
    captures_dir = os.path.join(args.unity_dir, "Captures")
    assert os.path.exists(args.unity_dir)
    assert not os.path.exists(captures_dir)

    loader = DatasetLoader(
        states_dir=args.states_dir, start_id=args.start_id, end_id=args.end_id
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
        msg_id, bullet_state = loader.get_next_state()

        # No more states, so we are done.
        if bullet_state is None:
            break

        # We update every frame if trial info is not provided.
        if args.update_cam_target_every_frame:
            update_cam_target = True
        else:
            # This is the first frame of the trial if the trial of the current
            # state is different from the last trial.
            trial = bullet_state["trial"]
            update_cam_target = trial != last_trial

        if update_cam_target:
            first_object_cam_target = bullet2unity.states.get_object_camera_target(
                bullet_odicts=list(bullet_state["objects"].values()), oidx=0
            )

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

        bullet_camera_targets = bullet2unity.states.create_bullet_camera_targets(
            camera_control=args.camera_control,
            # bullet_odicts=None,
            # use_oids=False,
            should_save=True,
            should_send=False,
            position=first_object_cam_target,
        )

        # print("bullet state objects:")
        # pprint.pprint(bullet_state["objects"])
        # input("x")

        # Encode, send, receive, and decode.
        message = interface.encode(
            state_id=msg_id,
            bullet_state=bullet_state,
            bullet_animation_target=None,
            bullet_cam_targets=bullet_camera_targets,
        )
        await websocket.send(message)
        reply = await websocket.recv()
        # input("enter")
        data = interface.decode(msg_id, reply, bullet_cam_targets=bullet_camera_targets)
        saver.save(msg_id, data)

        n_iter += 1
        avg_iter_time = (time.time() - start) / n_iter
        print(f"Average iteration time: {avg_iter_time:.2f}")

        # Store the current trial as the "last trial".
        if args.update_cam_target_every_frame:
            pass
        else:
            last_trial = trial
    print(f"Time elapsed: {time.time() - start}")
    sys.exit(0)


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
        "--start_id",
        required=True,
        type=int,
        help="The state ID to start generation at.",
    )
    parser.add_argument(
        "--end_id", required=True, type=int, help="The state ID to end generation at.",
    )
    parser.add_argument(
        "--camera_control",
        required=True,
        type=str,
        choices=["all", "center", "stack", "position"],
        help="The method of camera control.",
    )
    parser.add_argument(
        "--update_cam_target_every_frame",
        action="store_true",
        help="Whether to recompute / update cam target every frame.",
    )
    parser.add_argument(
        "--cam_dir",
        required=True,
        type=str,
        help="The output directory to save client data to.",
    )
    args = parser.parse_args()
    main()
