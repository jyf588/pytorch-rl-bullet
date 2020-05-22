"""A script that sends states to Unity and receives Unity images in return."""
import argparse
import copy
import os
import pprint
import sys
import time
import math
import numpy as np
from typing import *
import pybullet as p

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

    paths = [
        os.path.join(args.states_dir, f) for f in sorted(os.listdir(args.states_dir))
    ]

    start = time.time()
    idx = 0
    bullet_camera_targets = {}
    while idx < len(paths):
        path = paths[idx]
        bullet_state = util.load_json(path)
        example_id = f"{idx:06}"

        bullet_state["objects"] = assign_ids_to_objects(bullet_state["objects"])

        # X -> Z
        # Y -> X
        # Z -> Y
        # [Z, X, y]
        # a = (time.time() % (2 * math.pi)) - math.pi
        # bullet_state["objects"][2]["orientation"] = p.getQuaternionFromEuler([0, 0, a])
        # bullet_state["objects"][2]["position"] = [0.0, 0.0, 0.0]
        # Encode, send, receive, and decode.
        message = interface.encode(
            state_id=example_id,
            bullet_state=bullet_state,
            bullet_animation_target=None,
            bullet_cam_targets=bullet_camera_targets,
            head_speed=0,
            save_third_pov_image=False,
        )
        await websocket.send(message)
        reply = await websocket.recv()
        data = interface.decode(
            example_id, reply, bullet_cam_targets=bullet_camera_targets
        )

        idx += 10
        avg_iter_time = (time.time() - start) / idx
        print(f"Idx: {idx}\tAverage iteration time: {avg_iter_time:.2f}")

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
        "--states_dir",
        required=True,
        type=str,
        help="The directory of states to read from and send to client.",
    )
    args = parser.parse_args()
    main()
