"""A script that sends states to Unity and receives Unity images in return."""
import argparse
import copy
import os
import pprint
import sys
import time
from typing import *

import bullet2unity.interface as interface
from demo.dataset_loader import DatasetLoader
from demo.unity_saver import UnitySaver
import my_pybullet_envs.utils as utils

global args


def main():
    # Start the python server.
    interface.run_server(
        hostname=args.hostname, port=args.port, handler=send_to_client
    )


async def send_to_client(websocket, path):
    global args

    loader = DatasetLoader(
        states_dir=args.states_dir, start_id=args.start_id, end_id=args.end_id
    )
    saver = UnitySaver(
        out_dir=args.out_dir,
        save_keys=["camera_position", "camera_orientation"],
    )
    # cam_target_position = [0.075, 0.2, 0.155]
    cam_target_position = [-0.06, 0.3, 0.0]

    start = time.time()
    n_iter = 0
    while 1:
        msg_id, bullet_state = loader.get_next_state()

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

        # No more states, so we are done.
        if bullet_state is None:
            break

        # x = input("Enter new x: ")
        # y = input("Enter new y: ")
        # z = input("Enter new z: ")
        # if len(x) == 0:
        #     x = cam_target_position[0]
        # if len(y) == 0:
        #     y = cam_target_position[1]
        # if len(z) == 0:
        #     z = cam_target_position[2]
        # cam_target_position = [x, y, z]
        # cam_target_position = [float(val) for val in cam_target_position]
        # print(f"Entered cam_target_position: {cam_target_position}")

        bullet_camera_targets = create_bullet_camera_targets(
            camera_control=args.camera_control,
            position=cam_target_position,
            bullet_state=bullet_state,
        )

        # Encode, send, receive, and decode.
        message = interface.encode(
            state_id=msg_id,
            bullet_state=bullet_state,
            bullet_camera_targets=bullet_camera_targets,
        )
        await websocket.send(message)
        reply = await websocket.recv()
        data = interface.decode(
            msg_id, reply, bullet_camera_targets=bullet_camera_targets
        )
        saver.save(msg_id, data)

        n_iter += 1
        avg_iter_time = (time.time() - start) / n_iter
        print(f"Average iteration time: {avg_iter_time:.2f}")
    sys.exit(0)


def create_bullet_camera_targets(
    camera_control: str, position: List[float], bullet_state: Dict
):
    """ Creates bullet camera targets.

    Args:
        camera_control: The method of camera control.
        bullet_state: The bullet state.
    
    Returns:
        bullet_camera_targets: A dictionary of camera targets in the bullet
            world coordinate frame, with the following format:
            {
                <target_id: int>: {
                    "position": <List[float]>,
                    "should_save": <bool>,
                    "should_send": <bool>,
                }
            }
    """
    # Tell unity to look at every single object.
    if camera_control == "all":
        bullet_camera_targets = {}
        for oid, odict in bullet_state["objects"].items():
            bullet_camera_targets[oid] = {
                "position": odict["position"],
                "should_save": True,
                "should_send": False,
            }
    # Tell unity to look only once at the center of the object distribution.
    elif camera_control == "center":
        # position = utils.compute_object_distribution_mean()
        bullet_camera_targets = {
            0: {
                "position": position,
                "should_save": True,
                "should_send": False,
            }
        }
    elif camera_control == "stack":
        dst_odict = copy.deepcopy(bullet_state["objects"][0])
        position = dst_odict["position"]
        position[2] += dst_odict["height"] / 2
        bullet_camera_targets = {
            0: {
                "position": position,
                "should_save": True,
                "should_send": False,
            }
        }
    else:
        raise ValueError(f"Invalid camera control method: {camera_control}")
    return bullet_camera_targets


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        type=str,
        default="127.0.0.1",
        help="The hostname of the server.",
    )
    parser.add_argument(
        "--port", type=int, default=9000, help="The port of the server."
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
        "--end_id",
        required=True,
        type=int,
        help="The state ID to end generation at.",
    )
    parser.add_argument(
        "--camera_control",
        required=True,
        type=str,
        choices=["all", "center", "stack"],
        help="The method of camera control.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="The output directory to save client data to.",
    )
    args = parser.parse_args()

    main()
