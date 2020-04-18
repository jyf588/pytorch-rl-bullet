"""A script that sends states to Unity and receives Unity images in return."""
import argparse
import os
import sys
import time

import bullet2unity.interface as interface
from demo.dataset_loader import DatasetLoader
from demo.unity_saver import UnitySaver

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

    start = time.time()
    n_iter = 0
    while 1:
        msg_id, bullet_state = loader.get_next_state()

        # No more states, so we are done.
        if bullet_state is None:
            break

        bullet_camera_targets = create_bullet_camera_targets(bullet_state)

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


def create_bullet_camera_targets(bullet_state):
    # Tell unity to look at every single object.
    bullet_camera_targets = {}
    for oid, odict in bullet_state["objects"].items():
        bullet_camera_targets[oid] = {
            "position": odict["position"],
            "should_save": True,
            "should_send": False,
        }
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
        "--out_dir",
        type=str,
        default="/Users/michelleguo/data/temp_unity_data",
        help="The output directory to save client data to.",
    )
    args = parser.parse_args()

    main()
