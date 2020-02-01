"""Runs the python server and Bullet controller, sending poses to clients in
real-time."""

import asyncio
import argparse
import functools
import math
import numpy as np
import pybullet as p
import random
import time
import websockets

from bullet2unity.bullet import Bullet

global args


async def send_to_client(websocket, path):
    """Starts bullet and sends poses to client"""
    global args

    # Initialize bullet
    bullet = Bullet(args)

    # Run simulation steps.
    for _ in range(args.n_steps):
        poses = bullet.step(curtime=time.time())

        # Send the message to client.
        msg_to_client = str(poses)
        await websocket.send(msg_to_client)

    bullet.disconnect()


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
        "--n_steps",
        type=int,
        default=10000,
        help="The number of simulation steps to run in bullet.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second to run simulation and message sending at.",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="grasp",
        choices=["rest", "grasp"],
        help="The agent's stationary pose.",
    )
    parser.add_argument(
        "--move_agent", action="store_true", help="Whether to move the agent."
    )
    parser.add_argument(
        "--motion_type",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "circular"],
        help="If `move_agent` is true, the type of motion the agent applied to the joints in const.VARY_JOINTS.",
    )
    parser.add_argument(
        "--sin_amplitude",
        type=int,
        default=15,
        help="If the motion type is sinusoidal, this is the amplitude of the sin motion.",
    )
    parser.add_argument(
        "--sin_vshift",
        type=int,
        default=0,
        help="If the motion type is sinusoidal, this is the vertical shift of the sin motion.",
    )
    parser.add_argument(
        "--rotate_objects",
        action="store_true",
        help="Whether to rotate the objects.",
    )
    args = parser.parse_args()

    # Start the python server.
    start_server = websockets.serve(send_to_client, args.hostname, args.port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
