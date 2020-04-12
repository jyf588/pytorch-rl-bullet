"""Contains functions that serve as the interface between PyBullet (server) 
and Unity (client)."""
import asyncio
import base64
import numpy as np
import imageio
from io import BytesIO
from PIL import Image
import time
import websockets
from typing import *

import bullet2unity.states


def run_server(hostname: str, port: int, handler: Callable):
    """Runs the python server.

    Args:
        hostname: The hostname to run the server on.
        port: The port to run the server on.
        handler: The handler coroutine to run for each websocket connection.
    """
    print(f"Running server on {hostname}:{port}")

    start_server = websockets.serve(handler, hostname, port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def encode(
    state_id: str, bullet_state: List[Any], bullet_camera_targets
) -> str:
    """Converts the provided bullet state into a Unity state, and encodes the
    message for sending to Unity.

    Args:
        state_id: The ID of the state.
        bullet_state: The Bullet state dictionary, with the format 
            {
                "objects": {
                    "<oid>": {
                        "shape": shape,
                        "color": color,
                        "radius": radius,
                        "height": height,
                        "orientation": [x, y, z, w],
                        "position": [x, y, z]
                    },
                    ...
                },
                "robot": {
                    "<joint_name>": <joint_angle>,
                    ...
                }
            }. Note that if "robot" key is not provided, the default robot pose 
            will be used.
        bullet_camera_targets: A dictionary of target positions that we want
            Unity to point the camera at, in the format:
            {
                <id>: {
                    "position": <List[float]>  # The xyz position, in bullet world coordinate frame.
                    "save": <bool>,  # Whether to save an image using the camera.
                    "send": <bool>,  # Whether to send the image over the websocket.
                }
            }

    Returns:
        message: The message to send to unity.
    """
    unity_state = bullet2unity.states.bullet2unity_state(
        bullet_state=bullet_state, bullet_camera_targets=bullet_camera_targets
    )

    # Combine the id and state, and stringify into a msg.
    message = [state_id] + unity_state
    message = str(message)
    print(f"Sending to unity: state {state_id}\tTime: {time.time()}")
    return message


def decode(message_id: str, reply: str, bullet_camera_targets):
    """Decodes messages received from Unity.

    Args:
        message_id: The original message ID we sent to the client.
        reply: The string message from the client, which is expected to be a
            comma separated string containing the following components:
            [
                <state_id>,
                <object_tag>,
                <camera_position>,
                <camera_orientation>,
                <image>,
                ...
            ]
        bullet_camera_targets: A dictionary of target positions that we want
            Unity to point the camera at, in the format:
            {
                <id>: {
                    "position": <List[float]>  # The xyz position, in bullet world coordinate frame.
                    "save": <bool>,  # Whether to save an image using the camera.
                    "send": <bool>,  # Whether to send the image over the websocket.
                }
            }

    Returns:
        state_id: The state ID received in the reply.
        data: A dictionary containing the data in the message, in the format
            {
                <otag>:{
                    "camera_position": <camera_position>,
                    "camera_orientation": <camera_orientation>,
                    "image": <image>,
                },
                ...
            }
    """
    print(
        f"Received from client: {len(reply)} characters\tTime: {time.time()}"
    )

    # Split components by comma.
    reply = reply.split(",")
    print(f"Number of reply components: {len(reply)}")
    print(
        f"Attempting to parse unity data for {len(bullet_camera_targets)} objects..."
    )

    # Extract the state ID, and verify that the message IDs are equivalent,
    received_id = reply[0]
    assert received_id == message_id

    idx = 1
    data = {}
    for tid, info in bullet_camera_targets.items():
        unity_target_id = int(reply[idx])
        assert unity_target_id == tid
        idx += 1

        camera_position = [float(x) for x in reply[idx : idx + 3]]
        idx += 3
        camera_orientation = [float(x) for x in reply[idx : idx + 4]]
        idx += 4

        # Convert from base 64 to an image tensor.
        if info["should_send"]:
            rgb = base64_to_numpy_image(b64=reply[idx])
            idx += 1
            seg_img = base64_to_numpy_image(b64=reply[idx])
            idx += 1
        else:
            rgb, seg_img = None, None

        # Store the data.
        data[int(unity_target_id)] = {
            "camera_position": camera_position,
            "camera_orientation": camera_orientation,
            "rgb": rgb,
            "seg_img": seg_img,
        }

    return data


def base64_to_numpy_image(b64: str) -> np.ndarray:
    """Decodes a base 64 string representation of an image into a numpy array.
    Args:
        b64: The base 64 text to decode.
    
    Returns:
        image: The numpy version of the image.
    """
    image = np.array(Image.open(BytesIO(base64.b64decode(b64))))
    return image
