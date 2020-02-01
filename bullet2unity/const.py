"""Contains various project constants."""
import pybullet as p
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

# TODO: Fill this with your own "<path_to_inmoov_ros_repo>/inmoov_description/robots/inmoov_shadow_hand_v2_1.urdf",
ROBOT_URDF_PATH = os.path.join(parentdir,
    'my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1.urdf')
ASSETS_PATH = os.path.join(parentdir, 'my_pybullet_envs/assets')

"""Object-related constants"""
TABLE = [
    os.path.join(ASSETS_PATH, "tabletop.urdf"),
    0.200000,
    0.100000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    1.000000,
]

OBJECTS = [
    [
        os.path.join(ASSETS_PATH, "box.urdf"),
        0.200000,
        0.000000,
        0.100000,
        0.000000,
        0.000000,
        0.000000,
        1.000000,
    ],
    [
        os.path.join(ASSETS_PATH, "box.urdf"),
        0.100000,
        0.200000,
        0.100000,
        0.000000,
        0.000000,
        0.000000,
        1.000000,
    ],
    [
        os.path.join(ASSETS_PATH, "cylinder.urdf"),
        0.100000,
        -0.150000,
        0.100000,
        0.000000,
        0.000000,
        0.000000,
        1.000000,
    ],
]


"""Robot-related constants"""

ROBOT = [
    ROBOT_URDF_PATH,
    -0.300000,
    0.500000,
    -1.250000,
    0.000000,
    0.000000,
    0.000000,
    1.000000,
]

# Joint angles for the robot.
JOINT_ANGLES = [
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    -0.445277,  # RightArm
    0.005773,  # RightArm
    -0.519248,  # RightArm
    -1.873456,  # RightForeArm
    -0.440030,  # RightForeArm
    0.000000,  # skip
    -0.488880,  # RightHand
    -0.993523,  # RightHandWrist2
    0.000000,
    0.099977,
    0.399793,
    0.600794,
    0.499730,
    0.000000,
    0.094313,
    0.401571,
    0.603696,
    0.486814,
    0.000000,
    0.095436,
    0.400330,
    0.607240,
    0.495304,
    0.000000,
    0.000139,
    0.100520,
    0.399613,
    0.400063,
    0.399951,
    0.000000,
    -0.200602,
    0.999619,
    0.100527,
    0.501398,
    0.498885,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
]

# A list of joint indices to send to clients.
SEND_JOINTS = [
    25,  # RightArm (shoulder)
    26,  # RightArm
    27,  # RightArm
    28,  # RightForeArm
    29,  # RightForeArm
    31,  # RightHand
    32,  # RightHandWrist2
    34,
    35,
    36,
    37,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    49,
    50,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    59,
]

VARY_JOINTS = [
    25,  # RightArm X
    26,  # RightArm Z
    27,  # RightArm -Y
    28,  # RightForeArm Z
    29,  # RightForeArm Y
    31,  # RightHand Z: yes
    32,  # RightHandWrist2: yes
    34,  # RightHandIndex1 Z: yes
    35,  # RightHandIndex1 X: yes
    36,  # RightHandIndex2 X: yes
    37,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    49,  # RightHandMetaCarpal Y
    50,  # RightHandPinky1 Z: yes
    51,  # RightHandPinky1 X: yes, but slanted
    52,  # RightHandPinky2 X: yes, but pinky1 is slanted
    53,  # RightHandPinky3 X: yes
    55,  # RightHandThumb1 Y: yes
    56,  # RightHandThumb1 X: yes
    57,  # RightHandThumb2 X: yes
    58,  # RightHandThumb2 Z: yes
    59,  # RightHandThumb3 X -> Z: yes
]
