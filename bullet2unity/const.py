"""Contains various constants for the bullet2unity interface."""

BULLET_SHOULDER_POS = [
    -0.30000001192092896,
    0.28200000524520874,
    0.21199999749660492,
]
# note:
# in inmoov_shadow_hand_v2,  self.base_init_pos is [-0.30, 0.348, 0.272]
# up to a translation of "0 -0.066 -0.06"(r_shoulder_out_joint) in the urdf file

UNITY_SHOULDER_POS = [-5.000653, 1.402408, -2.594684]

"""Robot-related constants"""

# A list of joint names to send to the client.
SEND_JOINT_NAMES = [
    "r_shoulder_out_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_elbow_roll_joint",
    "rh_WRJ2",
    "rh_WRJ1",
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_FFJ2",
    "rh_FFJ1",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_MFJ2",
    "rh_MFJ1",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_RFJ2",
    "rh_RFJ1",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_LFJ2",
    "rh_LFJ1",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
]

DEFAULT_ROBOT_STATE = {joint_name: 0.0 for joint_name in SEND_JOINT_NAMES}
