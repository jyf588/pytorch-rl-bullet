import json
import os


joints = [
    "r_shoulder_lift_joint",
    "r_shoulder_out_joint",
    "r_upper_arm_roll_joint",
]


vision_dir = "/home/michelle/demo_poses/vision"

name2angles = {}

for scene_name in os.listdir(vision_dir):
    scene_dir = os.path.join(vision_dir, scene_name)
    for json_fname in os.listdir(scene_dir):
        poses_path = os.path.join(scene_dir, json_fname)

    with open(poses_path, "r") as f:
        poses = json.load(f)

    for pose in poses:
        for name, angle in pose["robot"].items():
            # print(name)
            if name in joints:
                if name not in name2angles:
                    name2angles[name] = []
                name2angles[name].append(angle)

for name, angles in name2angles.items():
    min_angle = min(angles)
    max_angle = max(angles)
    print(f"Joint: {name}\tMin angle: {min_angle}\tMax angle: {max_angle}")
