"""A test to check transformations between bullet world coordinate frame and
unity camera coordinate frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from ns_vqa_dart.bullet import util
from bullet2unity import const, states


def main():
    # Inputs.
    p_bw = np.array([-8.3, -9.7, 2.0]) / 100
    # orn_bw = [0.0, 0.0, 0.0, 1.0]
    orn_bw = [-0.25, 0.04, -0.13, 0.96]
    # euler_bw_list = [
    #     [0.0, 0.0, 0.0],
    #     [90, 0.0, 0.0],
    #     # [0.0, 90.0, 0.0],
    #     # [0.0, 0.0, 90.0],
    # ]

    # Camera pose
    uworld_cam_position = [-4.990062, 1.524398, -2.714187]
    uworld_cam_orientation = [0.1024841, -0.8730335, -0.8730335, 0.4292397]

    test(
        p_bw, np.array(orn_bw), uworld_cam_position, uworld_cam_orientation,
    )


def test(p_bw, orn_bw, uworld_cam_position, uworld_cam_orientation):
    # Compute transformation matrices.
    T_bw_bs = util.create_transformation(
        position=const.BULLET_SHOULDER_POS, orientation=[0.0, 0.0, 0.0, 1.0]
    )
    T_uw_us = util.create_transformation(
        position=const.UNITY_SHOULDER_POS, orientation=[0.0, 0.0, 0.0, 1.0]
    )
    T_us_uw = np.linalg.inv(T_uw_us)
    T_uw_uc = util.create_transformation(
        position=uworld_cam_position, orientation=uworld_cam_orientation
    )
    T_us_uc = T_uw_uc.dot(T_us_uw)

    # From bullet world to unity camera.
    p_bs = util.apply_transform(xyz=p_bw, transformation=T_bw_bs)
    p_us = states.bullet2unity_position(bullet_position=p_bs)
    p_uc = util.apply_transform(xyz=p_us, transformation=T_us_uc)
    # From unity camera to bullet world.
    p_us = util.apply_inv_transform(xyz=p_uc, transformation=T_us_uc)
    p_bs = states.unity2bullet_position(unity_position=p_us)
    p_bw = util.apply_inv_transform(xyz=p_bs, transformation=T_bw_bs)

    # Quat -> euler
    euler_uw = states.bullet2unity_euler(orn_bw)
    # euler_uw[2] = 0.0
    # euler -> up
    r = R.from_euler("xyz", euler_uw, degrees=True)
    rotmat = r.as_matrix()
    up_uw = rotmat[:, -1]
    # up_uw = util.euler_to_up(euler_uw)
    # world -> camera up
    euler_uc = util.apply_transform(xyz=euler_uw, transformation=T_uw_uc)

    up_uc = util.euler_to_up(euler_uc)

    print(f"Forward:")
    print(f"rotmat: {rotmat}")
    print(f"euler_uw: {euler_uw}")
    print(f"up_uw: {up_uw}")
    print(f"up_uc: {up_uc}")

    # camera -> world up
    up_uw = util.apply_inv_transform(xyz=up_uc, transformation=T_uw_uc)
    # up -> euler
    rotmat = np.zeros((3, 3))
    rotmat[:, -1] = up_uw
    r = R.from_matrix(rotmat)
    euler_uw = r.as_euler("xyz", degrees=True)

    up_bw = states.unity2bullet_up(unity_up=up_uw)
    print(f"Backwards")
    print(f"up_uw: {up_uw}")
    print(f"rotmat: {rotmat}")
    print(f"euler_uw: {euler_uw}")


def test2():
    uworld_cam_position = [-4.990062, 1.524398, -2.714187]
    uworld_cam_orientation = [0.1024841, -0.8730335, -0.8730335, 0.4292397]
    T_uw_uc = util.create_transformation(
        position=uworld_cam_position, orientation=uworld_cam_orientation
    )

    q_bw1 = [-0.25, 0.04, -0.13, 0.96]

    up = util.orientation_to_up(q_bw1)
    up = up / np.linalg.norm(up)
    print(f"orientation_to_up: {up}")

    e_bw1 = R.from_quat(q_bw1).as_euler("xyz", degrees=True)
    e_bw1[2] = 0.0  # Set z rotation to zero.
    u_bw1 = e_bw1 / np.linalg.norm(e_bw1)

    u_uc = bworld2ucam(vec_bw=up, T_uw_uc=T_uw_uc)
    u_bw2 = ucam2bworld(vec_uc=u_uc, T_uw_uc=T_uw_uc)

    q_bw2 = util.up_to_orientation(u_bw2)

    print(f"q_bw1: {q_bw1}")
    print(f"q_bw2: {q_bw2}")

    print(f"u_bw1: {u_bw1}")
    print(f"u_bw2: {u_bw2}")

    for i in range(3):
        denom = 1.0 if u_bw1[i] == 0.0 else u_bw1[i]
        print(np.abs(u_bw1[i] - u_bw2[i]) / np.abs(denom))


def bworld2ucam(vec_bw, T_uw_uc):
    # Convert from bullet to unity.
    x, y, z = vec_bw
    vec_uw = [-y, -z, x]

    # Transform from unity world to camera.
    vec_uc = util.apply_transform(xyz=vec_uw, transformation=T_uw_uc)
    return vec_uc


def ucam2bworld(vec_uc, T_uw_uc):
    vec_uw = util.apply_inv_transform(xyz=vec_uc, transformation=T_uw_uc)
    x, y, z = vec_uw
    vec_bw = [z, -x, -y]
    return vec_bw


if __name__ == "__main__":
    test2()
