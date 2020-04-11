"""A test to check transformations between bullet world coordinate frame and
unity camera coordinate frame.
"""
import numpy as np

from ns_vqa_dart.bullet import util
from bullet2unity import const, states


def main():
    # Inputs.
    p_bw = np.array([-8.3, -9.7, 2.0]) / 100
    orn_bw = [0.0, 0.0, 0.0, 1.0]
    up_bw = util.orientation_to_up(orientation=orn_bw)

    print("Inputs:")
    print(f"p_bw: {p_bw}")
    print(f"orn_bw: {orn_bw}")
    print(f"up_bw: {up_bw}")
    print()

    """
    Camera pose (gen_dataset example)

    y right before saving: 
        [  0.           0.           1.           0.           0.
        1.           0.           0.04814836   0.09629671 -10.949661
        1.7312028   -2.4183233    0.01448431   0.07868722   0.9967941 ]
    
    up_bw: [0.0, 0.0, 1.0]
    up_uw: [-1.2246467991473532e-16, -0.0, -1.0]
    up_uc: [-4.548458843142521, 0.6100001328911564, -2.8075033713003124]
    """
    uworld_cam_position = [-4.950212, 1.524235, -2.754882]
    uworld_cam_orientation = [0.07587584, -0.9197314, -0.9197314, 0.3168075]

    # Camera pose (html example)
    # uworld_cam_position = [-5.025658, 1.496986, -2.6641]
    # uworld_cam_orientation = [-0.1608511, 0.7957029, 0.7957029, -0.532488]

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

    up_uw = states.bullet2unity_up(bullet_up=up_bw)
    up_uc = util.apply_transform(xyz=up_uw, transformation=T_uw_uc)

    print(f"p_bw: {p_bw}")
    print(f"p_uc: {p_uc}")
    print(f"up_bw: {up_bw}")
    print(f"up_uw: {up_uw}")
    print(f"up_uc: {up_uc}")
    print()
    # From unity camera to bullet world.
    p_us = util.apply_inv_transform(xyz=p_uc, transformation=T_us_uc)
    p_bs = states.unity2bullet_position(unity_position=p_us)
    p_bw = util.apply_inv_transform(xyz=p_bs, transformation=T_bw_bs)

    up_uw = util.apply_inv_transform(xyz=up_uc, transformation=T_uw_uc)
    up_bw = states.unity2bullet_up(unity_up=up_uw)

    print(f"T_uw_uc: {T_uw_uc}")

    print(f"p_bw: {p_bw}")
    print(f"p_uc: {p_uc}")

    print(f"up_uc: {up_uc}")
    print(f"up_uw: {up_uw}")
    print(f"up_bw: {up_bw}")


if __name__ == "__main__":
    main()
