import inspect
import numpy as np
import os
import pybullet as p

from my_pybullet_envs.inmoov_arm_obj_imaginary_sessions import (
    ImaginaryArmObjSession,
)
from my_pybullet_envs.inmoov_shadow_place_env_v9 import (
    InmoovShadowHandPlaceEnvV9,
)
from my_pybullet_envs import utils

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)


g_tx = 0.2
g_ty = 0.4
p_tx = 0.1
p_ty = -0.05
p_tz = 0.16096354831753318
GRASP_PI = "0313_2_n_25_45"

p.connect(p.GUI)

sess = ImaginaryArmObjSession()

Qreach = np.array(sess.get_most_comfortable_q_and_refangle(g_tx, g_ty)[0])

desired_obj_pos = [p_tx, p_ty, utils.PLACE_START_CLEARANCE + p_tz]
a = InmoovShadowHandPlaceEnvV9(renders=False, grasp_pi_name=GRASP_PI)
a.seed(101)

table_id = p.loadURDF(
    os.path.join("my_pybullet_envs/assets/tabletop.urdf"),
    utils.TABLE_OFFSET,
    useFixedBase=1,
)

p_pos_of_ave, p_quat_of_ave = p.invertTransform(
    a.o_pos_pf_ave, a.o_quat_pf_ave
)
# TODO: [1] is the 2nd candidate
Qdestin = utils.get_n_optimal_init_arm_qs(
    a.robot, p_pos_of_ave, p_quat_of_ave, desired_obj_pos, table_id
)[0]

print(f"g_tx: {g_tx}")
print(f"g_ty: {g_ty}")

print(f"p_tx: {p_tx}")
print(f"p_ty: {p_ty}")
print(f"p_tz: {p_tz}")

print(f"table_id: {table_id}")
print(f"a.o_pos_pf_ave: {a.o_pos_pf_ave}")
print(f"a.o_quat_pf_ave: {a.o_quat_pf_ave}")
print(f"p_pos_of_ave: {p_pos_of_ave}")
print(f"p_quat_of_ave: {p_quat_of_ave}")
print(f"desired_obj_pos: {desired_obj_pos}")
print(f"Qdestin: {Qdestin}")

"""
table_id: 4
a.o_pos_pf_ave: [0.026215675510054614, -0.06007841587896836, 0.09882477570470663]
a.o_quat_pf_ave: [0.59438397 0.38620253 0.66134631 0.2453087 ]
p_pos_of_ave: (-0.007356900721788406, -0.1177569255232811, -0.011924607679247856)
p_quat_of_ave: (0.594383955001831, 0.38620251417160034, 0.6613463163375854, -0.24530869722366333)
desired_obj_pos: [0.1, -0.05, 0.3009635483175332]
Qdestin: [-1.237025880768228, -0.29011387801944555, -0.6697510090107007, -1.2814231160276015, -0.6907147775522865, -0.6363587411204584, -0.18803418791061538]
"""
