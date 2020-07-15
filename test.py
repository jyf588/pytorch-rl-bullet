import numpy as np


from my_pybullet_envs import utils
from my_pybullet_envs.inmoov_shadow_hand_v2 import InmoovShadowNew

robot = InmoovShadowNew(
    init_noise=False, timestep=utils.TS, np_random=np.random,
)

robot2 = InmoovShadowNew(
    init_noise=False, timestep=utils.TS, np_random=np.random,
)

robot2.arm_id = self.sim.loadURDF(os.path.join(currentdir,
                                             "assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf"),
                                 list(self.base_init_pos), self.sim.getQuaternionFromEuler(list(self.base_init_euler)),
                                 flags=self.sim.URDF_USE_SELF_COLLISION | self.sim.URDF_USE_INERTIA_FROM_FILE
                                       | self.sim.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                 useFixedBase=1)