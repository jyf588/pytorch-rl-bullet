import gym
from gym.envs.registration import registry, make, spec
from .allegro_hand_env import AllegroHandPickEnv
from .allegro_hand_nofloor_env import AllegroHandGraspEnv
from .inmoov_hand_nofloor_env import InmoovHandGraspEnv
from .shadow_hand_grasp_env import ShadowHandGraspEnv
# from .inmoov_shadow_hand_grasp_env import InmoovShadowHandGraspEnv
# from .inmoov_shadow_hand_grasp_env_tmp import InmoovShadowHandGraspEnvTmp
from .shadow_hand_place_env import ShadowHandPlaceEnv
from .shadow_hand_grasp_env_velc import ShadowHandGraspEnvVelC
# from .shadow_hand_grasp_env_pc_simple import ShadowHandGraspEnvPC
# from .inmoov_shadow_hand_demo_fixed_grasp_env import InmoovShadowHandDemoFixedGraspEnv
from .inmoov_shadow_grasp_env_v2 import InmoovShadowHandGraspEnvNew
from .inmoov_shadow_place_env_v2 import InmoovShadowHandPlaceEnvNew
from .inmoov_shadow_demo_env_v2 import InmoovShadowHandDemoEnvNew
from .inmoov_shadow_place_env_v3 import InmoovShadowHandPlaceEnvV3
from .inmoov_shadow_grasp_env_v3 import InmoovShadowHandGraspEnvV3
from .inmoov_shadow_place_env_v4 import InmoovShadowHandPlaceEnvV4
from .inmoov_shadow_grasp_env_v4 import InmoovShadowHandGraspEnvV4
from .inmoov_shadow_place_env_v5 import InmoovShadowHandPlaceEnvV5
from .inmoov_shadow_grasp_place_env_v1 import InmoovShadowHandGraspPlaceEnvV1


def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id='AllegroHandPickBulletEnv-v0',
    entry_point='my_pybullet_envs:AllegroHandPickEnv',
    max_episode_steps=700,
)

register(
    id='AllegroHandGraspBulletEnv-v0',
    entry_point='my_pybullet_envs:AllegroHandGraspEnv',
    max_episode_steps=400,
)

register(
    id='InmoovHandGraspBulletEnv-v0',
    entry_point='my_pybullet_envs:InmoovHandGraspEnv',
    max_episode_steps=400,
)

register(
    id='InmoovHandGraspBulletEnv-v1',
    entry_point='my_pybullet_envs:InmoovShadowHandGraspEnvNew',
    max_episode_steps=134,
)

register(
    id='InmoovHandGraspBulletEnv-v3',
    entry_point='my_pybullet_envs:InmoovShadowHandGraspEnvV3',
    max_episode_steps=134,
)

register(
    id='InmoovHandGraspBulletEnv-v4',
    entry_point='my_pybullet_envs:InmoovShadowHandGraspEnvV4',
    max_episode_steps=134,
)

register(
    id='InmoovHandPlaceBulletEnv-v1',
    entry_point='my_pybullet_envs:InmoovShadowHandPlaceEnvNew',
    max_episode_steps=100,
)

register(
    id='InmoovHandPlaceBulletEnv-v3',
    entry_point='my_pybullet_envs:InmoovShadowHandPlaceEnvV3',
    max_episode_steps=100,
)

register(
    id='InmoovHandPlaceBulletEnv-v4',
    entry_point='my_pybullet_envs:InmoovShadowHandPlaceEnvV4',
    max_episode_steps=100,
)

register(
    id='InmoovHandPlaceBulletEnv-v5',
    entry_point='my_pybullet_envs:InmoovShadowHandPlaceEnvV5',
    max_episode_steps=100,
)

register(
    id='ShadowHandGraspBulletEnv-v0',
    entry_point='my_pybullet_envs:ShadowHandGraspEnv',
    max_episode_steps=400,
)

register(
    id='ShadowHandGraspBulletEnv-v1',
    entry_point='my_pybullet_envs:ShadowHandGraspEnvVelC',
    max_episode_steps=100,
)

# register(
#     id='ShadowHandGraspBulletEnv-v2',
#     entry_point='my_pybullet_envs:ShadowHandGraspEnvPC',
#     max_episode_steps=400,
# )

register(
    id='ShadowHandPlaceBulletEnv-v0',
    entry_point='my_pybullet_envs:ShadowHandPlaceEnv',
    max_episode_steps=300,
)

register(
    id='ShadowHandDemoBulletEnv-v1',
    entry_point='my_pybullet_envs:InmoovShadowHandDemoEnvNew',
    max_episode_steps=8000,
)

# register(
#     id='InmoovShadowHandDemoGraspBulletEnv-v0',
#     entry_point='my_pybullet_envs:InmoovShadowHandDemoFixedGraspEnv',
#     max_episode_steps=400,
# )

register(
    id='InmoovHandGraspPlaceBulletEnv-v1',
    entry_point='my_pybullet_envs:InmoovShadowHandGraspPlaceEnvV1',
    max_episode_steps=300,      # large enough, controlled by done
)

def getList():
  btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
  return btenvs