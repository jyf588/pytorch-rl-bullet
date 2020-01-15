import gym
from gym.envs.registration import registry, make, spec
from .allegro_hand_env import AllegroHandPickEnv
from .allegro_hand_nofloor_env import AllegroHandGraspEnv
from .inmoov_hand_nofloor_env import InmoovHandGraspEnv
from .shadow_hand_grasp_env import ShadowHandGraspEnv
from .inmoov_shadow_hand_grasp_env import InmoovShadowHandGraspEnv
from .inmoov_shadow_hand_grasp_env_tmp import InmoovShadowHandGraspEnvTmp
from .shadow_hand_place_env import ShadowHandPlaceEnv
from .shadow_hand_grasp_env_velc import ShadowHandGraspEnvVelC
# from .shadow_hand_grasp_env_pc_simple import ShadowHandGraspEnvPC
# from .inmoov_shadow_hand_demo_fixed_grasp_env import InmoovShadowHandDemoFixedGraspEnv
from .inmoov_shadow_grasp_env_v2 import InmoovShadowHandGraspEnvNew
from .inmoov_shadow_place_env_v2 import InmoovShadowHandPlaceEnvNew


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
    id='InmoovHandPlaceBulletEnv-v1',
    entry_point='my_pybullet_envs:InmoovShadowHandPlaceEnvNew',
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
    id='InmoovShadowHandGraspBulletEnv-v0',
    entry_point='my_pybullet_envs:InmoovShadowHandGraspEnv',
    max_episode_steps=400,
)

register(
    id='InmoovShadowHandGraspBulletEnvTmp-v0',
    entry_point='my_pybullet_envs:InmoovShadowHandGraspEnvTmp',
    max_episode_steps=800,
)

# register(
#     id='InmoovShadowHandDemoGraspBulletEnv-v0',
#     entry_point='my_pybullet_envs:InmoovShadowHandDemoFixedGraspEnv',
#     max_episode_steps=400,
# )

def getList():
  btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
  return btenvs