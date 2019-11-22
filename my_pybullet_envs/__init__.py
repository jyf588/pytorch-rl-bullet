import gym
from gym.envs.registration import registry, make, spec
from .allegro_hand_env import AllegroHandPickEnv
from .allegro_hand_nofloor_env import AllegroHandGraspEnv
from .inmoov_hand_nofloor_env import InmoovHandGraspEnv


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

def getList():
  btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
  return btenvs