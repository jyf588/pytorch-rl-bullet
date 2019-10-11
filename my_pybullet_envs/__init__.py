import gym
from gym.envs.registration import registry, make, spec
from .allegro_hand_env import AllegroHandPickEnv


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

def getList():
  btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
  return btenvs