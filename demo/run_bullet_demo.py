"""Runs the demo in bullet."""
import demo.base_scenes
from demo.env import DemoEnvironment
from demo.options import OPTIONS
from demo.scene import SceneGenerator


def main():
    scene = SceneGenerator(
        base_scene=demo.base_scenes.SCENE, mu=OPTIONS.obj_mu
    ).generate()

    env = DemoEnvironment(
        opt=OPTIONS,
        scene=scene,
        observation_mode="gt",
        visualize_bullet=True,
        visualize_unity=False,
    )

    while 1:
        done = env.step()

        # We are done.
        if done:
            break


if __name__ == "__main__":
    main()
