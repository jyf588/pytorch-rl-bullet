"""Runs the demo in bullet."""
import demo.base_scenes
from demo.env import DemoEnvironment
from demo.options import OPTIONS
from demo.scene import SceneGenerator


def main():
    generator = SceneGenerator(
        base_scene=demo.base_scenes.SCENE1,
        seed=OPTIONS.seed,
        mu=OPTIONS.obj_mu,
    )
    for _ in range(1):
        env = DemoEnvironment(
            opt=OPTIONS,
            scene=generator.generate(),
            command="Put the green box on top of the blue cylinder",
            observation_mode="gt",
            visualize_bullet=True,
            visualize_unity=False,
        )

        while 1:
            done = env.step()
            if done:
                break
        del env


if __name__ == "__main__":
    main()
