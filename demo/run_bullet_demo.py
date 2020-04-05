"""Runs the demo in bullet."""
import demo.base_scenes
from demo.env import DemoEnvironment
from demo.options import OPTIONS
from demo.scene import SceneGenerator
import my_pybullet_envs.utils as utils
from ns_vqa_dart.bullet.random_objects import RandomObjectsGenerator


def main():
    # generator = SceneGenerator(
    #     base_scene=demo.base_scenes.SCENE1,
    #     seed=OPTIONS.seed,
    #     mu=OPTIONS.obj_mu,
    # )
    generator = RandomObjectsGenerator(
        seed=OPTIONS.seed,
        n_objs_bounds=(2, 2),
        obj_dist_thresh=0.2,
        max_retries=50,
        shapes=["box"],
        radius_bounds=(utils.HALF_W_MIN, utils.HALF_W_MAX),
        height_bounds=(utils.H_MIN, utils.H_MAX),
        x_bounds=(utils.TX_MIN, utils.TX_MAX),
        y_bounds=(utils.TY_MIN, utils.TY_MAX),
        z_bounds=(0.0, 0.0),
        position_mode="com",
    )
    for _ in range(1):
        scene = generator.generate_tabletop_objects()
        scene[0]["color"] = "green"
        scene[1]["color"] = "blue"
        env = DemoEnvironment(
            opt=OPTIONS,
            scene=scene,
            command=f"Put the green box on top of the blue box.",
            observation_mode="gt",
            visualize_bullet=False,
            visualize_unity=False,
        )

        while 1:
            done = env.step()
            if done:
                break
        del env


if __name__ == "__main__":
    main()
