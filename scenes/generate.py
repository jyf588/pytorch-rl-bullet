"""Generates pickle files of scenes. Used to generate scenes to run the full system on."""
import os
import argparse
from typing import *
from tqdm import tqdm

import scenes.options as OPT
from scenes.generator import SceneGenerator
import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    print(f"Generating scenes for experiment: {args.experiment}...")
    exp_opt = OPT.EXPERIMENT_OPTIONS[args.experiment]
    for set_name, set_opt in exp_opt.items():
        generators = create_generators(
            seed=set_opt["seed"], generator_options=OPT.TASK2OPTIONS[set_opt["task"]],
        )
        scenes = generate_scenes(n_scenes=set_opt["n_scenes"], generators=generators)

        # Saving.
        set_dir = os.path.join(
            util.get_user_homedir(), "data/dash", args.experiment, set_name
        )
        util.delete_and_create_dir(set_dir)
        path = os.path.join(set_dir, "scenes.p")
        util.save_pickle(path=path, data=scenes)
        print(f"Saved scenes to: {path}.")


def create_generators(seed: int, generator_options: List) -> List:
    """Create RandomObjectsGenerator's given generator options.

    Args:
        generator_options: A list of options, one for each generator.
    
    Returns:
        generators: A list of RandomObjectsGenerator's.
    """
    generators = [SceneGenerator(seed=seed, opt=opt) for opt in generator_options]
    return generators


def generate_scenes(n_scenes: int, generators: List):
    """
    Args:
        n_scenes: The number of scenes to generate.
        generators: Scene generators. Generators for manipulated objects must 
            come before generators for surrounding objects.
    
    Returns:
        scenes: A scene, which is a list of object dictionaries, ordered 
            according to the order of generators provided.
    """
    scenes = []
    for _ in tqdm(range(n_scenes)):
        scene = []
        for generator in generators:
            # Enforce uniqueness between manipulated objects and surrounding objects.
            # (But not within surrounding objects themselves)
            scene += generator.generate_tabletop_objects(
                existing_odicts=scene, unique_odicts=scene
            )
        scenes.append(scene)
    return scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment", type=str, help="The name of the experiment to run.",
    )
    args = parser.parse_args()
    main(args)
