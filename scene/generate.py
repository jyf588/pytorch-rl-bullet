"""Generates pickle files of scenes. Used to generate scenes to run the full system on."""
import os
import argparse
from typing import *
from tqdm import tqdm

import exp.loader
import scene.options
import exp.options
from scene.generator import SceneGenerator


def main(args: argparse.Namespace):
    print(f"Generating scenes for experiment: {args.exp}...")
    set_name2opt = exp.loader.ExpLoader(exp_name=args.exp).set_name2opt
    for set_name, set_opt in set_name2opt.items():
        # Generate scenes.
        generators = create_generators(
            seed=set_opt["seed"],
            generator_options=scene.options.TASK2OPTIONS[set_opt["task"]],
        )
        scenes = generate_scenes(n_scenes=set_opt["n_scenes"], generators=generators)

        # Save the scenes.
        set_loader = exp.loader.SetLoader(exp_name=args.exp, set_name=set_name)
        set_loader.save_scenes(scenes)


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
        "exp", type=str, help="The name of the experiment to run.",
    )
    args = parser.parse_args()
    main(args)
