"""Generates pickle files of scenes. Used to generate scenes to run the full system on."""
import argparse
from typing import *
from tqdm import tqdm

import scenes.options as OPT
from scenes.generator import SceneGenerator


def main(args: argparse.Namespace):
    for task in OPT.TASK_LIST:
        print(f"Generating scenes for task: {task}...")
        generators = create_generators(
            seed=OPT.EXPERIMENT2SEED[args.experiment][task],
            generator_options=OPT.TASK2OPTIONS[task],
        )
        _ = generate_scenes(n_scenes=OPT.N_SCENES, generators=generators)


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
        "experiment",
        type=str,
        choices=list(OPT.EXPERIMENT2SEED.keys()),
        help="The name of the experiment to run.",
    )
    args = parser.parse_args()
    main(args)
