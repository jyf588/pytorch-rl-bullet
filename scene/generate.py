"""Generates pickle files of scenes. Used to generate scenes to run the full system on."""
import os
import shutil
import argparse
from typing import *
from tqdm import tqdm

import exp.loader
import scene.options
import exp.options
from scene.generator import SceneGenerator
from ns_vqa_dart.bullet import util


def main(args: argparse.Namespace):
    dst_dir = os.path.join(args.scenes_dir, args.json_name)
    util.delete_and_create_dir(dst_dir)

    json_path = os.path.join(args.json_dir, f"{args.json_name}.json")
    task2opt, task2gen_opt = scene.options.create_options(json_path)

    # Create the scenes for each task, using the options for each task.
    for task in ["stack", "place"]:
        opt = task2opt[task]
        # Create the scene generator. Seeds are specified at the task-level.
        type2gen = create_generators(seed=opt["seed"], type2gen_opt=task2gen_opt[task])
        scenes = generate_scenes(n_scenes=opt["n_scenes"], type2gen=type2gen)

        # Save the scenes.
        save_scenes(dst_dir, task, scenes)


def create_generators(seed: int, type2gen_opt: Dict) -> Dict:
    """Create RandomObjectsGenerator's given generator options.

    Args:
        generator_options: A list of options, one for each generator.
    
    Returns:
        generators: A list of RandomObjectsGenerator's.
    """
    type2gen = {}
    for k, opt_list in type2gen_opt.items():
        type2gen[k] = [SceneGenerator(seed=seed, opt=opt) for opt in opt_list]
    return type2gen


def generate_scenes(n_scenes: int, type2gen: Dict):
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
        # Generate manipulable objects before generating surrounding objects.
        # Enforce uniqueness between manipulated objects and surrounding objects.
        # (But not within surrounding objects themselves).
        manip_scene: List[Dict] = []
        for gen in type2gen["manip"]:
            manip_scene += gen.generate_tabletop_objects(
                existing_odicts=manip_scene, unique_odicts=manip_scene
            )
        surround_scene = []
        for gen in type2gen["surround"]:
            surround_scene += gen.generate_tabletop_objects(
                existing_odicts=manip_scene, unique_odicts=manip_scene
            )

        # Combine the scenes. Ensure that manipulable objects come first.
        scene = manip_scene + surround_scene
        scenes.append(scene)
    return scenes


def save_scenes(dst_dir: str, task: str, scenes: List[Dict]):
    task_dir = os.path.join(dst_dir, task)
    os.makedirs(task_dir)
    for i, s in enumerate(scenes):
        path = os.path.join(task_dir, f"{i:04}.json")
        util.save_json(path=path, data=s)
    print(f"Saved {task} scenes to: {task_dir}")


def load_scenes(load_dir: str, task) -> List[Dict]:
    if task is None:
        task_dir = load_dir
    else:
        task_dir = os.path.join(load_dir, task)
    scenes = []
    for f in sorted(os.listdir(task_dir)):
        path = os.path.join(task_dir, f)
        scene = util.load_json(path=path)
        scenes.append(scene)
    return scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_name", type=str, help="The name of the json file containing params.",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="scene/json",
        help="The directory of json files.",
    )
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default="/home/mguo/data/dash",
        help="The directory to save scenes to.",
    )
    args = parser.parse_args()
    main(args)
