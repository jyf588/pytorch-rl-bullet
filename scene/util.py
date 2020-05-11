from typing import *


def convert_scene_for_placing(opt, scene: List) -> Tuple:
    """Converts the scene into a modified scene for placing, with the following steps:
        1. Denote the (x, y) location of object index 0 as the placing destination (x, y).
        2. Remove object index 0 from the scene list.
    
    Args:
        scene: The original scene.
    
    Returns:
        new_scene: The scene modified for placing.
        place_dst_xy: The (x, y) placing destination for the placing task.
    """
    # Make sure that removing the destination object from the scene won't modify the
    # source object index.
    assert opt.scene_place_src_idx < opt.scene_place_dst_idx

    # Remove the destination object from the scene.
    new_scene = copy.deepcopy(
        scene[: opt.scene_place_dst_idx] + scene[opt.scene_place_dst_idx + 1 :]
    )

    # Use the location of the destination object as the (x, y) placing destination.
    place_dst_xy = scene[opt.scene_place_dst_idx]["position"][:2]

    # Construct an imaginary object with the same shape attribute as the source object
    # to visualize the placing destination.
    place_dest_object = copy.deepcopy(scene[opt.scene_place_src_idx])
    place_dest_object["position"] = place_dst_xy + [0.0]
    place_dest_object["height"] = 0.005
    return new_scene, place_dst_xy, place_dest_object
