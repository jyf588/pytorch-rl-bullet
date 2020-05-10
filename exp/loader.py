import os

import ns_vqa_dart.bullet.util as util


def get_exp_set_dir(exp: str, set_name: str, create_dir=False) -> str:
    set_dir = os.path.join(util.get_user_homedir(), "data/dash", exp, set_name)
    if create_dir:
        util.delete_and_create_dir(set_dir)
    return set_dir

