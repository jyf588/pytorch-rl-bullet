import os
import copy
import argparse
import ns_vqa_dart.bullet.util as util


BASE_SYSTEM_OPTIONS = argparse.Namespace(
    seed=101,
    is_cuda=True,
    enable_reaching=True,
    enable_retract=True,
    scene_place_src_idx=0,
    scene_place_dst_idx=1,
    scene_stack_src_idx=0,
    scene_stack_dst_idx=1,
    obs_mode=None,
    obs_noise=False,
    position_noise=0.03,
    upv_noise=0.04,
    height_noise=0.02,
    render_unity=False,
    render_bullet=False,
    visualize_unity=False,
    use_control_skip=True,
    render_frequency=100,
    render_obs=False,
    animate_head=False,
    save_states=False,
    container_dir=None,
    policy_id="0404",  # [0404, 0411, 0510]
    table1_dir="figures/table1",
    root_outputs_dir=os.path.join(util.get_user_homedir(), "outputs/system"),
)

VISION_STATES_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
VISION_STATES_OPTIONS.enable_reaching = False
VISION_STATES_OPTIONS.enable_retract = False
VISION_STATES_OPTIONS.obs_mode = "gt"
VISION_STATES_OPTIONS.obs_noise = False
VISION_STATES_OPTIONS.save_states = True
VISION_STATES_OPTIONS.container_dir = "/home/mguo/container_data_v1"

UNITY_DATASET_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
UNITY_DATASET_OPTIONS.render_frequency = 1  # Render and save every state.
UNITY_DATASET_OPTIONS.render_unity = True

TEST_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
TEST_OPTIONS.enable_reaching = True
TEST_OPTIONS.enable_retract = True
TEST_OPTIONS.container_dir = "/home/mguo/container_data_v2"

TEST_VISION_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_VISION_OPTIONS.obs_mode = "vision"
TEST_VISION_OPTIONS.render_unity = True

TEST_GT_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_GT_OPTIONS.obs_mode = "gt"

DEBUG_VISION_OPTIONS = copy.deepcopy(TEST_OPTIONS)
DEBUG_VISION_OPTIONS.obs_mode = "vision"
DEBUG_VISION_OPTIONS.render_obs = True
DEBUG_VISION_OPTIONS.render_unity = True
DEBUG_VISION_OPTIONS.container_dir = "/home/mguo/container_data_v1"


SYSTEM_OPTIONS = {
    "vision_states": VISION_STATES_OPTIONS,
    "unity_dataset": UNITY_DATASET_OPTIONS,
    "test_vision": TEST_VISION_OPTIONS,
    "test_gt": TEST_GT_OPTIONS,
    "debug_vision": DEBUG_VISION_OPTIONS,
}


BULLET_OPTIONS = argparse.Namespace(
    det=True,
    bullet_contact_iter=200,
    det_contact=0,
    ts=1.0 / 240,
    hand_mu=1.0,
    floor_mu=1.0,
)


POLICY_OPTIONS = argparse.Namespace(
    seed=101,
    init_noise=True,
    restore_fingers=True,
    use_height=False,
    n_plan_steps=305,
    grasp_control_steps=35,
    place_control_steps=75,
    control_skip=6,
    grasp_env_name="InmoovHandGraspBulletEnv-v6",
    place_env_name="InmoovHandPlaceBulletEnv-v9",
    vision_delay=2,
)


def get_policy_options_and_paths(policy_id: str):
    policy_options = copy.deepcopy(POLICY_OPTIONS)
    if policy_id == "0411":
        grasp_pi = f"0411_0_n_25_45"
        grasp_dir = f"./trained_models_0411_0_n/ppo/"
        place_dir = f"./trained_models_0411_0_n_place_0411_0/ppo/"
    elif policy_id == "0404":
        policy_options.use_height = True
        grasp_pi = f"0404_0_n_20_40"
        grasp_dir = f"./trained_models_%s/ppo/" % "0404_0_n"
        place_pi = f"0404_0_n_place_0404_0"
        place_dir = f"./trained_models_%s/ppo/" % place_pi

    shape2policy_paths = {
        "universal": {
            "grasp_pi": grasp_pi,
            "grasp_dir": grasp_dir,
            "place_dir": place_dir,
        },
        "sphere": {
            "grasp_pi": "0422_sph_n_25_45",
            "grasp_dir": "./trained_models_0422_sph_n/ppo/",
            "place_dir": "./trained_models_0422_sph_n_place_0422_sph/ppo/",
        },
    }
    return policy_options, shape2policy_paths


VISION_OPTIONS = argparse.Namespace(
    renderer="unity",
    use_segmentation_module=True,
    separate_vision_modules=True,
    use_gt_obs=False,
    seg_checkpoint_path="/home/mguo/outputs/detectron/2020_04_27_20_12_14/model_final.pth",
    planning_checkpoint_path="/home/mguo/outputs/planning_v003_20K/checkpoint_best.pt",
    placing_checkpoint_path="/home/mguo/outputs/placing_v003_2K_20K/checkpoint_best.pt",
    stacking_checkpoint_path="/home/mguo/outputs/stacking_v003_2K_20K/checkpoint_best.pt",
    # seg_checkpoint_path="/home/mguo/outputs/detectron/seg_tiny/2020_05_11_20_52_07/model_final.pth",
    # attr_checkpoint_path="/home/mguo/outputs/stacking_v003_2K_20K/checkpoint_best.pt",
    # attr_checkpoint_path="/home/mguo/outputs/attr_net/seg_tiny/checkpoint_best.pt",
    coordinate_frame="unity_camera",
    save_predictions=True,
    debug_dir=None,
)
