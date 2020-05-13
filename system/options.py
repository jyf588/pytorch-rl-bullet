import copy
import argparse


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
    render_unity=True,
    render_bullet=False,
    visualize_unity=False,
    use_control_skip=True,
    render_frequency=20,
    render_obs=False,
    animate_head=False,
    save_states=False,
    container_dir=None,
)

VISION_STATES_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
VISION_STATES_OPTIONS.enable_reaching = False
VISION_STATES_OPTIONS.enable_retract = False
VISION_STATES_OPTIONS.render_unity = False
VISION_STATES_OPTIONS.obs_mode = "gt"
VISION_STATES_OPTIONS.obs_noise = False
VISION_STATES_OPTIONS.save_states = True
VISION_STATES_OPTIONS.container_dir = "/home/mguo/container_data_v1"

UNITY_DATASET_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
UNITY_DATASET_OPTIONS.render_frequency = 1  # Render and save every state.

TEST_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
TEST_OPTIONS.enable_reaching = True
TEST_OPTIONS.enable_retract = True
TEST_OPTIONS.container_dir = "/home/mguo/container_data_v2"

TEST_VISION_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_VISION_OPTIONS.obs_mode = "vision"
TEST_VISION_OPTIONS.obs_noise = False
TEST_VISION_OPTIONS.render_obs = True

TEST_GT_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_GT_OPTIONS.obs_mode = "gt"
TEST_GT_OPTIONS.obs_noise = False
TEST_GT_OPTIONS.render_unity = False
TEST_GT_OPTIONS.render_bullet = False


SYSTEM_OPTIONS = {
    "vision_states": VISION_STATES_OPTIONS,
    "unity_dataset": UNITY_DATASET_OPTIONS,
    "test_vision": TEST_VISION_OPTIONS,
    "test_gt": TEST_GT_OPTIONS,
    "demo": None,
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

UNIVERSAL_POLICY_MODELS = argparse.Namespace(
    grasp_pi="0411_0_n_25_45",
    grasp_dir="./trained_models_0411_0_n/ppo/",
    place_dir="./trained_models_0411_0_n_place_0411_0/ppo/",
)

SPHERE_POLICY_MODELS = argparse.Namespace(
    grasp_pi="0422_sph_n_25_45",
    grasp_dir="./trained_models_0422_sph_n/ppo/",
    place_dir="./trained_models_0422_sph_n_place_0422_sph/ppo/",
)

NAME2POLICY_MODELS = {
    "universal": UNIVERSAL_POLICY_MODELS,
    "sphere": SPHERE_POLICY_MODELS,
}


VISION_OPTIONS = argparse.Namespace(
    renderer="unity",
    use_segmentation_module=False,
    separate_vision_modules=False,
    use_gt_obs=True,
    # seg_checkpoint_path="/home/mguo/outputs/detectron/2020_04_27_20_12_14/model_final.pth",
    # planning_checkpoint_path="/home/mguo/outputs/planning_v003_20K/checkpoint_best.pt",
    # placing_checkpoint_path="/home/mguo/outputs/placing_v003_2K_20K/checkpoint_best.pt",
    # stacking_checkpoint_path="/home/mguo/outputs/stacking_v003_2K_20K/checkpoint_best.pt",
    seg_checkpoint_path="/home/mguo/outputs/detectron/seg_tiny/2020_05_11_20_52_07/model_final.pth",
    # attr_checkpoint_path="/home/mguo/outputs/stacking_v003_2K_20K/checkpoint_best.pt",
    attr_checkpoint_path="/home/mguo/outputs/attr_net/seg_tiny/checkpoint_best.pt",
    coordinate_frame="unity_camera",
    save_predictions=True,
    debug_dir=None,
)
