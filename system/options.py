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
    render_unity=False,
    render_bullet=False,
    visualize_unity=False,
    use_control_skip=False,
    render_frequency=6,
    render_obs=False,
    animate_head=True,
    save_states=False,
    container_dir=None,
    table1_dir="figures/table1",
    two_commands=False
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
TEST_OPTIONS.container_dir = "/home/yifengj/container_data"     # TODO

TEST_VISION_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_VISION_OPTIONS.obs_mode = "vision"
TEST_VISION_OPTIONS.render_obs = False
TEST_VISION_OPTIONS.render_unity = True

TEST_GT_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_GT_OPTIONS.obs_mode = "gt"


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
    use_arm_blending=True,
    use_height=True,        # assume sph policy does not use height or one bit.
    use_place_stack_bit=True,
    use_slow_policy=False,
    n_reach_steps=305,
    n_transport_steps=505,
    n_retract_steps=305,
    grasp_control_steps=30,
    place_control_steps=55,
    grasp_control_skip=12,
    place_control_skip=6,
    place_clip_init_tar=False,
    place_clip_init_tar_value=0.2,
    grasp_env_name="InmoovHandGraspBulletEnv-v6",
    place_env_name="InmoovHandPlaceBulletEnv-v9",
    vision_delay=2,
)

UNIVERSAL_POLICY_NAMES = argparse.Namespace(
    # grasp_pi="0411_0_n_25_45",
    # grasp_dir="./trained_models_0411_0_n/ppo/",
    # place_dir="./trained_models_0411_0_n_place_0411_0/ppo/",
    # grasp_pi="0404_0_n_20_40",
    # grasp_dir="./trained_models_0404_0_n/ppo/",
    # place_dir="./trained_models_0404_0_n_place_0404_0/ppo/",
    grasp_pi="0510_0_n_25_45",
    grasp_dir="./trained_models_0510_0_n/ppo/",         # TODO
    place_dir="./trained_models_0510_0_n_place_0510_0/ppo/",
)

SPHERE_POLICY_NAMES = argparse.Namespace(
    grasp_pi="0422_sph_n_25_45",
    grasp_dir="./trained_models_0422_sph_n/ppo/",
    place_dir="./trained_models_0422_sph_n_place_0422_sph/ppo/",
)

SHAPE2POLICY_NAMES = {
    "universal": UNIVERSAL_POLICY_NAMES,
    "sphere": SPHERE_POLICY_NAMES,
}


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
