import copy
import argparse


BASE_SYSTEM_OPTIONS = argparse.Namespace(
    seed=101,
    is_cuda=True,
    enable_reaching=None,
    enable_retract=None,
    scene_place_src_idx=0,
    scene_place_dst_idx=1,
    scene_stack_src_idx=0,
    scene_stack_dst_idx=1,
    obs_mode=None,
    obs_noise=None,
    position_noise=0.03,
    upv_noise=0.04,
    height_noise=0.02,
    render_unity=None,
    render_bullet=False,
    use_control_skip=None,
    render_frequency=None,
    render_obs=None,
    animate_head=None,
    save_states=None,
)
VISION_STATES_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
VISION_STATES_OPTIONS.enable_reaching = False
VISION_STATES_OPTIONS.enable_retract = False
VISION_STATES_OPTIONS.render_unity = False
VISION_STATES_OPTIONS.use_control_skip = True
VISION_STATES_OPTIONS.obs_mode = "gt"
VISION_STATES_OPTIONS.obs_noise = True
VISION_STATES_OPTIONS.save_states = True

UNITY_DATASET_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
UNITY_DATASET_OPTIONS.render_unity = True
UNITY_DATASET_OPTIONS.render_frequency = 1
UNITY_DATASET_OPTIONS.animate_head = 1


SYSTEM_OPTIONS = {
    "vision_states": VISION_STATES_OPTIONS,
    "unity_dataset": UNITY_DATASET_OPTIONS,
    "test": None,
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
    use_segmentation_module=True,
    seg_checkpoint_path="/home/mguo/outputs/detectron/2020_04_27_20_12_14/model_final.pth",
    planning_checkpoint_path="/home/mguo/outputs/planning_v003_20K/checkpoint_best.pt",
    placing_checkpoint_path="/home/mguo/outputs/placing_v003_2K_20K/checkpoint_best.pt",
    stacking_checkpoint_path="/home/mguo/outputs/stacking_v003_2K_20K/checkpoint_best.pt",
    coordinate_frame="unity_camera",
    save_predictions=True,
    debug_dir=None,
)
