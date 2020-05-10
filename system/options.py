import copy
import argparse

USE_HEIGHT = False

# Models with height:
if USE_HEIGHT:
    GRASP_PI = "0404_0_n_20_40"
    GRASP_DIR = "./trained_models_0404_0_n/ppo/"
    PLACE_DIR = "./trained_models_0404_0_n_place_0404_0/ppo/"
else:
    GRASP_PI = "0411_0_n_25_45"
    GRASP_DIR = "./trained_models_0411_0_n/ppo/"
    PLACE_DIR = "./trained_models_0411_0_n_place_0411_0/ppo/"

# Baseline model, w/o height:
# PLACE_DIR = "./trained_models_0411_0_n_place_0411_0_np_0/ppo/"

BASE_SYSTEM_OPTIONS = argparse.Namespace(
    is_cuda=True,
    enable_reaching=None,
    enable_retract=None,
    gt_place_idx=0,  # The index of the placing object in the GT scene.
    obs_mode=None,
    obs_noise=None,
    render_unity=None,
    render_bullet=False,
    use_control_skip=None,
    render_frequency=None,
    render_obs=None,
    animate_head=None,
)
VISION_DATASET_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
VISION_DATASET_OPTIONS.enable_reaching = False
VISION_DATASET_OPTIONS.enable_retract = False
VISION_DATASET_OPTIONS.render_unity = False
VISION_DATASET_OPTIONS.use_control_skip = True
VISION_DATASET_OPTIONS.obs_mode = "gt"
VISION_DATASET_OPTIONS.obs_noise = True

SYSTEM_OPTIONS = {
    "vision_tiny": VISION_DATASET_OPTIONS,
    "vision": VISION_DATASET_OPTIONS,
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
    use_height=USE_HEIGHT,
    n_plan_steps=305,
    grasp_control_steps=35,
    place_control_steps=75,
    control_skip=6,
    grasp_pi=GRASP_PI,
    grasp_dir=GRASP_DIR,
    place_dir=PLACE_DIR,
    grasp_env_name="InmoovHandGraspBulletEnv-v6",
    place_env_name="InmoovHandPlaceBulletEnv-v9",
    vision_delay=2,
)

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
