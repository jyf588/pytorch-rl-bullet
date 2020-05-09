import argparse

GRASP_CONTROL_STEPS = 35
PLACE_CONTROL_STEPS = 75
CONTROL_SKIP = 6
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

OPTIONS = argparse.Namespace(
    seed=101,
    det=True,
    is_cuda=True,
    bullet_contact_iter=200,
    det_contact=0,
    init_noise=True,
    ts=1.0 / 240,
    obj_mu=1.0,
    hand_mu=1.0,
    floor_mu=1.0,
    init_pose=False,
    enable_reaching=False,
    enable_retract=False,
    restore_fingers=True,
    use_height=USE_HEIGHT,
    n_plan_steps=305,
    gt_place_idx=0,  # The index of the placing object in the GT scene.
    grasp_control_steps=GRASP_CONTROL_STEPS,
    place_control_steps=PLACE_CONTROL_STEPS,
    control_skip=CONTROL_SKIP,
    grasp_pi=GRASP_PI,
    grasp_dir=GRASP_DIR,
    place_dir=PLACE_DIR,
    grasp_env_name="InmoovHandGraspBulletEnv-v6",
    place_env_name="InmoovHandPlaceBulletEnv-v9",
    vision_delay=2,
    seg_checkpoint_path="/home/mguo/outputs/detectron/2020_04_27_20_12_14/model_final.pth",
    planning_checkpoint_path="/home/mguo/outputs/planning_v003_20K/checkpoint_best.pt",
    placing_checkpoint_path="/home/mguo/outputs/placing_v003_2K_20K/checkpoint_best.pt",
    stacking_checkpoint_path="/home/mguo/outputs/stacking_v003_2K_20K/checkpoint_best.pt",
    use_segmentation_module=True,
    coordinate_frame="unity_camera",
    save_predictions=True,
    debug_dir=None,
)
