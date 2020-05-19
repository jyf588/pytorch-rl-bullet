import os
import copy
import argparse
import ns_vqa_dart.bullet.util as util


homedir = util.get_user_homedir()


# PORT = 8000
# CONTAINER_DIR = "/home/mguo/container_data_8000"
# CONTAINER_DIR = "/home/mguo/container_data_8001"
# CONTAINER_DIR = "/home/mguo/container_data_8002"
# CONTAINER_DIR = f"/home/mguo/container_data_{PORT}"
# CONTAINER_DIR = "/home/mguo/container_data_8010"
# CONTAINER_DIR = "/home/mguo/container_data_8011"

# UNITY_NAME = "Linux8000_0512"
# UNITY_NAME = "Linux8001_0515"
# UNITY_NAME = "Linux8002_0515"
# UNITY_NAME = f"Linux{PORT}_0515"
# UNITY_CAPTURES_DIR = os.path.join(homedir, f"unity/builds/{UNITY_NAME}/Captures")

# UNITY_CAPTURES_DIR = None


def set_unity_container_cfgs(opt, port):
    unity_name = f"Linux{port}_0515"
    unity_captures_dir = os.path.join(homedir, f"unity/builds/{unity_name}/Captures")
    opt.container_dir = f"/home/mguo/container_data_{port}"
    opt.unity_captures_dir = unity_captures_dir
    return opt


BASE_SYSTEM_OPTIONS = argparse.Namespace(
    seed=101,
    is_cuda=True,
    enable_reaching=True,
    enable_retract=True,
    scene_place_src_idx=0,
    scene_place_dst_idx=1,
    scene_stack_src_idx=0,
    scene_stack_dst_idx=1,
    start_sid=None,  # Inclusive
    end_sid=None,  # Exclusive
    task_subset=None,
    obs_mode=None,
    obs_noise=False,
    position_noise=0.03,
    upv_noise=0.04,
    height_noise=0.02,
    render_unity=False,
    render_bullet=False,
    visualize_unity=False,
    cam_version="v2",
    use_control_skip=True,
    render_frequency=100,
    render_obs=False,
    animate_head=True,
    save_states=True,  # To check reproducibility.
    policy_id="0411",  # [0404, 0411, 0510]
    save_first_pov_image=False,
    scenes_root_dir=os.path.join(util.get_user_homedir(), "data/dash"),
    root_outputs_dir=os.path.join(util.get_user_homedir(), "outputs/system"),
    container_dir=None,
    unity_captures_dir=None,
    two_commands=False,
)

VISION_STATES_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
VISION_STATES_OPTIONS.enable_reaching = False
VISION_STATES_OPTIONS.enable_retract = False
VISION_STATES_OPTIONS.obs_mode = "gt"
VISION_STATES_OPTIONS.obs_noise = False

UNITY_DATASET_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
UNITY_DATASET_OPTIONS.render_frequency = 1  # Render and save every state.
UNITY_DATASET_OPTIONS.render_unity = True

TEST_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
TEST_OPTIONS.enable_reaching = True
TEST_OPTIONS.enable_retract = False  # Retract is excluded from Table 1 evaluation.

TEST_GT_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_GT_OPTIONS.obs_mode = "gt"

DEMO_OPTIONS = copy.deepcopy(BASE_SYSTEM_OPTIONS)
DEMO_OPTIONS.enable_reaching = True
DEMO_OPTIONS.enable_retract = True
DEMO_OPTIONS.render_unity = True
DEMO_OPTIONS.visualize_unity = True
DEMO_OPTIONS.render_bullet = True
DEMO_OPTIONS = copy.deepcopy(DEMO_OPTIONS)
DEMO_OPTIONS.obs_mode = "gt"  # TODO
# DEMO_OPTIONS.start_sid = 1
# DEMO_OPTIONS.end_sid = 2

TEST_VISION_OPTIONS = copy.deepcopy(TEST_OPTIONS)
TEST_VISION_OPTIONS.obs_mode = "vision"
TEST_VISION_OPTIONS.render_unity = True
TEST_VISION_OPTIONS.render_obs = True
TEST_VISION_OPTIONS.save_first_pov_image = True

DEBUG_VISION_OPTIONS = copy.deepcopy(TEST_OPTIONS)
DEBUG_VISION_OPTIONS.obs_mode = "vision"
DEBUG_VISION_OPTIONS.render_unity = True
DEBUG_VISION_OPTIONS.render_obs = True
DEBUG_VISION_OPTIONS.save_first_pov_image = True
DEBUG_VISION_OPTIONS.enable_reaching = True
DEBUG_VISION_OPTIONS.enable_retract = False
DEBUG_VISION_OPTIONS.start_sid = 31
DEBUG_VISION_OPTIONS.end_sid = 32
DEBUG_VISION_OPTIONS.task_subset = ["stack"]


SYSTEM_OPTIONS = {
    "vision_states": VISION_STATES_OPTIONS,
    "unity_dataset": UNITY_DATASET_OPTIONS,
    "test_vision": TEST_VISION_OPTIONS,
    "test_gt": TEST_GT_OPTIONS,
    "debug_vision": DEBUG_VISION_OPTIONS,
    "demo": DEMO_OPTIONS,
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
    use_height=False,  # assume sph policy does not use height or one bit.
    use_place_stack_bit=False,
    use_slow_policy=False,
    n_reach_steps=405,
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


def get_policy_options_and_paths(policy_id: str):
    policy_options = copy.deepcopy(POLICY_OPTIONS)
    if policy_id == "0404":
        policy_options.use_height = True
        grasp_pi = f"0404_0_n_20_40"
        grasp_dir = f"./trained_models_%s/ppo/" % "0404_0_n"
        place_pi = f"0404_0_n_place_0404_0"
        place_dir = f"./trained_models_%s/ppo/" % place_pi
    elif policy_id == "0411":
        grasp_pi = f"0411_0_n_25_45"
        grasp_dir = f"./trained_models_0411_0_n/ppo/"
        place_dir = f"./trained_models_0411_0_n_place_0411_0/ppo/"
    elif policy_id == "0510":
        policy_options.use_height = True
        policy_options.use_place_stack_bit = True
        grasp_pi = f"0510_0_n_25_45"
        grasp_dir = f"./trained_models_%s/ppo/" % "0510_0_n"
        place_pi = f"0510_0_n_place_0510_0"
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


# VISION_V1_MODELS = {
#     "plan": "planning_v003_20K/2020_04_19_07_14_00",
#     "place": "placing_v003_2K_20K/2020_04_22_04_35",
#     "stack": "stacking_v003_2K_20K/2020_04_19_22_12_00",
# }

# VISION_V2_MODELS = {  # Reproduce v1 with re-rendered unity images.
#     "plan": "planning_v003_20K/2020_05_15_23_43_08",
#     "place": "placing_v003_2K_20K/2020_05_15_02_18_08",
#     "stack": "stacking_v003_2K_20K/2020_05_16_02_36_30",
# }

# VISION_V3_MODELS = {  # With v2 camera.
#     "plan": "plan_20K_0518/2020_05_18_03_19_01",
#     "place": "place_2K_20K_0518/2020_05_18_02_39_29",
#     "stack": "stack_2K_20K_0518/2020_05_18_02_50_47",
# }


# plan_model = VISION_V2_MODELS["plan"]
# place_model = VISION_V3_MODELS["place"]
# stack_model = VISION_V3_MODELS["stack"]

# MASK_RCNN_MODEL = "2020_04_27_20_12_14/model_final.pth"
# MASK_RCNN_MODEL = "2020_05_16_22_49_31/model_0293999.pth"  # Rerendered Unity images.

VISION_MODELS_DIR = os.path.join(homedir, "outputs/0518_results")

VISION_OPTIONS = argparse.Namespace(
    seed=None,
    renderer="unity",
    use_segmentation_module=True,
    separate_vision_modules=True,
    use_gt_obs=False,
    seg_checkpoint_path=os.path.join(VISION_MODELS_DIR, "mask_rcnn.pth"),
    planning_checkpoint_path=os.path.join(VISION_MODELS_DIR, "plan.pth"),
    placing_checkpoint_path=os.path.join(VISION_MODELS_DIR, "place.pth"),
    stacking_checkpoint_path=os.path.join(VISION_MODELS_DIR, "stack.pth"),
    coordinate_frame="unity_camera",
    save_predictions=True,
)


def print_and_save_options(
    run_dir: str, system_opt, bullet_opt, policy_opt, vision_opt
):
    options_dir = os.path.join(run_dir, "options")
    os.makedirs(options_dir)
    name2opt = {
        "system": system_opt,
        "bullet": bullet_opt,
        "policy": policy_opt,
        "vision": vision_opt,
    }
    for name, opt in name2opt.items():
        args = vars(opt)
        print("| options")
        for k, v in args.items():
            print("%s: %s" % (str(k), str(v)))

        filename = f"{name}.txt"
        file_path = os.path.join(options_dir, filename)
        with open(file_path, "wt") as fout:
            fout.write("| options\n")
            for k, v in sorted(args.items()):
                fout.write("%s: %s\n" % (str(k), str(v)))
