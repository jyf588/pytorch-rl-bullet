import argparse

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
    disable_reaching=True,
    grasping_control_skip=6,
    n_plan_steps=250,
    n_grasp_steps=30,
    n_place_steps=50,
    n_release_steps=100,
    grasp_pi="0404_0_n_20_40",
    grasp_dir="./trained_models_0404_0_n/ppo/",
    grasp_env_name="InmoovHandGraspBulletEnv-v6",
    placing_control_skip=6,
    place_dir="./trained_models_0404_0_n_place_0404_0/ppo/",
    place_env_name="InmoovHandPlaceBulletEnv-v9",
    vision_delay=2,
    vision_checkpoint_path="/home/michelle/mguo/outputs/dash_v005_20K/checkpoint_best.pt",
    coordinate_frame="unity_camera",
)
