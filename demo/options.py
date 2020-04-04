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
    grasping_control_skip=6,
    n_plan_steps=200,
    n_grasp_steps=30,
    n_place_steps=50,
    n_release_steps=100,
    grasp_pi="0313_2_n_25_45",
    grasp_dir="./trained_models_0313_2_n/ppo/",
    grasp_env_name="InmoovHandGraspBulletEnv-v5",
    placing_control_skip=6,
    place_dir="./trained_models_0313_2_placeco_0316_1/ppo/",
    place_env_name="InmoovHandPlaceBulletEnv-v9",
    vision_delay=2,
)