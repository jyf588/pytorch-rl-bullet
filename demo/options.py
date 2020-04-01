import argparse

OPTIONS = argparse.Namespace(
    seed=101,
    det=True,
    is_cuda=True,
    bullet_contact_iter=200,
    det_contact=0,
    init_noise=True,
    ts=1.0 / 240,
    hand_mu=1.0,
    floor_mu=1.0,
    grasping_control_skip=6,
    n_grasp_steps=30,
    grasp_pi="0313_2_n_25_45",
    grasp_dir="./trained_models_0313_2_n/ppo/",
    grasp_env_name="InmoovHandGraspBulletEnv-v5",
)
