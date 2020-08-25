# Left Arm / Ambidextrous Instructions

**Left arm URDF**  = `./my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/left_arm.urdf`

* In order to use left arm for run_demo.py, change ./system/options.py:71 from `policy_id="0510"` to `policy_id="0729",`

**Left Arm Relevant Files**:
- `./main_sim_clean_test_left.py`: 
  - `my_pybullet_envs/inmoov_shadow_hand_v2_left.py`
- `./main_sim_clean_test_left_no_orientation.py`
  - `my_pybullet_envs/inmoov_shadow_demo_env_v4_left_no_orientation.py`

**Ambidextrous Arm Relevant Files** - Ambidextrous control, calculates midpoint between grasping/placing then uses right/left arm:
- `./main_sim_clean_test_2_arms.py`
  - `my_pybullet_envs/inmoov_shadow_demo_env_v4_2_arms.py`
  - `my_pybullet_envs/inmoov_shadow_hand_v2_2_arms.py`
  - `./util_2_arms.py`
- `./run_demo_voice_2_arms.py`
  - `./env_voice_2_arms.py`
- `./system/data/demo_2_arms` (Folders of scenes for run_demo_voice_2_arms.py)
  - `right_side.json`: Green cylinder, yellow box, red cylinder, and blue box on the right side.
  - `both_side.json`: Green cylinder and yellow box on right side, red cylinder and blue box on the left side.
- ``

**Other added files** - Using updated policy that does not require orientation:
- `./main_sim_clean_test_no_orientation.py`
  - `my_pybullet_envs/inmoov_shadow_demo_env_v4_no_orientation.py`  
  - `my_pybullet_envs/inmoov_shadow_place_env_no_orientation_v0.py`
- `./run_demo_voice.py`
  - `./env_voice.py`

**Utils**:
- `./flip_r_to_l.py`: Takes the "right" side of an arm (URDF) and outputs the corresponding left side. Xacro is outdated. Use this instead. 