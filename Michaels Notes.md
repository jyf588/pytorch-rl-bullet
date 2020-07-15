----------
New
------------

# Changes
- Left Arm (Check sticky note)

TODO:
CHECK UTILS.PY CONSTANTS

Changes:

main_sim_clean_test.py

Line 147:
Added:
def switchDirections(target_list):
183:
Added:
tar_arm_q = switchDirections(tar_arm_q)


Line 646: 
Added:
    for i in range(len(all_dicts)):
        all_dicts[i]["position”][1] *= -1

utils.py

Line 50:
Changed: (original for now)
# manipulable range
TX_MIN = -0.1
TX_MAX = 0.25
TY_MIN = -0.1
TY_MAX = 0.5
# whole table range
X_MIN = -0.1
X_MAX = 0.3
Y_MIN = -0.3
Y_MAX = 0.7
To:
# manipulable range
TX_MIN = -0.1
TX_MAX = 0.25
TY_MIN = -0.1
TY_MAX = 0.1
# whole table range
X_MIN = -0.1
X_MAX = 0.3
Y_MIN = -0.3
Y_MAX = 0.3


Line 61: 
Changed:
TABLE_OFFSET = [0.1, 0.2, 0.0]
To
TABLE_OFFSET = [0.1, -0.2, 0.0]

Inmoov_shadow_demo_env_v4.py
Line 1:
Change:
from .inmoov_shadow_hand_v2 import InmoovShadowNew
To:
from .inmoov_shadow_hand_v2_left import InmoovShadowNew

Added File inmoov_shadow_hand_v2_left
Copied inmoov_shadow_hand but loadUrdf points to leftmod mod instead of original
Changed:
Line 83:
From:
self.base_init_pos = np.array([-0.30, 0.348, 0.272])
To:
self.base_init_pos = np.array([-0.30, -0.348, 0.272])


# Joints

- 0-6 Rotate in same direction
- 7 on Rotate in opposite direction with 5 wrist roll joint yaw 3.141

(0, b'r_shoulder_out_joint', 0, 7, 6, 1, 0.0, 0.0, -2.07079632679, 1.57079632679, 1000.0, 1745.32925199, b'r_bicep_link_aux_0', (1.0, 0.0, 0.0), (0.0, -0.066, -0.06), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'r_shoulder_lift_joint', 0, 8, 7, 1, 0.0, 0.0, -2.07079632679, 1.57079632679, 1000.0, 1745.32925199, b'r_bicep_link_aux_1', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
(2, b'r_upper_arm_roll_joint', 0, 9, 8, 1, 0.0, 0.0, -1.57079632679, 1.57079632679, 1000.0, 1745.32925199, b'r_bicep_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1)
(3, b'r_elbow_flex_joint', 0, 10, 9, 1, 0.0, 0.0, -3.14159265359, 0.0, 1000.0, 1745.32925199, b'r_forearm_link_aux', (0.0, 1.0, 0.0), (-0.054, -0.009, -0.2621), (0.0, 0.0, 0.0, 1.0), 2)
(4, b'r_elbow_roll_joint', 0, 11, 10, 1, 0.0, 0.0, -1.57079632679, 1.57079632679, 1000.0, 1745.32925199, b'r_forearm_link', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 3)
(5, b'r_wrist_roll_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'r_hand_link', (0.0, 0.0, 0.0), (0.0303, -0.00729, -0.302), (1.0, 0.0, 0.0, 1.0341155355510722e-13), 4)
(6, b'rh_WRJ2', 0, 12, 11, 1, 0.1, 0.0, -1.0471975512, 1.0471975512, 10.0, 20.0, b'rh_wrist', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 5)
(7, b'rh_WRJ1', 0, 13, 12, 1, 0.1, 0.0, -1.57079632679, 1.57079632679, 30.0, 20.0, b'rh_palm', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0009999999999999974), (0.0, 0.0, 0.0, 1.0), 6)

- Axis multiplied by negative 1 8 and on

Index
(8, b'rh_FFJ4', 0, 14, 13, 1, 0.1, 0.0, -0.349065850399, 0.349065850399, 2.0, 2.0, b'rh_ffknuckle', (0.0, -1.0, 0.0), (0.033, 0.0, 0.06), (0.0, 0.0, 0.0, 1.0), 7)
(9, b'rh_FFJ3', 0, 15, 14, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_ffproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 8)
(10, b'rh_FFJ2', 0, 16, 15, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_ffmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 9)
(11, b'rh_FFJ1', 0, 17, 16, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_ffdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 10)
(12, b'rh_FFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rh_fftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013999999999999999), (0.0, 0.0, 0.0, 1.0), 11)

Middle
(13, b'rh_MFJ4', 0, 18, 17, 1, 0.1, 0.0, -0.349065850399, 0.349065850399, 2.0, 2.0, b'rh_mfknuckle', (0.0, -1.0, 0.0), (0.011, 0.0, 0.064), (0.0, 0.0, 0.0, 1.0), 7)
(14, b'rh_MFJ3', 0, 19, 18, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_mfproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 13)
(15, b'rh_MFJ2', 0, 20, 19, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_mfmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 14)
(16, b'rh_MFJ1', 0, 21, 20, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_mfdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 15)
(17, b'rh_MFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rh_mftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013999999999999999), (0.0, 0.0, 0.0, 1.0), 16)

Ring
(18, b'rh_RFJ4', 0, 22, 21, 1, 0.1, 0.0, -0.349065850399, 0.349065850399, 2.0, 2.0, b'rh_rfknuckle', (0.0, 1.0, 0.0), (-0.011, 0.0, 0.06), (0.0, 0.0, 0.0, 1.0), 7)
(19, b'rh_RFJ3', 0, 23, 22, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_rfproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 18)
(20, b'rh_RFJ2', 0, 24, 23, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_rfmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 19)
(21, b'rh_RFJ1', 0, 25, 24, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_rfdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 20)
(22, b'rh_RFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rh_rftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013999999999999999), (0.0, 0.0, 0.0, 1.0), 21)

(23, b'rh_LFJ5', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rh_lfmetacarpal', (0.0, 0.0, 0.0), (-0.033, 0.0, -0.014290000000000004), (0.0, 0.0, 0.0, 1.0), 7)

Pinky
(24, b'rh_LFJ4', 0, 26, 25, 1, 0.1, 0.0, -0.349065850399, 0.349065850399, 2.0, 2.0, b'rh_lfknuckle', (0.0, 1.0, 0.0), (0.0, 0.0, 0.02579), (0.0, 0.0, 0.0, 1.0), 23)
(25, b'rh_LFJ3', 0, 27, 26, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_lfproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 24)
(26, b'rh_LFJ2', 0, 28, 27, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_lfmiddle', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 25)
(27, b'rh_LFJ1', 0, 29, 28, 1, 0.1, 0.0, 0.0, 1.57079632679, 2.0, 2.0, b'rh_lfdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0, 1.0), 26)
(28, b'rh_LFtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rh_lftip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.013999999999999999), (0.0, 0.0, 0.0, 1.0), 27)

Thumb
(29, b'rh_THJ5', 0, 30, 29, 1, 0.2, 0.0, -1.0471975512, 1.0471975512, 5.0, 4.0, b'rh_thbase', (0.0, 0.0, -1.0), (0.034, -0.0085, -0.006000000000000002), (0.0, -0.38268343236488267, 0.0, 0.9238795325113726), 7)
(30, b'rh_THJ4', 0, 31, 30, 1, 0.2, 0.0, 0.0, 1.2217304764, 3.0, 4.0, b'rh_thproximal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 29)
(31, b'rh_THJ3', 0, 32, 31, 1, 0.2, 0.0, -0.209439510239, 0.209439510239, 2.0, 4.0, b'rh_thhub', (1.0, 0.0, 0.0), (0.0, 0.0, 0.019), (0.0, 0.0, 0.0, 1.0), 30)
(32, b'rh_THJ2', 0, 33, 32, 1, 0.1, 0.0, -0.698131700798, 0.6917304764, 2.0, 2.0, b'rh_thmiddle', (0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 31)
(33, b'rh_THJ1', 0, 34, 33, 1, 0.2, 0.0, 0.0, 1.57079632679, 1.0, 4.0, b'rh_thdistal', (1.0, 0.0, 0.0), (0.0, 0.0, 0.016), (0.0, 0.0, 0.7071067811848163, 0.7071067811882787), 32)
(34, b'rh_thtip', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rh_thtip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.01375), (0.0, 0.0, 0.0, 1.0), 33)

Creating catkin ws:
https://www.youtube.com/watch?v=JE6QmshipyQ

# 14.04
cmake fails with “CMake Error: your CXX compiler: ”CMAKE_CXX_COMPILER-NOTFOUND“ was not found.”: 
https://askubuntu.com/questions/152653/cmake-fails-with-cmake-error-your-cxx-compiler-cmake-cxx-compiler-notfound

# gcp
ssh-keygen -t rsa -f ./ssh_keys/key -C mikephayashi
ssh -i ~/Desktop/key mikephaysh@35.233.191.126
scp -i ~/Desktop/key ./text.txt mikephayashi@35.233.191.126:~/dash
source /opt/ros/kinetic/setup.bash (http://wiki.ros.org/rosbash)
rosrun xacro xacro --inorder -o tx.urdf tx.xacro (https://answers.ros.org/question/10401/how-to-convert-xacro-file-to-urdf-file/)

# xacro -> urdf:

scp -i ./key /Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2.urdf.xacro mikephayashi@35.233.191.126:~/catkin_ws/src/inmoov_description/robots/

rosrun xacro xacro --inorder -o inmoov_shadow_hand_v2_2.urdf ./src/inmoov_description/robots/inmoov_shadow_hand_v2.urdf.xacro

scp -i ./key mikephayashi@35.233.191.126:~/catkin_ws/inmoov_shadow_hand_v2_2.urdf /Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf


# TODO 

#systems/options.py
Change render_unity=False to true

# urdf
- /Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/inmoov_shadow_hand_v2.py
- /Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf


#### Running

Clear files from container_data:
Traceback (most recent call last):
  File "main_sim_clean_test.py", line 616, in <module>
    OBJECTS, np.array([0.0] * 7), Qreach, reach_save_path, reach_read_path
  File "/home/mikehaya/Desktop/pytorch-rl-bullet/system/openrave.py", line 163, in get_traj_from_openrave_container
    assert not os.path.exists(load_path)
AssertionError


Grasping:
- python enjoy.py --env-name InmoovHandGraspBulletEnv-v6 --load-dir ~/Desktop/pytorch-rl-bullet/trained_models_0510_0_n/ppo/ --non-det 0 --seed=18991 --renders 1 --random_top_shape 1 --obs_noise 1 --n_best_cand 2 --warm_start_phase 0 --has_test_phase 1 --save_final_states 0 --r_thres 1800 --use_obj_heights 1 --cotrain_onstack_grasp 0 --grasp_floor 1

Whole Module without language and vision:
- python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 0 --add_place_stack_bit 0
- python main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 1 --add_place_stack_bit 1 --render 1 --sleep 1


#### Openrave

xhost +si:localuser:mikehaya


sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/mikehaya/container_data:/data --name openravecont openrave-ha:v3 /bin/bash

sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/mikehaya/container_data:/data --name openravecont2 openrave-ha:v3 /bin/bash

sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/mikehaya/container_data:/data --name openravecont3 openrave-ha:v3 /bin/bash




#### Detectron
https://detectron2.readthedocs.io/tutorials/install.html
Used:
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html

#### Conda
https://stackoverflow.com/questions/35246386/conda-command-not-found
1. bash
2. conda

------------
Local Mac
------------

#### Installation

check submodules

Did not install:
cudatoolkit=10.2

No GPU

Detectron on mac:
https://github.com/pytorch/pytorch/issues/16805
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++

Open rave:
Conainer 1:
real    30m23.077s
user    0m1.287s
sys     0m1.126s
Container 2: 
real    10m26.819s
user    0m0.561s
sys     0m0.575s
Container 3:
real    0m45.626s
user    0m0.257s
sys     0m0.317s

#### Problems

Starting containers
https://www.youtube.com/watch?v=CCAK7GhmiEM
https://github.com/moby/moby/issues/8710 slobo commented on Jan 22, 2015

---

sudo docker run -ti --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data:/data --name openravecont openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data:/data --name openravecont2 openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data:/data --name openravecont3 openrave-ha:v3 /bin/bash

----

socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"

sudo docker run -ti --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/Documents/CURIS/data:/data --name openravecont openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/Documents/CURIS/data:/data --name openravecont2 openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/Documents/CURIS/data:/data --name openravecont3 openrave-ha:v3 /bin/bash

----

Make sure "exp" is not installed for python

Changed testmanip_0510.sh "python" to "python3"

Changed IS_CUDA = True to False in main_sim_clean_test.py:89

main_sim_clean_test.py
Changed GRASP_PI = "0510_0_n_25_45" to 1226_from_fixed

#### TODO: 
(dash) pytorch-rl-bullet $python3 main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 1 --add_place_stack_bit 1
pybullet build time: Jun 25 2020 11:31:09
| loading policy from ./trained_models_0510_0_n/ppo/InmoovHandGraspBulletEnv-v6.pt
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/serialization.py:593: SourceChangeWarning: source code of class 'torch.nn.modules.container.Sequential' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/serialization.py:593: SourceChangeWarning: source code of class 'torch.nn.modules.linear.Linear' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/serialization.py:593: SourceChangeWarning: source code of class 'torch.nn.modules.activation.Tanh' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
| loading policy from ./trained_models_0510_0_n_place_0510_0/ppo/InmoovHandPlaceBulletEnv-v9.pt
Qreach: [-0.8884957785960691, 0.031125633539708274, -0.6022961778257007, -1.2006211693263578, -0.719912363910822, -0.1676889542046939, 0.13964397212735585]
OR trajectory for /Users/michaelhayashi/container_data/OR_REACH.npz: (400, 7)
init_tar_fin_q
['0.415', '0.412', '0.416', '0.419', '0.401', '0.419', '0.414', '0.405', '0.387', '0.383', '0.398', '0.387', '0.018', '1.009', '0.084', '0.519', '0.110']
init_fin_q
['0.415', '0.412', '0.416', '0.419', '0.401', '0.419', '0.414', '0.405', '0.387', '0.383', '0.398', '0.387', '0.018', '1.009', '0.084', '0.519', '0.110']
diff final 0.0010071153864928833
vel final 0.026847934087620023
fin dofs
['0.415', '0.412', '0.416', '0.419', '0.401', '0.419', '0.414', '0.405', '0.387', '0.383', '0.398', '0.387', '0.018', '1.009', '0.084', '0.519', '0.110']
cur_fin_tar_q
['0.415', '0.412', '0.416', '0.419', '0.401', '0.419', '0.414', '0.405', '0.387', '0.383', '0.398', '0.387', '0.018', '1.009', '0.084', '0.519', '0.110']
arm q [-0.88800607  0.03139917 -0.60194858 -1.2008238  -0.71963003 -0.16771707
  0.13912958]
Traceback (most recent call last):
  File "main_sim_clean_test.py", line 654, in <module>
    g_obs, recurrent_hidden_states, masks, deterministic=args.det
  File "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/a2c_ppo_acktr/model.py", line 55, in act
    value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
  File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/Users/michaelhayashi/Documents/CURIS/pytorch-rl-bullet/a2c_ppo_acktr/model.py", line 226, in forward
    hidden_critic = self.critic(x)
  File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/nn/modules/container.py", line 100, in forward
    input = module(input)
  File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/torch/nn/functional.py", line 1370, in linear
    ret = torch.addmm(bias, input, weight.t())
RuntimeError: size mismatch, m1: [1 x 135], m2: [132 x 64] at ../aten/src/TH/generic/THTensorMath.cpp:136



python3 main_sim_clean_test.py --seed 1099 --test_placing 0 --long_move 1 --use_height 0 --add_place_stack_bit 0