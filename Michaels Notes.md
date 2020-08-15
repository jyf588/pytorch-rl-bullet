----------
New
------------


# Running

- python3 main_sim_clean_test_2_arms.py --seed 1099 --test_placing 0 --long_move 1 --use_height 1 --add_place_stack_bit 1 --render 1 --sleep 1

# Openrave
(My mac)

Left:

sudo docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data_left:/data --name openravecont1 openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data_left:/data --name openravecont2 openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data_left:/data --name openravecont3 openrave-ha:v3 /bin/bash

Right:

sudo docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data:/data --name openravecont4 openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data:/data --name openravecont5 openrave-ha:v3 /bin/bash

sudo docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /Users/michaelhayashi/container_data:/data --name openravecont6 openrave-ha:v3 /bin/bash

Sydney:
xhost +si:localuser:root

sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data:/data --name openravecont openrave-ha:v3 /bin/bash

sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data:/data --name openravecont2 openrave-ha:v3 /bin/bash

sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data:/data --name openravecont3 openrave-ha:v3 /bin/bash

python move_single.py 0 l
python move_single.py 1 l
python move_single.py 2 l

# Chrome remote desktop
Use xfce on sydney, everything else doesn't work

# Dual boot linux
If wifi doesn't work, delete it then add it again

# Refind
Restart to boot refind


# Catkin

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
