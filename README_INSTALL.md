## Requirements

This section contains prerequisites to running the DASH system.

First, clone the main repo and its submodules. (modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and https://github.com/openai/baselines)
```
git clone https://github.com/jyf588/pytorch-rl-bullet.git
git submodule update --init --recursive
```

Next, create a conda environment (make sure Python >=3.6)
```
conda create -n dash python=3.6
conda activate dash
```

Install python packages.
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

OpenAI baselines:
```
cd baselines
pip install -e .
```

Pytorch installation (>=1.4 required). We use CUDA 10.2 but feel free to change this.
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Check that PyTorch is properly installed:
```
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
print(torch.version.cuda)
torch.zeros(2).cuda()
```

Make sure that your torch CUDA version printed out matches your CUDA version.
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

Check the gcc version. gcc & g++ ≥ 5 are required.
```
gcc --version
```

Install Detectron2 and pycocotools:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Try importing detectron2:
```
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
```

## Instructions for Building & Running OpenRAVE:

### Prerequisites
1. Docker (https://docs.docker.com/engine/install/ubuntu/)
2. `nvidia-container-toolkit` (https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster)

### Building docker images

First, build the images. This can take a few minutes.

```
git clone git@github.com:jyf588/openrave-installation.git
cd openrave-installation
git checkout my_docker

cd nvidia-docker
time sudo docker build -t openrave-ha:v0 .

cd ../openrave-docker
time sudo docker build -t openrave-ha:v2 .

cd ../or-my-docker
time sudo docker build -t openrave-ha:v3 .  # (ETA: 30 seconds)
```

## Running OpenRAVE

On the host machine, create a folder which we will grant container access to.
```
mkdir <path_to_container_data_dir>
```

On the host machine, clone the repository containing OpenRAVE scripts we will run into the `container_data` folder we just created.
```
cd <path_to_container_data_dir>
git clone git@github.com:jyf588/or_planning_scripts.git
```

Start three separate docker containers, each in separate terminal sessions, with the following commands:
```
xhost +si:localuser:root

# Start the first container (--name is not necessary)
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <path_to_container_data_dir>:/data --name openravecont openrave-ha:v3 /bin/bash

# Start the second container
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <path_to_container_data_dir>:/data --name openravecont2 openrave-ha:v3 /bin/bash

# Start the thrid container
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <path_to_container_data_dir>:/data --name openravecont3 openrave-ha:v3 /bin/bash

```

Inside the running container, see if you can use Firefox and glxgears with GUI.
```
glxgears
firefox
```

Then, in each of the three containers, run the following commands:
```
source bashrc
cd /data/or_planning_scripts
```

Finally, run the following commands in each of the three containers:
```
python move_single.py 0 l
python move_single.py 1 l
python move_single.py 2 l
```
These correspond to reach, move and retract, respectively.


### Troubleshooting
If you installed cuda 9.1 before and saw errors about `nvidia-cuda-dev` when installing `nvidia-container-toolkit`:
```
sudo apt-get --purge remove nvidia-cuda-dev
sudo apt autoremove
```
If this error:
```
>>> docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
Check if this is done properly: https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster

If you need to rebuild an image, run the following command (replace `v3` with the desired tag).
```
sudo docker build --no-cache -t openrave-ha:v3 .
```
And make sure to restart the containers after rebuilding.


To run openrave examples: (https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(openrave-config --python-dir)/openravepy/_openravepy_
export PYTHONPATH=$PYTHONPATH:$(openrave-config --python-dir)
openrave.py --example hanoi
```

Other useful docker commands (with sudo): 

```
docker ps --all (list containers)
docker rm -f contrainer_number (kill and remove container)
docker image list
docker image rm image_number
docker cp <src_path> openravecont:<dst_path>
```
