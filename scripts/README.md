## Generating vision module datasets

Step 1. Generate states for planning and placing.

```
# First, generate states for various stages of manipulation. (ETA: XX:XX)
./scripts/states/planning_v003.sh
./scripts/states/stacking_v003_box.sh
./scripts/states/stacking_v003_cyl.sh

# Next, complete the placing states by randomly assigning values to attributes 
that are missing from the states. (ETA: XX:XX)
./scripts/states/complete_stacking_v003_box.sh

# Subsample and merge into a single set of states. (ETA: XX:XX)
./scripts/states/combine.sh
```

Step 2. Zip and transfer the states to the machine where you will be running 
Unity.

```
# Current:
scp -r sydney:~/datasets/delay_box_states workspace/lucas/states/

# Desired:
scp -r sydney:~/datasets/dash_v001/states workspace/lucas/states
```

Step 3. Generate Unity images from the states.

```
python scripts/server.py
```

Step 4. Zip up and scp the generated Unity images to the machine where training
will occur.

```
# Zip up the images (ETA: XX:XX)
time zip -r delay_box_states.zip delay_box_states

# Transfer the the images (ETA: XX:XX)
scp -r delay_box_states.zip sydney:~/datasets/dash_v001/
```

Step 5. Generate the dataset for training and testing.

```
./ns_vqa_dart/scripts/dash_v001/generate.sh
```

## Training and testing the vision module on datasets

To run training and testing on a tiny subset of the dataset for a few 
iterations as a dry run:

```
./ns_vqa_dart/scripts/dash_v001/dry_run.sh
```

To run training and testing on the full dataset:

```
./ns_vqa_dart/scripts/dash_v001/run.sh
```

## Generating result tables

### Table 1: Results on the full system

```
TODO
```

### Table 2: Results on stacking

```
# Ground truth
./scripts/table2/delay_box.sh
./scripts/table2/delay_cyl.sh

# Vision module
./scripts/table2/delay_vision_box.sh
./scripts/table2/delay_vision_cyl.sh

# Baseline
./scripts/table2/baseline_box.sh
./scripts/table2/baseline_cyl.sh
```

## NLP-related installation instructions

Mainly to run `main_sim_stack_new.py`:

```
pip install spacy nltk
python -m spacy download en_core_web_sm
```

## Instructions for Building & Running OpenRAVE:

First, build the image.

```
git clone git@github.com:jyf588/openrave-installation.git
cd openrave-installation
git checkout my_docker
cd nvidia-docker
sudo docker build -t openrave-ha:v0 .
cd ../openrave-docker
sudo docker build -t openrave-ha:v2 .
cd ../or-my-docker
sudo docker build -t openrave-ha:v3 .
```

Next, run the container of the built image:
(create a container with access to data from the host machine create a folder "container_data" in the home directory)

```
xhost +si:localuser:root
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data:/data --name openravecont openrave-ha:v3 /bin/bash
```

Troubleshooting
```
>>> docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
Solution: https://devtalk.nvidia.com/default/topic/1061452/docker-and-nvidia-docker/could-not-select-device-driver-quot-quot-with-capabilities-gpu-/

Inside the running container, see if you can use Firefox and openrave with GUI.
```
source bashrc
glxgears
firefox
```

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

To run OpenRAVE for DASH:

Outside of docker container:
```
cd ~/container_data
git clone https://github.com/jyf588/or_planning_scripts
```

Update `tabletop_2.kinbody.xml` to the following:

```
<KinBody name="box0">
  <Body name="base">
    <Geom type="box">
      <extents>1.3 0.6 0.05</extents>
      <translation>0.2 0.1 -0.13</translation>
      <diffusecolor>0.6 0 0</diffusecolor>
    </Geom>
  </Body>
</KinBody>
```

Inside of docker container:
```
source bashrc
cd /data/or_planning_scripts
python move_single.py
```

If you need to run reach_single.py also, open another session for the same 
container:
```
sudo docker exec -it openravecont /bin/bash
source bashrc
cd /data/or_planning_scripts
python reach_single.py
```
