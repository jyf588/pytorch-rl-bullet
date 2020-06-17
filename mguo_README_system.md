# DASH

Before running the following commands, follow the instructions located at 
`mguo_INSTALL.md` to ensure that you have the necessary prerequisites installed and 
setup.

Before running the below commands, run the following from the repo root:
```
conda activate dash
export PYTHONPATH=.
```

## System

### Setup

Before running the system, make sure that OpenRAVE is running. See `mguo_INSTALL.md` for
instructions on doing so.

#### Download the DASH Unity client application.

Next, download the Unity client application from the Google Drive Link
[here](https://drive.google.com/open?id=1kTZubGbp-q_axyPG980J2uTggsYKITXT).
Unzip and place in `<home_dir>/unity`.

#### Download pretrained models, or train your own.

You can download the pretrained policies [here](https://drive.google.com/open?id=1XNy0GPO1U78VWjUY6Gl18yLNROcA12HQ), and pretrained vision models can be downloaded
[here](https://drive.google.com/open?id=1Jcta_Ye5wG8l1traU4e414lHKOtjtssW).

Unzip and store the policy folders in the main root of the repo. Unzip and store the
vision models to `<home_dir>/outputs/0518_results`.

Feel free to change various paths for running the system by modifying the paths 
specified in `system/options.py`.

### Running the demo

Start the python server, and start up the Unity client application in a separate command
line:
```
python system/run_demo.py <exp> <option_name>
./Linux8000.x86_64
```

For example, to run the demo without vision (ground truth), run the following command:
```
python system/run_demo.py demo gt_demo
```

To run the vision demo, run the following command:
```
python system/run_demo.py demo vision_demo
```

### Running Table 1 experiments

Similar to running the demo, first start python server (note that we are running 
`run.py` instead of `run_demo.py` here), and run the Unity executable in a separate 
command line:
```
python system/run.py <exp> <option_name>
./Linux8000.x86_64
```

For instance, to run the experiments without vision (ground truth), run the following
command line arguments:
```
python system/run.py t1 test_gt
```

To run with vision, run the following:
```
python system/run.py t1 test_vision
```

### Running without Unity / headless mode

You can run the system without vision (ground truth) in headless mode, without running
Unity / a graphical display, by spinning up a dummy client with
the following command:
```
python system/client.py
```

Currently, the system with vision is only supported by running the graphical display of
the Unity application, in order to properly render the Unity images that are input to 
the vision module. If graphical displays do not work on your machine, you can 
alternatively run the Unity application from a different machine that does have a 
graphical display. Simply update the hostname IP address and port that the python server
and Unity client use.

### Generating your own scenes

You can generate your own scenes by running
```
python scene/generate.py
```

You can also manually define your own scene dataset by following the same folder 
structure as found in `system/data` datasets.

### Notes on running the system

If specified by the set of options associated with `option_name`, Unity images will be
saved to a folder named `Captures`, in the same directory that the Unity executable lies
in.

Here are explanations for the command line arguments for running the system:
1. `exp`: The name of the experiment dataset containing scene JSONs to run.
2. `option_name`: The name of the set of options you'd like to use. See `SYSTEM_OPTIONS`
 in `system/options.py` for a list of available options.

Occasionally if the system is interrupted in the middle of its execution, you may find
unprocessed `.npz` files in the OpenRAVE directory that are leftover from the execution.
Delete these files before restarting the system.

## Internal Notes

System datasets:
- `system/data/demo` is a copy of `~/data/dash/demo_z0_v2`
- `system/data/t1` is a copy of `~/data/dash/t1`
