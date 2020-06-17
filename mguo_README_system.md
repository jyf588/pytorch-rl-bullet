# DASH

TODO: Add Unity executable and scene dataset to github? Latest working one is Builds0519b/Linux8000
TODO: figure out how to run unity executable from command line
TODO: Figure out how to specify which demo to run
TODO: Instructions on either manually defining or auto generating scene JSONs.
TODO: Original Unity project files in case you want to change Unity code.

Before running the following commands, follow the instructions located at 
`mguo_INSTALL.md` to ensure that you have the necessary prerequisites installed and 
setup.

Before running the below commands, run the following from the repo root:
```
conda activate dash
export PYTHONPATH=.
```

## System

Before running the system, make sure that OpenRAVE is running. See `mguo_INSTALL.md` for
instructions on doing so.

### Running the demo

Start the python server, and run the Unity executable in a separate command line:
```
python system/run_demo.py <exp> <option_name> <port>
./Linux8000.x86_64
```

For example, to run the demo without vision (ground truth), run the following command:
```
python system/run_demo.py demo_z0_v2 gt_demo
```

To run the vision demo, run the following command:
```
python system/run_demo.py demo_z0_v2 vision_demo
```

### Running Table 1 experiments

Similar to running the demo, first start python server (note that we are running 
`run.py` instead of `run_demo.py` here), and run the Unity executable in a separate 
command line:
```
python system/run.py <exp> <option_name> <port>
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
