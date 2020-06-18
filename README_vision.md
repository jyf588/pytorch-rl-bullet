# Vision Module

## Vision Datasets

To train and evaluate the vision module, you can either use our existing dataset, or you
can generate your own dataset. Instructions for both are below.

### Option 1: Download existing dataset

The vision dataset used in the paper can be found at the Google Drive link [here].

### Option 2: Generate your own dataset

To generate your own datasets for training the vision module, follow the below steps.

#### Step 1. Generate planning states

```
./scripts/planning_v003_20K/01_generate_states.sh
```

Replace the `--output_dir` flag with your own desired folder path to generate states to.

#### Step 2. Generate placing states

To generate placing states, run policy evaluation for stacking and placing with the
commands below.
Note: Change the `--states_dir` flag to your desired folder path to generate states to.
```
./scripts/policy/0411/place.sh
./scripts/policy/0411/stack.sh
```

This will generate a folder of states, organized as `<trial_id>/<ts>.p` pickle files.

Next, run the following to process the states. This script subsamples, adds additional
attributes, and adds surrounding objects to the states generated from the policy
evaluation step.
```
./scripts/data_0518/02_complete_place.sh
./scripts/data_0518/02_complete_stack.sh
```

#### Step 3. Generate Unity data.

Finally, we can use Unity to render images from the states. To do this, run the following 
script to start the python server to generate planning images:
```
./scripts/data_0518/03_unity_plan.sh
```

Then, in a separate command line, run the Unity client application:
```
./Linux8000.x86_64
```

The images will be generated the XX path. 

You can repeat the above process for placing and stacking states as well. To do so, 
change the above command to either of the following:
```
./scripts/data_0518/03_unity_place.sh
./scripts/data_0518/03_unity_stack.sh
```

After the data has finished generating, run the following commands to re-organize the
data into the proper format for the vision module.

```
move captures into the json folder, etc.
```

## Model Training

You can train your own segmentation and vision models, or find pretrained ones
that were used for the paper at the Google Drive link [here](https://drive.google.com/open?id=1Jcta_Ye5wG8l1traU4e414lHKOtjtssW).

### Segmentation Model

Train a segmentation model on the segmentation dataset.

```
python ns_vqa_dart/scene_parse/detectron2/dash.py train
```

Check visualizations and metrics on the training set.

```
python ns_vqa_dart/scene_parse/detectron2/dash.py eval
```

### Attribute Network

First, specify the dataset(s) that you would like to train on in 
`./scripts/data_0518/<stage>_0518/data_dirs.json`.

You can train the attribute network with the following command:
```
./scripts/data_0518/<stage>_0518/train.sh
```

To plot the loss curve, run:
```
./scripts/data_0518/<stage>_0518/plot_loss.sh
```

To run evaluation and compute metrics, run the following:
```
./scripts/data_0518/<stage>_0518/eval.sh
./scripts/data_0518/<stage>_0518/metrics.sh
```

You can replace `stage` with `plan`, `place`, or `stack`.


### Internal Notes

Internal notes for internal use, will be removed for official code release.

Here is a changelog of dataset versions and the diffs between successive versions.

- `planning_v004`, `placing_v004`, `stacking_v004` (May 8, 2020)
  - Use system code instead of enjoy.py to generate policy rollouts.
- `planning_v003` and `placing_v003` (April 17, 2020)
  - Change the camera "look at" locations:
    - `planning_v003`: Instead of looking at every single object, look once at the center of the distribution of object locations and heights.
    - `placing_v003`: Instead of looking at every single object, look once at the top of the bottom object.
  - FIX: Shadow bias is removed, now shadow and object are joined instead of 
  having a gap between them.
- `dash_v005` (April 14, 2020)
  - Changed from `placing_v002` (100 trials) to `placing_v003` (1500 trials).
  - Changed from 50/50 split between planning and placing to 25/75 split.
- `dash_v004` (Apr 13, 2020)
  - Update to use Yifeng's new placing policy: `0404_0_n_20_40`.
  - Changed from `placing.sh` to `placing_v002.sh`.
- `dash_v003`: 
  - FIX: Update the transformation of y labels (specifically, position and 
  orientation of objects) from the incorrect bworld -> ucam to the correct 
  bworld -> bshoulder -> ushoulder -> ucam transformation.
  - FIX: Images; Unity delayed rendering vs. snapshot where arm was never 
    really holding any objects before.
  - FIX: Bug where saved camera orientation is changed from xyyw -> xyzw.
- `dash_v002`: The first dataset to train both planning and placing, together.
