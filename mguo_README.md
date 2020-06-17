# DASH

Installation instructions: See INSTALL.md.

## System

First, start the python server:

```
python system/run.py
```

Next, run the Unity executable:
```
./LinuxBuildLocalhost.x86_64
```

## Tables and Figures

### Table 1

First, generate the test scenes:

```
time python scene/generate.py table1  # ETA: 1 second
```

### Table 2
TODO

### Table 3: Vision Performance

```
python system/vision_metrics.py t1
```


## Vision Module

### Vision Datasets

To generate your own datasets for training the vision module, run the following
commands. You can choose to either run `vision_tiny` or `vision` as the dataset name.

Step 1. Generate scenes for planning, placing, and stacking. 

```
time python scene/generate.py <dataset>
```

Step 2. Run evaluation of the policy to generate placing and stacking frames.
(`vision_tiny`: 1 minute 24 seconds)

```
time python system/run.py <dataset> vision_states
python system/client.py
```

Step 3. Generate unity images and segmentations.
# 30 minutes on 20K states
```
time python system/run.py <dataset> unity_dataset
```

Then, launch the Unity application.

### Segmentation Model

You can train your own segmentation module and vision module.

Train a segmentation module on the segmentation dataset.

```
time python ns_vqa_dart/scene_parse/detectron2/dash.py train <seg_dataset>
```

Check visualizations and metrics on the training set.

```
time python ns_vqa_dart/scene_parse/detectron2/dash.py eval <dataset>
```

### Feature Extraction Model

First, generate segmentation masks on the feature extraction training set using the 
trained segmentation model.

```
time python ns_vqa_dart/scene_parse/detectron2/dash.py eval <attr_dataset> --save_segs
```

Next, generate the dataset we will use for training. Essentially this step creates
training inputs for each object in the dataset (as opposed to one example per scene).

```
TODO
```

Finally, train the feature extraction model:

```
time ./ns_vqa_dart/scripts/exp/seg_tiny/run.sh
```

### Vision dataset versions 
Here is a changelog of dataset versions and the diffs between successive
versions:

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
