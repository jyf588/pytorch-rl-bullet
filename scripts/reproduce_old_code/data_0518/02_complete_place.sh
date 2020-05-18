STATES_DIR=/home/mguo/states
POLICY_STATES_DIR=$STATES_DIR/partial
COMPLETE_STATES_DIR=$STATES_DIR/complete_policy
SURROUND_STATES_DIR=$STATES_DIR/scenes

TASK=place
ORIGINAL_SET=place_100K_0518
SET=place_2K_20K_0518

SAMPLE_SRC_DIR=$POLICY_STATES_DIR/$ORIGINAL_SET
SAMPLE_DST_DIR=$POLICY_STATES_DIR/$SET
COMPLETE_SRC_DIR=$SAMPLE_DST_DIR
COMPLETE_DST_DIR=$COMPLETE_STATES_DIR/$SET
SURROUND_SRC_DIR=$COMPLETE_DST_DIR
SURROUND_DST_DIR=$SURROUND_STATES_DIR/$SET

SCENE_JSON_PATH=scene/json/vision_0517.json

time python ns_vqa_dart/bullet/states/sample_states.py \
    --seed 1 \
    --src_dir $SAMPLE_SRC_DIR \
    --dst_dir $SAMPLE_DST_DIR \
    --start_trial 1 \
    --end_trial 2001 \
    --sample_size 20000

time python ns_vqa_dart/bullet/states/complete_states.py \
    --task $TASK \
    --src_dir $COMPLETE_SRC_DIR \
    --dst_dir $COMPLETE_DST_DIR \
    --scene_json_path $SCENE_JSON_PATH

time python ns_vqa_dart/bullet/states/add_surrounding_states.py \
    --task $TASK \
    --src_dir $SURROUND_SRC_DIR \
    --dst_dir $SURROUND_DST_DIR \
    --scene_json_path $SCENE_JSON_PATH
