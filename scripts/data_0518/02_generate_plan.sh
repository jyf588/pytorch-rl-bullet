DST_DIR=/home/mguo/states/scenes/plan_20K_0518
SCENE_JSON_PATH=scene/json/vision_0518.json


python ns_vqa_dart/bullet/states/generate_planning_states.py \
    --scene_json_path $SCENE_JSON_PATH \
    --dst_dir $DST_DIR \
    --n_examples 20000