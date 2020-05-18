DATA_DIRS_JSON=ns_vqa_dart/scripts/combined/data_dirs.json
MODEL_NAME=2020_05_16_22_47_25
MODEL_DIR=/home/mguo/outputs/combined/$MODEL_NAME
RUN_DIR=$MODEL_DIR/eval
CHECKPOINT_PATH=$MODEL_DIR/checkpoint_best.pt

python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --data_dirs_json $DATA_DIRS_JSON \
    --run_dir $RUN_DIR \
    --load_checkpoint_path $CHECKPOINT_PATH
