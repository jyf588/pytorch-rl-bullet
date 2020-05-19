SHORT_SET=stack_0518
SET=stack_2K_20K_0518
DATA_DIRS_JSON=scripts/data_0518/$SHORT_SET/data_dirs.json
MODEL_NAME=2020_05_18_02_50_47
MODEL_DIR=/home/mguo/outputs/$SET/$MODEL_NAME
RUN_DIR=$MODEL_DIR/eval
CHECKPOINT_PATH=$MODEL_DIR/checkpoint_best.pt

python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --data_dirs_json $DATA_DIRS_JSON \
    --run_dir $RUN_DIR \
    --load_checkpoint_path $CHECKPOINT_PATH
