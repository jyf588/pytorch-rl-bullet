SHORT_SET=plan_0518
SET=plan_20K_0518
DATA_DIRS_JSON=scripts/data_0518/$SHORT_SET/data_dirs.json
MODEL_NAME=2020_05_18_03_19_01
MODEL_DIR=/home/mguo/outputs/$SET/$MODEL_NAME
RUN_DIR=$MODEL_DIR/eval
CHECKPOINT_PATH=$MODEL_DIR/checkpoint_best.pt

python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --data_dirs_json $DATA_DIRS_JSON \
    --run_dir $RUN_DIR \
    --load_checkpoint_path $CHECKPOINT_PATH
