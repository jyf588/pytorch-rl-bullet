SET=planning_v003_20K
MODEL_NAME=2020_05_15_23_43_08
DATA_DIR=/home/mguo/data/$SET/data
MODEL_DIR=/home/mguo/outputs/$SET/$MODEL_NAME
RUN_DIR=$MODEL_DIR/eval
CHECKPOINT_PATH=$MODEL_DIR/checkpoint_best.pt

python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --data_dir $DATA_DIR \
    --run_dir $RUN_DIR \
    --load_checkpoint_path $CHECKPOINT_PATH
