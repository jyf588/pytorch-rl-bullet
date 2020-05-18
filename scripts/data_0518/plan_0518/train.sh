DATA_DIRS_JSON=scripts/data_0518/plan_0518/data_dirs.json

SET=plan_20K_0518
RUN_DIR=/home/mguo/outputs/$SET
CHECKPOINT_EVERY=7500
NUM_ITERS=600000


python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --data_dirs_json $DATA_DIRS_JSON \
    --run_dir $RUN_DIR \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8
