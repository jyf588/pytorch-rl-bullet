SET=placing_v003_2K_20K
DATA_DIR=/home/mguo/data/$SET/data
RUN_DIR=/home/mguo/outputs/$SET
CHECKPOINT_EVERY=1000
NUM_ITERS=600000


python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --data_dir $DATA_DIR \
    --run_dir $RUN_DIR \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8
