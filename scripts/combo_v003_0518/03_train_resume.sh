SET=stacking_v003_2K_20K
DATA_DIR=/home/mguo/data/$SET/data
RUN_DIR=/home/mguo/outputs/$SET
CHECKPOINT_EVERY=5000
NUM_ITERS=600000


python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --data_dir $DATA_DIR \
    --run_dir $RUN_DIR \
    --checkpoint_every $CHECKPOINT_EVERY \
    --checkpoint_t 20000 \
    --resume_dir $RUN_DIR/2020_05_15_19_51_41 \
    --load_checkpoint_path $RUN_DIR/2020_05_15_19_51_41/checkpoint_iter00020000.pt \
    --num_iters $NUM_ITERS \
    --num_workers 8
