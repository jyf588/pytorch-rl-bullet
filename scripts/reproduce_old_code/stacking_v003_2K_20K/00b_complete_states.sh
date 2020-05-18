STATES_DIR=~/mguo/data/states
PARTIAL_STATES_DIR=$STATES_DIR/partial
FULL_STATES_DIR=$STATES_DIR/full

time python enjoy.py \
    --env-name InmoovHandPlaceBulletEnv-v9 \
    --load-dir trained_models_0404_0_n_place_0404_0/ppo \
    --non-det 0 \
    --seed=18980 \
    --random_top_shape 1 \
    --renders 0 \
    --exclude_hard 0 \
    --obs_noise 1 \
    --n_best_cand 2 \
    --cotrain_stack_place 0 \
    --place_floor 0 \
    --grasp_pi_name "0404_0_n_20_40" \
    --use_obj_heights 1 \
    --with_stack_place_bit 0 \
    --save_states 1 \
    --states_dir $PARTIAL_STATES_DIR/stacking_v003_2K \
    --n_trials 2000

time python ns_vqa_dart/bullet/states/sample_states.py \
    --seed 1 \
    --src_dir $PARTIAL_STATES_DIR/stacking_v003_2K \
    --dst_dir $PARTIAL_STATES_DIR/stacking_v003_2K_20K \
    --sample_size 20000

time python ns_vqa_dart/bullet/states/complete_states.py \
    --src_dir $PARTIAL_STATES_DIR/stacking_v003_2K_20K \
    --dst_dir $FULL_STATES_DIR/stacking_v003_2K_20K

time python ns_vqa_dart/bullet/states/add_surrounding_states.py \
    --src_dir $FULL_STATES_DIR/stacking_v003_2K_20K \
    --dst_dir $FULL_STATES_DIR/stacking_v003_2K_20K

cd $FULL_STATES_DIR
rm stacking_v003_2K_20K.zip
time zip -r stacking_v003_2K_20K.zip stacking_v003_2K_20K
