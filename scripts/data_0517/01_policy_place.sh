STATES_NAME=place_100K_0517

time python enjoy.py \
    --env-name InmoovHandPlaceBulletEnv-v9 \
    --load-dir trained_models_0404_0_n_place_0404_0/ppo \
    --non-det 0 \
    --seed=18980 \
    --random_top_shape 1 \
    --renders 0 \
    --exclude_hard 0 \
    --obs_noise 1 \
    --n_best_cand 1 \
    --cotrain_stack_place 0 \
    --place_floor 1 \
    --grasp_pi_name "0404_0_n_20_40" \
    --use_obj_heights 1 \
    --with_stack_place_bit 0 \
    --save_states 1 \
    --states_dir /home/mguo/states/partial/$STATES_NAME \
    --n_trials 100000
