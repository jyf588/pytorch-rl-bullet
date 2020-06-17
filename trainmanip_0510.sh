#!/usr/bin/env bash
# same as 0426 except:
# grasp best_cand 2 test can 1
# add 4 finger r to placing 0.3

# diff from 0411: one bit, placing cand

# use obj height for both

grasp_env="InmoovHandGraspBulletEnv-v6"

grasp_pi="0510_11_n"
grasp_dir="trained_models_${grasp_pi}"
samples="12000000"
seed="10150"

r_thres="1800"
n_trials="10000"

place_env="InmoovHandPlaceBulletEnv-v9"
place_dir="${grasp_dir}_place_0510_11"
samples_place="16000000"
seed_place="8350"

python main.py --env-name ${grasp_env} --algo ppo --use-gae --log-interval 10 \
    --num-steps 520 --num-processes 16  --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 \
    --num-mini-batch 16 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${samples} --use-linear-lr-decay \
    --use-proper-time-limits --clip-param 0.2 --save-dir ${grasp_dir} --seed ${seed} \
    --random_top_shape 1 --cotrain_onstack_grasp 0 --grasp_floor 1 --obs_noise 1 --n_best_cand 2 \
    --use_obj_heights 1

#has to be test phase 0, has assertion there
python enjoy.py --env-name ${grasp_env} --load-dir "${grasp_dir}/ppo/" --non-det 0 --seed=18991 \
    --renders 0 --random_top_shape 1 --cotrain_onstack_grasp 0 --grasp_floor 1 --obs_noise 1 --n_best_cand 1 \
    --has_test_phase 0 --use_obj_heights 1 \
    --save_final_states 1 --r_thres ${r_thres} --save_final_s 25 --save_final_e 45 --n_trials ${n_trials}

python main.py --env-name ${place_env} --algo ppo --use-gae --log-interval 10 \
    --num-steps 1600 --num-processes 16  --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 \
    --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${samples_place} --use-linear-lr-decay \
    --use-proper-time-limits --clip-param 0.2 --save-dir ${place_dir} --seed ${seed_place} \
    --random_top_shape 1 --cotrain_stack_place 1 --obs_noise 0 --exclude_hard 0 --n_best_cand 1 \
    --use_obj_heights 1 --with_stack_place_bit 1 \
    --grasp_pi_name "${grasp_pi}_25_45"

