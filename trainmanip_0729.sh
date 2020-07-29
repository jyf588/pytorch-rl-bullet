#!/usr/bin/env bash
# same as 0426 except:
# grasp best_cand 2 test can 1
# add 4 finger r to placing 0.3

# diff from 0411: one bit, placing cand

# use obj height for both

grasp_env="InmoovHandGraspBulletEnv-v6-no-orientation"

grasp_pi="0729_12_n"
grasp_dir="trained_models_${grasp_pi}"
samples="12000000"
seed="10150"

r_thres="1800"
n_trials="10000"

python3 main.py --env-name ${grasp_env} --algo ppo --use-gae --log-interval 10 \
    --num-steps 520 --num-processes 16  --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 \
    --num-mini-batch 16 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${samples} --use-linear-lr-decay \
    --use-proper-time-limits --clip-param 0.2 --save-dir ${grasp_dir} --seed ${seed} \
    --random_top_shape 1 --cotrain_onstack_grasp 0 --grasp_floor 1 --obs_noise 1 --n_best_cand 2 \
    --use_obj_heights 1

