cl## Training Grasping and Stacking/Placing policies:

```
sh trainmanip_0510.sh
```
For each new training, you will need to modify the grasp and place pi names to avoid overwriting previous experiments.

(Make sure we understand what each command means in the shell script)

## Testing Grasping Policy only

```
python enjoy.py --env-name InmoovHandGraspBulletEnv-v6 --load-dir <path to grasp model>/ppo/ --non-det 0 --seed=18991 --renders 1 --random_top_shape 1 --obs_noise 1 --n_best_cand 2 --warm_start_phase 0 --has_test_phase 1 --save_final_states 0 --r_thres 1800 --use_obj_heights 1 --cotrain_onstack_grasp 0 --grasp_floor 1
```

## Testing Stacking/Placing Policy only

```
python enjoy.py --env-name InmoovHandPlaceBulletEnv-v9 --load-dir <path to place model>/ppo --non-det 0 --seed=18981 --random_top_shape 1 --renders 1 --exclude_hard 0 --obs_noise 1 --n_best_cand 1 --cotrain_stack_place 1 --grasp_pi_name "${grasp_pi}_25_45"(grasp policy name, see train_0510.sh) --with_stack_place_bit 1 --use_obj_heights 1 --r_thres 4000 
```

## Testing the whole manipulation module without language and vision
If you modified training pi names (i.e. their storing dir), 
open `main_sim_clean_test.py`, search for the strings containing "0510" and replace them with the grasp & place pi names in `trainmanip_0510.sh`
```
sh testmanip_0510.sh
```
(Make sure we understand what each command means in the shell script)

The test stats will be printed in final_stats.txt, `total w/o OR` should be around 80% to 90% to mean policy being good enough; if they are not, change render from 0 to 1 in the previous test script to see if you can observe problems from bullet GUI: