#!/bin/bash
#
#SBATCH --job-name="enjoy_no_orientation"
#SBATCH --output=enjoyp_no_orientation.out

#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --partition=move
# only use the following if you want email notification
#SBATCH --mail-user=mikephayashi@gmail.com
#SBATCH --mail-type=ALL
# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
# sample process

grasp_env="InmoovHandGraspBulletEnvNoOrientation-v0-v0"

grasp_pi="0729_12_n"
grasp_dir="trained_models_${grasp_pi}"
samples="12000000"
seed="10150"

r_thres="1800"
n_trials="10000"

srun bash -c "/sailhome/mikehaya/miniconda2/envs/dash/bin/python3 ./pytorch-rl-bullet/enjoy.py --env-name ${grasp_env} --load-dir \"${grasp_dir}/ppo/\" --non-det 0 --seed=18991 \
    --renders 0 --random_top_shape 1 --cotrain_onstack_grasp 0 --grasp_floor 1 --obs_noise 1 --n_best_cand 1 \
    --has_test_phase 0 --use_obj_heights 1 \
    --save_final_states 1 --r_thres ${r_thres} --save_final_s 25 --save_final_e 45 --n_trials ${n_trials}"
# done echo "Done"