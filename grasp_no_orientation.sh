#!/bin/bash
#
#SBATCH --job-name="grasp_no_orientation"
#SBATCH --output=grasp_no_orientation.out

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

grasp_env="InmoovHandGraspBulletEnvNoOrientation-v0"

grasp_pi="0729_12_n"
grasp_dir="trained_models_${grasp_pi}"
samples="12000000"
seed="10150"

r_thres="1800"
n_trials="10000"

srun bash -c "/sailhome/mikehaya/miniconda2/envs/dash/bin/python3 ./pytorch-rl-bullet/main.py --cuda --env-name ${grasp_env} --algo ppo --use-gae --log-interval 10 \
    --num-steps 260 --num-processes 32  --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 \
    --num-mini-batch 16 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${samples} --use-linear-lr-decay \
    --use-proper-time-limits --clip-param 0.2 --save-dir ${grasp_dir} --seed ${seed} \
    --random_top_shape 1 --cotrain_onstack_grasp 0 --grasp_floor 1 --obs_noise 1 --n_best_cand 2 \
    --use_obj_heights 1 --cuda --warm-start ./trained_models_0729_12_n/ppo/InmoovHandGraspBulletEnvv6NoOorientation-v0.pt"
# done echo "Done"