#!/bin/bash
#
#SBATCH --job-name="place_no_orientation"
#SBATCH --output=place_no_orientation.out

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

grasp_pi="0729_12_n"
grasp_dir="trained_models_${grasp_pi}"
place_env="InmoovShadowHandPlaceEnvNoOrientation-v0"
place_dir="${grasp_dir}_place_0729_12"
samples_place="16000000"
seed_place="8350"

r_thres="1800"
n_trials="10000"

srun bash -c "/sailhome/mikehaya/miniconda2/envs/dash/bin/python3 ./main.py --cuda --env-name ${place_env} --algo ppo --use-gae --log-interval 10 \
    --num-steps 1600 --num-processes 32  --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 \
    --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${samples_place} --use-linear-lr-decay \
    --use-proper-time-limits --clip-param 0.2 --save-dir ${place_dir} --seed ${seed_place} \
    --random_top_shape 1 --cotrain_stack_place 1 --obs_noise 0 --exclude_hard 0 --n_best_cand 1 \
    --use_obj_heights 1 --with_stack_place_bit 1 \
    --grasp_pi_name \"${grasp_pi}_25_45\""
# done echo "Done"

