#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 00:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -J bohb_runs_test # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/bohb_runs_test/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/bohb_runs_test/log/%A.%a.%N.out  # STDERR
# SBATCH --array 0-64%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/bohb_runs_test

# Job to perform
python run_train_eval.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 4 \
      --local_crops_number 8 \
      --out_dim 32768 \
      --dataset CIFAR10 \
      --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
      --output_dir $EXP_D \
      --use_fp16 true \
      --saveckp_freq 0 \
      --stn_res 32 16 \
      --world_size 4 \
      --seed 0

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
