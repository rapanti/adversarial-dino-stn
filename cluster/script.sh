#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -J dino-stn-tcp-eps100-baseline # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/dino-stn-tcp-eps100-baseline/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/dino-stn-tcp-eps100-baseline/log/%A.%a.%N.out  # STDERR
#SBATCH --array 0-3%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/dino-stn-tcp-eps100-baseline

# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
    run_train_eval.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 4 \
      --local_crops_number 8 \
      --out_dim 32768 \
      --dataset CIFAR10 \
      --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
      --output_dir $EXP_D \
      --epochs 300 \
      --warmup_epochs 30 \
      --batch_size 256 \
      --use_fp16 true \
      --saveckp_freq 100 \
      --stn_res 32 16 \
      --invert_stn_gradients true \
      --use_stn_optimizer false \
      --stn_theta_norm true \
      --use_unbounded_stn true \
      --stn_mode translation_scale_symmetric \
      --stn_penalty ThetaCropsPenalty \
      --invert_penalty true \
      --epsilon 100 \
      --stn_color_augment true \
      --summary_writer_freq 100

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
