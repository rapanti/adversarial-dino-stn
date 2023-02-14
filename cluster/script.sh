#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:8
#SBATCH -J dino-stn-mtadam-tcp # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/dino-stn-mtadam-tcp/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/dino-stn-mtadam-tcp/log/%A.%a.%N.out  # STDERR
#SBATCH --array 0-7%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/dino-stn-mtadam-tcp

# Job to perform
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --standalone \
    run_train_eval.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 4 \
      --out_dim 32768 \
      --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
      --dataset CIFAR10 \
      --output_dir $EXP_D \
      --epochs 100 \
      --warmup_epochs 10 \
      --batch_size 256 \
      --use_fp16 false \
      --saveckp_freq 34 \
      --stn_res 32 16 \
      --invert_stn_gradients true \
      --use_stn_optimizer true \
      --stn_theta_norm true \
      --use_unbounded_stn true \
      --stn_mode translation_scale_symmetric \
      --use_stn_penalty true \
      --invert_penalty true \
      --penalty_loss ThetaCropsPenalty \
      --epsilon 1 \
      --local_crops_number 8 \
      --stn_color_augment true \
      --summary_writer_freq 100

x=$?
if [ $x == 0 ]
then
  scancel $SLURM_JOB_ID
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
