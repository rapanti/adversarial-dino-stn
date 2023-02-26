#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080-advanced # partition (queue)
#SBATCH -t 23:59:59 # time (D-HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH -J eval-pretrained-backbone-100 # sets the job name. If not specified, the file name will be used as job name
#SBATCH -o /work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval-pretrained-backbone-100/log/%A.%a.%N.out  # STDOUT
#SBATCH -e /work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval-pretrained-backbone-100/log/%A.%a.%N.out  # STDERR
#SBATCH --array 0-3%1

# Print some information about the job to STDOUT
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source /home/rapanti/.profile
source activate dino

EXP_D=/work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval-pretrained-backbone-100

# Job to perform
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --standalone \
    eval_linear.py \
      --arch vit_nano \
      --img_size 32 \
      --patch_size 4 \
      --dataset CIFAR10 \
      --data_path /work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10 \
      --output_dir $EXP_D \
      --epochs 300 \
      --batch_size 768 \
      --pretrained_weights /work/dlclarge2/rapanti-metassl-dino-stn/experiments/eval-pretrained-backbone-100/checkpoint0100.pth


# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
