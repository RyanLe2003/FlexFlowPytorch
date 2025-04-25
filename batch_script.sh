#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH -c 40
#SBATCH -p gpu
#SBATCH --job-name=my_torch_job
#SBATCH --output=logs/slurm-%j.out

# Set up modules
source /etc/profile.d/modules.sh
module load slurm mpi cmake
module load cuda

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Torchrun distributed launch
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc-per-node=1 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  training.py