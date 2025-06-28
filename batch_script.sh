#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH -c 40
#SBATCH -p gpu
#SBATCH --job-name=my_torch_job
#SBATCH --output=check/log_batch_script.out
#SBATCH --error=check/log_batch_script.err

# Set up modules
source /etc/profile.d/modules.sh
module load slurm mpi cmake
module load cuda

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# export NCCL_DEBUG=INFO

# ---- 1. Distributed Run ----
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m tests.MNIST \

# ---- 2. Non-Distributed Run (single process, single GPU) ----
# Only run on the first node, first task
python -m tests.linear_MNIST

# ---- 3. Compare Outputs ----
python compare_files.py