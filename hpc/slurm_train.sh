#!/bin/bash
#SBATCH --job-name=deepearth_train
#SBATCH --output=logs/deepearth_%j.out
#SBATCH --error=logs/deepearth_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

# Load modules (adjust based on your HPC)
module load cuda/11.8
module load cudnn/8.6
module load gcc/11.2
module load python/3.10
module load openmpi/4.1.4

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepearth

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Distributed training settings
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0

# Wandb settings (optional)
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="deepearth_hpc"
export WANDB_RUN_NAME="deepearth_${SLURM_JOB_ID}"

# Create directories
mkdir -p logs checkpoints results

# Log system info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run training
python -u hpc/train_distributed.py \
    --config configs/deepearth_hpc.yaml \
    --num_nodes 1 \
    --gpus_per_node 4 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --num_epochs 100 \
    --checkpoint_dir checkpoints/job_${SLURM_JOB_ID} \
    --mixed_precision fp16 \
    --compile_model \
    --data_path /scratch/datasets/earth_observations \
    --resume_from_checkpoint
