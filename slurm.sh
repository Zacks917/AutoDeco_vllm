#!/bin/bash
#SBATCH --job-name=mtp                 # Job name
#SBATCH --partition=llm                 # Select llm-debug partition for debugging
#SBATCH --qos=llm                       # Use the llm-debug QOS
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                          # Number of GPUs (1 GPU)
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH --cpus-per-task=16                    # Number of CPU cores per task
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00                       # Time limit: 3 days
#SBATCH --output=/aifs4su/guhao/MTP/logs/slurm-%j.out   # Standard output log
#SBATCH --error=/aifs4su/guhao/MTP/logs/slurm-%j.err    # Standard error log


export PIP_USER=false
export PYTHONNOUSERSITE=0

source /home/guhao/miniconda3/etc/profile.d/conda.sh
source /aifs4su/guhao/envs/mtp/bin/activate
module add cuda/12.8

echo "Job ID: $SLURM_JOB_ID"

cd /aifs4su/guhao/MTP/AutoDeco_vllm
mkdir -p /aifs4su/guhao/MTP/logs/generation_log

# CUDA_VISIBLE_DEVICES=0 python test_eagle3.py --spe_model_path /aifs4su/guhao/Models/Qwen3-8B-StopHead --mtp_size 3 --data math500 &
# CUDA_VISIBLE_DEVICES=1 python test_eagle3.py --spe_model_path /aifs4su/guhao/Models/Qwen3-8B-StopHead --mtp_size 3 --data math500 --disable_should_stop True &

export PROFILE_EAGLE_LOG_INTERVAL=500 PROFILE_SHOULD_STOP_LOG_INTERVAL=200

MTP_SIZE=6
BATCH_SIZE=1

PROFILE_EAGLE_GLOBAL=1 PROFILE_SHOULD_STOP=1 \
CUDA_VISIBLE_DEVICES=0 python test_eagle3.py \
  --spe_model_path /aifs4su/guhao/Models/Qwen3-8B-StopHead \
  --mtp_size $MTP_SIZE \
  --batch_size $BATCH_SIZE \
  --single_prompt_batch \
  --sync_stop_in_batch &


PROFILE_EAGLE_GLOBAL=1 PROFILE_SHOULD_STOP=1 \
CUDA_VISIBLE_DEVICES=1 python test_eagle3.py \
  --spe_model_path /aifs4su/guhao/Models/Qwen3-8B-StopHead \
  --mtp_size $MTP_SIZE \
  --batch_size $BATCH_SIZE \
  --single_prompt_batch \
  --disable_should_stop True \
  --sync_stop_in_batch &

wait