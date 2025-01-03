source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate textgrad

export VLLM_RPC_TIMEOUT=100000000

# Set the visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve /mnt/bn/gilesluo000/pretrained_models/Qwen2.5-1.5B-Instruct --port 8000 --dtype bfloat16 --tensor-parallel-size 4
# --pipeline-parallel-size 2