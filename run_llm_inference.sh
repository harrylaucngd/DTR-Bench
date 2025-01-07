#!/bin/bash

cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou1
export no_proxy=localhost

# List of model names, temperatures, and policy names
model_names=("/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-0.5B-Instruct")
temperatures=(0 0.7 1.0)
policy_names=("hidden-sys" "full-sys" "base" "cot")

# Loop over all models, temperatures, and policy names
for temperature in "${temperatures[@]}"; do
  for model_path in "${model_names[@]}"; do
    for policy_name in "${policy_names[@]}"; do
      (
        # Determine port based on model size
        if [[ "$model_path" == *"0.5B"* ]]; then
          port=8000
        elif [[ "$model_path" == *"1.5B"* ]]; then
          port=8001
        elif [[ "$model_path" == *"7B"* ]]; then
          port=8002
        elif [[ "$model_path" == *"14B"* ]]; then
          port=8003
        elif [[ "$model_path" == *"32B"* ]]; then
          port=8004
        else
          echo "Unknown model size in path: $model_path"
          exit 1
        fi

        model_name=$(basename "$model_path")
        output_file="./output_${model_name}_temp${temperature}_policy${policy_name}.txt"

        echo "Preparing to run inference with model: $model_name, temperature: $temperature, policy: $policy_name" 

        # Run your Python script
        python ./DTRBench/run_RL/run_llm_inference.py \
          --max_concurrency 64 \
          --model_path "$model_path" \
          --temperature "$temperature" \
          --output_file "$output_file" \
          --port "$port" \
          --policy_name "$policy_name"
      )
        hdfs dfs -put -f "$output_file" hdfs://haruna/home/byte_aml_rl/user/zhiyao/llm4rl_inference/
    done
  done
done
# Wait for all background jobs to complete
