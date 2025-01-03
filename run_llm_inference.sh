#!/bin/bash

cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou1
export no_proxy=localhost

# List of model names and temperatures
model_names=("/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-1.5B-Instruct")
temperatures=(0.7 0.8 0.9)

# Loop over all models and temperatures
for temperature in "${temperatures[@]}"; do
  for model_path in "${model_names[@]}"; do
    (
      # Determine port based on model size
      if [[ "$model_path" == *"0.5B"* ]]; then
        port=8001
      elif [[ "$model_path" == *"1.5B"* ]]; then
        port=8000
      else
        echo "Unknown model size in path: $model_path"
        exit 1
      fi

      model_name=$(basename "$model_path")
      output_file="output_${model_name}_temp${temperature}.txt"

      echo "Preparing to run inference with model: $model_name, temperature: $temperature" 

      # Run your Python script
      python ./DTRBench/run_RL/run_llm_inference.py \
        --max_concurrency 64 \
        --model_path "$model_path" \
        --temperature "$temperature" \
        --output_file "$output_file" \
        --port "$port"
    )
  done
done
# Wait for all background jobs to complete

