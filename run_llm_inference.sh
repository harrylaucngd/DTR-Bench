#!/bin/bash

cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou1
export no_proxy=localhost

# List of model names and temperatures
model_names=("/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-0.5B-Instruct" \
             "/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-1.5B-Instruct")
temperatures=(0.7 0.8 0.9)

# Define pairs of CUDA devices
device_pairs=("0,1" "2,3" "4,5" "6,7")

# Function to find and lock an available device pair atomically
get_available_device_pair() {
  while true; do
    for i in "${!device_pairs[@]}"; do
      local lock_file="/tmp/device_pair_$i.lock"
      # Attempt an atomic lock: only one process can create lock_file at a time
      if ( set -o noclobber; echo "$$" > "$lock_file" ) 2>/dev/null; then
        echo "$i"
        return
      fi
    done
    # If no device pair is free, wait and try again
    sleep 1
  done
}

# Loop over all models and temperatures
for model_path in "${model_names[@]}"; do
  for temperature in "${temperatures[@]}"; do
    (
      # Acquire a device pair
      pair_index=$(get_available_device_pair)
      cuda_devices="${device_pairs[$pair_index]}"
      model_name=$(basename "$model_path")
      output_file="output_${model_name}_temp${temperature}.txt"

      echo "Preparing to run inference with model: $model_name, temperature: $temperature"
      echo "Running inference on CUDA devices: $cuda_devices"

      # Run your Python script
      python ./DTRBench/run_RL/run_llm_inference.py \
        --max_concurrency 64 \
        --model_path "$model_path" \
        --temperature "$temperature" \
        --output_file "$output_file" \
        --cuda_visible_devices "$cuda_devices"

      # Release this device pair
      rm -f "/tmp/device_pair_$pair_index.lock"
    ) &
  done
done

# Wait for all background jobs to complete
wait
