cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou1
export no_proxy=localhost

# List of model names and temperatures
model_names=("/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-0.5B-Instruct" "/mnt/bn/gilesluo000/pretrained_models/Qwen2.5-1.5B-Instruct")
temperatures=(0.7 0.8 0.9)

# Loop over model names and temperatures
for model_name in "${model_names[@]}"; do
  for temperature in "${temperatures[@]}"; do
    # Generate a unique output file name
    model_name=$(basename "$model_path")  # Extract the last part of the path
    output_file="output_${model_name}_temp${temperature}.txt"
    
    echo "Running inference with model: $model_name, temperature: $temperature, output file: $output_file"
    
    # Run the Python script with the specified parameters and redirect output to the file
    python ./DTRBench/run_RL/run_llm_inference.py \
      --max_concurrency 64 \
      --model_path "$model_path" \
      --temperature "$temperature" \
      --output_file "$output_file"
  done
done
