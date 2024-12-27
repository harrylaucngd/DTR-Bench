cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou1
export no_proxy=localhost
python ./DTRBench/run_RL/run_llm_inference.py --max_concurrency 64