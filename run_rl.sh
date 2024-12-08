cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou1


PROJECT_name=Qwen0.5B-RL-1127
task="SimGlucoseEnv-adult1"
role="run_single"
policy_name="LLM-DQN"
sweep_id=blabla

python ./DTRBench/run_RL/online_discrete_search.py --wandb_project_name $PROJECT_name --task $task --role $role --policy_name $policy_name --sweep_id $sweep_id