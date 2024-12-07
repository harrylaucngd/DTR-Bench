cd /opt/tiger/DTR-Bench
export PYTHONPATH='./'

# Activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou


PROJECT_name=wandb_project_name
task="SimGlucoseEnv-adult1"
role="sweep"
policy_name="LLM-DQN"
sweep_id=blabla

python ./DTRBench/run_RL/online_discrete_search.py --wandb_project_name $PROJECT_name --task $task --role $role --policy_name $policy_name --sweep_id $sweep_id