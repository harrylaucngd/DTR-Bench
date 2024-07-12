#!/bin/bash

# Number of screens to create
n=$1

# Bash script to execute in each screen
bash_script=$2

# Conda environment to activate
conda_env=$3

# Check if the number of screens, Bash script, and Conda environment are provided
if [ -z "$n" ] || [ -z "$bash_script" ] || [ -z "$conda_env" ]; then
  echo "Usage: $0 <number_of_screens> <bash_script> <conda_env>"
  exit 1
fi

# Check if the specified Bash script exists and is executable
if [ ! -x "$bash_script" ]; then
  echo "Error: $bash_script does not exist or is not executable."
  exit 1
fi

# Loop to create and execute the Bash script in each screen
for ((i=1; i<=n; i++))
do
  screen_name="screen_$i"
  # Create a new screen session, activate the Conda environment, and execute the Bash script
  screen -dmS $screen_name bash -c "source ~/.bashrc; conda activate $conda_env; bash $bash_script; exec bash"
  echo "Created $screen_name, activated Conda environment $conda_env, and executed: $bash_script"
done

echo "Finished creating $n screens."
