export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
git clone https://github.com/harrylaucngd/DTR-Bench.git
cd DTR-Bench
git checkout LLM
git config pull.rebase false

# activate conda
source /mnt/bn/gilesluo000/miniconda3/etc/profile.d/conda.sh
conda activate tianshou


# Function to copy directories
copy_directories() {
    echo "Copying /mnt/bn/gilesluo000/pretrained_model/ to /opt/tiger/DTR-Bench/GlucoseLLM/model_hub/"
    cp -r /mnt/bn/gilesluo000/pretrained_model/ /opt/tiger/DTR-Bench/GlucoseLLM/model_hub/
    echo "Directories copied."
}

# Function to give read and write permissions to all files in hindsight
set_permissions() {
    echo "Setting read and write permissions for all files in /opt/tiger/hindsight/"
    chmod -R u+rw /opt/tiger/DTR-Bench/
    echo "Permissions set."
}

# Execute functions
copy_directories
set_permissions

echo "Script completed."