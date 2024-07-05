import optuna
import argparse
import torch
import os
import wandb
from pathlib import Path

from DTRBench.src.helper_fn import get_policy_class, get_hparam_class, get_obj_class, get_policy_type
from DTRBench.utils.misc import to_bool
import DTRGym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def call_agent():
    try:
        obj = obj_class(args.task, hparam_space, device=args.device, logger="tensorboard")
        obj.search_once(wandb.config)
    except TimeoutError as e:
        # Update the status to 'crashed' due to timeout
        wandb.run.summary["status"] = "crashed"
        wandb.run.summary["failure_reason"] = str(e)
        wandb.run.finish()
        raise e
    else:
        # Finish the wandb experiment normally if no issues
        wandb.finish()
    return


def parse_args():
    parser = argparse.ArgumentParser()

    # training-aid hyperparameters
    parser.add_argument("--task", type=str, default="SimGlucoseEnv")
    parser.add_argument("--setting", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="debug")
    parser.add_argument("--training_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--buffer_size", type=int, default=5e4)
    parser.add_argument("--linear", type=to_bool, default=False)
    parser.add_argument("--policy_name", type=str, default="DQN",
                        choices=["DQN",])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--role", type=str, default="run_single", choices=["sweep", "agent", "run_single"])
    args = parser.parse_known_args()[0]

    return args

if __name__ == "__main__":
    args = parse_args()
    hparam_class = get_hparam_class(args.policy_name, offline=False)
    obj_class = get_obj_class(args.policy_name, offline=False)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    policy_type = get_policy_type(args.policy_name, offline=False)
    env_name = args.task + f"-{policy_type}-setting{args.setting}"
    log_dir = os.path.join(args.log_dir, env_name+'-'+args.policy_name)
    hparam_space = hparam_class(args.policy_name,
                                log_dir,
                                args.training_num,  # number of training envs
                                args.test_num,  # number of test envs
                                args.epoch,
                                args.step_per_epoch,  # number of training steps per epoch
                                args.buffer_size,
                                args.num_actions,
                                linear=args.linear
                                )
    search_space = hparam_space.get_search_space()

    print("All prepared. Start to experiment")
    if args.role == "sweep":
        sweep_configuration = {
            "method": "grid",
            "project": 'LLM4RL',
            "name": env_name + f"-{args.policy_name}",
            "metric": {"goal": "maximize", "name": "reward_best"},
            "parameters": search_space
        }
        sweep_id = wandb.sweep(sweep_configuration, project=args.project)
        wandb.agent(sweep_id=sweep_id, function=call_agent, project=args.project, entity="gilesluo")
    else:
        if args.role == "agent":
            wandb.agent(sweep_id=args.sweep_id, function=call_agent, project=args.project, entity="gilesluo")
        if args.role == "run_single":
            obj = obj_class(env_name, hparam_space, device=args.device, logger="tensorboard")
            config_dict = hparam_space.sample(mode="random")
            obj.search_once(config_dict)
        else:
            print("role must be one of [sweep, agent, run_single], get {}".format(args.role))
            raise NotImplementedError