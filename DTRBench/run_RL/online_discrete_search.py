#!/usr/bin/env python

import argparse
import torch
import os
import wandb
from pathlib import Path
from DTRBench.src.helper_fn import get_hparam_class, get_obj_class, get_policy_type
from DTRBench.utils.misc import to_bool
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
wandb.require("core")

def call_agent():
    try:
        obj = obj_class(env_name, env_args, hparam_space, device=args.device)
        obj.wandb_search()
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

    parser.add_argument("--wandb_project_name", type=str, default="LLM4RL")
    parser.add_argument("--sweep_id", type=str, default="1did1f4s", help="sweep id for wandb,"
                                                                         " only used in agent mode")
    parser.add_argument("--task", type=str, default="SimGlucoseEnv-adult1",
                        help="remember to change this for different tasks! "
                             "Wandb sweep won't work correctly if this is not changed!")
    parser.add_argument("--log_dir", type=str, default="sweep_log/")
    parser.add_argument("--training_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--num_actions", type=int, default=11)
    parser.add_argument("--step_per_epoch", type=int, default=10 * 12 * 16)
    parser.add_argument("--buffer_size", type=int, default=1e6)
    parser.add_argument("--linear", type=to_bool, default=False)
    parser.add_argument("--policy_name", type=str, default="LLM-DQN",  # Change this for different sweep!
                        choices=["LLM-DQN", "DQN", "TD3"],
                        help="remember to change this for different tasks! "
                             "Wandb sweep won't work correctly if this is not changed!")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--role", type=str, default="sweep", choices=["sweep", "agent", "run_single"])
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    hparam_class = get_hparam_class(args.policy_name, offline=False)
    obj_class = get_obj_class(args.policy_name, offline=False)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    policy_type = get_policy_type(args.policy_name, offline=False)
    env_args = {"discrete": policy_type == "discrete",
                "n_act": args.num_actions,}

    env_name = args.task
    log_dir = os.path.join(args.log_dir, env_name + '-' + args.policy_name)
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
            "project": args.wandb_project_name,
            "name": env_name + f"-{args.policy_name}",
            "metric": {"goal": "maximize", "name": "test/returns_stat/mean"},
            "parameters": search_space
        }
        sweep_id = wandb.sweep(sweep_configuration, project=args.wandb_project_name)
        wandb.agent(sweep_id=sweep_id, function=call_agent, project=args.wandb_project_name)
    else:
        if args.role == "agent":
            wandb.agent(sweep_id=args.sweep_id, function=call_agent, project=args.wandb_project_name)
        if args.role == "run_single":
            obj = obj_class(env_name, env_args, hparam_space, device=args.device)
            config_dict = hparam_space.sample(mode="random")
            obj.search_once({**config_dict, **{"wandb_project_name": args.wandb_project_name}})
        else:
            print("role must be one of [sweep, agent, run_single], get {}".format(args.role))
            raise NotImplementedError
