#!/usr/bin/env python
import argparse
import warnings
import wandb
from DTRBench.naive_baselines.baselineHparams import BaselineHyperParams
from DTRBench.naive_baselines.baselineObj import BaselineObj

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
wandb.require("core")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project_name", type=str, default="LLM4RL-1122")
    parser.add_argument("--sweep_id", type=str, default="7yum2pwn", help="sweep id for wandb,"
                                                                         " only used in agent mode")
    parser.add_argument("--role", type=str, default="agent", choices=["sweep", "agent", "run_single"])
    args = parser.parse_known_args()[0]
    return args


def call_agent():
    try:
        obj = BaselineObj({"discrete": False}, hparam)
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


if __name__ == "__main__":
    args = parse_args()
    hparam = BaselineHyperParams(obs_window=1)
    if args.role == "sweep":
        sweep_configuration = {
            "method": "grid",
            "project": args.wandb_project_name,
            "name": "all-naive-baselines",
            "metric": {"goal": "maximize", "name": "test/returns_stat/mean"},
            "parameters": hparam.get_search_space(),
        }
        sweep_id = wandb.sweep(sweep_configuration, project=args.wandb_project_name)
        wandb.agent(sweep_id=sweep_id, function=call_agent, project=args.wandb_project_name)
    elif args.role == "agent":
        wandb.agent(sweep_id=args.sweep_id, function=call_agent, project=args.wandb_project_name)
    else:
        print("role must be one of [sweep, agent, run_single], get {}".format(args.role))
        raise NotImplementedError
