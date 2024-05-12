import optuna
import argparse
import torch
import os
from pathlib import Path

from DTRBench.src.helper_fn import get_best_hparams, get_policy_class, get_hparam_class, get_obj_class, get_policy_type
from DTRBench.utils.misc import to_bool, early_stopping_callback
from DTRBench.src.helper_fn import create_study_with_filter
import DTRGym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    # training-aid hyperparameters
    parser.add_argument("--sampler", type=str, default="TPESampler", choices=["TPESampler", "BruteForceSampler"])
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--task", type=str, default="SimGlucoseEnv")
    parser.add_argument("--setting", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="settings_db")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training_num", type=int, default=10)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--multi_obj", type=to_bool, default=False)
    parser.add_argument("--buffer_size", type=int, default=5e4)
    parser.add_argument("--linear", type=to_bool, default=False)
    parser.add_argument("--cat_num", type=int, default=1)
    parser.add_argument("--need_act_explain", type=bool, default=True)
    parser.add_argument("--need_obs_explain", type=bool, default=True)
    parser.add_argument("--llm", type=str, default="gpt2",
                        choices=["llama2-14b", "llama-14b",
                                 "llama3-8b", "llama2-7b", "llama-7b",
                                 "gpt2"])
    parser.add_argument("--policy_name", type=str, default="C51",
                        choices=["LLM-DQN", "LLM-DDQN", "LLM-C51", "LLM-discrete-SAC",
                                 "DQN", "DDQN", "DQN-rnn", "DDQN-rnn", "DQN-dueling", "DDQN-dueling",
                                 "C51", "C51-rnn", 
                                 "discrete-SAC", "discrete-SAC-rnn"])
    parser.add_argument("--scale_obs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_known_args()[0]

    return args

llm_dim_table = {
    "llama2-14b": {"llm_dim": 5120},
    "llama-14b": {"llm_dim": 5120},
    "llama3-8b": {"llm_dim": 4096},
    "llama2-7b": {"llm_dim": 4096},
    "llama-7b": {"llm_dim": 4096},
    "gpt2": {"llm_dim": 768}
}

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()

    if "rnn" in args.policy_name:
        use_rnn = True
    else:
        use_rnn = False

    llm_dim = llm_dim_table[args.llm]["llm_dim"]

    hparam_class = get_hparam_class(args.policy_name, offline=False)
    obj_class = get_obj_class(args.policy_name, offline=False)

    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    policy_type = get_policy_type(args.policy_name, offline=False)
    args.task += f"-{policy_type}-setting{args.setting}"
    study_name = f"{args.task}-{args.policy_name}"
    study_path = os.path.abspath(os.path.join(args.logdir, study_name)) + ".db"

    if args.sampler == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=50)
    else:
        sampler = optuna.samplers.BruteForceSampler(seed=args.seed)
    study = create_study_with_filter(study_name, study_path,
                                     direction=["maximize", "minimize"] if args.multi_obj else "maximize",
                                     sampler=sampler, load_if_exists=True, pruner=None)

    hparam_space = hparam_class(args.policy_name,
                                args.logdir,
                                args.seed,
                                args.training_num,  # number of training envs
                                args.test_num,  # number of test envs
                                args.epoch,
                                args.step_per_epoch,  # number of training steps per epoch
                                args.buffer_size,
                                use_rnn,
                                args.num_actions,
                                cat_num=args.cat_num,
                                linear=args.linear
                                )
    obj = obj_class(args.task, hparam_space, device=args.device, llm=args.llm, llm_dim=llm_dim, multi_obj=args.multi_obj,
                    logger="tensorboard",
                    )

    if args.sampler == "BruteForceSampler":
        n_trials = None
        TUNE_PARAMS = True
    else:
        # load trials
        try:
            trials_df = study.trials_dataframe()
            TUNE_PARAMS = len(trials_df[trials_df["state"] == "COMPLETE"]) < args.n_trials
            n_trials = args.n_trials - len(trials_df[trials_df["state"] == "COMPLETE"])
        except:
            TUNE_PARAMS = True
            n_trials = args.n_trials
    # start tuning
    if TUNE_PARAMS:
        try:
            study.optimize(obj, n_trials=n_trials, n_jobs=1, show_progress_bar=True,
                           callbacks=[early_stopping_callback])
        except optuna.exceptions.TrialPruned:
            pass
    else:
        print("Already finished hyperparameter tuning, jump to evaluation")
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)
