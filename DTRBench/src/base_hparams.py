# seed is generated by the following code:
# import numpy as np
# np.random.seed(0)
# print(np.random.randint(0, 10000, 10))

common_hparams = {
    "seed": [2732, 9845, 3264, 4859],
    "lr": [3e-3, 1e-3, 3e-4],
    "batch_size": 256,
    "obs_mode": ["cat", "stack"],

    "batch_norm": False,
    "dropout": 0,
    "target_update_freq": [50, 200],  #
    "update_per_step": 1,  # off-policy only
    "update_actor_freq": 1,
    "step_per_collect": [1, 100],  # off-policy only
    "onpolicy_step_per_collect": 288,  # for on-policy only
    "repeat_per_collect": 20,  # for on-policy only
    "n_step": 1,
    "start_timesteps": 1000,
    "gamma": 0.99,
    "tau": 0.005,
    "exploration_noise": 0.1,

    # epsilon-greedy exploration
    "eps_train": 0.1,
    "eps_train_final": 0.1,
    "eps_test": 0.001,

    # LLM generation length
    "max_new_tokens": 512,
}


def get_common_hparams(use_rnn):
    hp = common_hparams.copy()
    if not use_rnn:
        hp["stack_num"] = 1
    return hp
