common_hparams = {
    "seed": [6311, 6890, 663, 4242, 8376],
    "lr": [1e-3, 1e-4],
    "batch_size": 256,
    "obs_mode": [{"stack": {"stack_num": 24,
                            "cat_num": 1}},
                 {"cur": {"stack_num": 1,
                          "cat_num": 1}}],
    "batch_norm": False,
    "dropout": 0,
    "target_update_freq": 500, #
    "update_per_step": 1,  # off-policy only
    "update_actor_freq": 1,
    "step_per_collect": 1,  # off-policy only
    "onpolicy_step_per_collect": [1024, 2048], # for on-policy only
    "repeat_per_collect": [10, 20], # for on-policy only
    "n_step": 1,
    "start_timesteps": 0,
    "gamma": 0.99,
    "tau": 0.001,
    "exploration_noise": [0.1, 0.5],

    # epsilon-greedy exploration
    "eps_train": 0.99,
    "eps_train_final": 0.005,
    "eps_test": 0.001,
}



def get_common_hparams(use_rnn):
    hp = common_hparams.copy()
    if not use_rnn:
        hp["stack_num"] = 1
    return hp
