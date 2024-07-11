common_hparams = {
    "seed": [6311, 6890, 663, 4242, 8376],
    "lr": [1e-3, 1e-4, 1e-5],
    "batch_size": [32, 256],
    # ATTENTION: cat_num must be 1 since tianshou does not support it for env interaction! Please don't use cat_num>1
    "obs_mode": [{"stack": {"stack_num": 24,
                            "cat_num": 1}},
                 {"cur": {"stack_num": 1,
                          "cat_num": 1}}],
    "batch_norm": False,
    "dropout": 0,
    "target_update_freq": 3000, #
    "update_per_step": [0.1, 1],
    "update_actor_freq": [1, 5],
    "step_per_collect": [6, 6*12],
    "n_step": 1,
    "start_timesteps": 0,
    "gamma": 0.99,
    "tau": 0.001,
    "exploration_noise": [0.1, 0.5],

    # epsilon-greedy exploration
    "eps_train": 0.99,
    "eps_train_final": 0.005,
    "eps_test": 1e-9,
}


def get_common_hparams(use_rnn):
    hp = common_hparams.copy()
    if not use_rnn:
        hp["stack_num"] = 1
    return hp
