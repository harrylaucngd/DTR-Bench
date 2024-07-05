common_hparams = {
    "seed": [6311, 6890, 663, 4242, 8376],
    "lr": [0.001, 0.0005, 0.0001, 0.00005],
    "batch_size": [128, 256],
    "obs_mode": [{"stack": {"stack_num": 24,
                            "cat_num": 1},
                  "cat": {"stack_num": 1,
                          "cat_num": 24}, }],
    "batch_norm": False,
    "dropout": 0,
    "target_update_freq": 1000,
    "update_per_step": [0.1, 0.5],
    "update_actor_freq": [1, 5],
    "step_per_collect": [500],
    "n_step": 1,
    "start_timesteps": 0,
    "gamma": 0.99,
    "tau": 0.001,
    "exploration_noise": [0.1, 0.2, 0.5],

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
