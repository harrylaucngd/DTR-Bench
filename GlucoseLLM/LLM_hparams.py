import DTRBench.src.offpolicyRLHparams as offpolicyRLHparams
from DTRBench.src.offpolicyRLHparams import common_hparams

'''
Open LLM Leaderboard Top 3 Average Performance Model under 10B (2024.7.7):
1. internlm/internlm2_5-7b-chat
2. microsoft/Phi-3-small-128k-instruct
3. 01-ai/Yi-1.5-9b-Chat
Open LLM Leaderboard Top 1 Average Performance Model under 1B (2024.7.7):
1. Qwen/Qwen2-1.5B-Instruct
'''

token_dim_table = {
    "internlm2_5-7b-chat": {"llm_dim": 4096},
    "Phi-3-small-128k-instruct": {"llm_dim": 4096},
    "Yi-1.5-9b-Chat": {"llm_dim": 4096},
    "Qwen2-1.5B-Instruct": {"llm_dim": 1536},
    "llama-2-13b": {"llm_dim": 5120},
    "llama-13b": {"llm_dim": 5120},
    "llama-3-8b": {"llm_dim": 4096},
    "llama-2-7b": {"llm_dim": 4096},
    "llama-7b": {"llm_dim": 4096},
    "gpt2": {"llm_dim": 768}
}

obs_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. "
                  "The primary goal in the Simglucose environment is to maintain a patient's blood glucose levels "
                  "(the observation) within a target range through the administration of insulin (the action). "
                  "The reason for a high value observation (high Blood Glucose Level (BG): the current blood glucose "
                  "concentration in mg/dL) is typically in last/last several timestep, more insulin (action) was "
                  "injected, a raising or high level of action is witnessed, and vice versa. The following would be "
                  "conversation history between user and you as an assistant, where the user gives blood glucose "
                  "observation and the agent's action (Insulin Bolus Dose) at every timestep, and next timestep's "
                  "observation, then the assistant give explaination to the observation at every next timestep.")  # expertised system prompt of background knowledge for observation explanation

Q_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics of "
            "glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
            "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a target "
            "range through the administration of insulin (the action). 5 number of actions (Insulin Bolus Dose) "
            "represents 5 degrees of insulin injection to restrain high blood glucose level. In Q-learning, "
            "the Q-value represents the expected future rewards for taking a given action in a given state, "
            "with high Q-values indicating more favorable actions and low Q-values indicating less favorable actions. "
            "So for a q-learning agent, if the blood glucose level is observed to be high, the q value of the high "
            "value action should be high, and q value of the low value action should be low, and vice versa for low "
            "blood glucose level. The following would be conversation history between user and you as an assistant, "
            "where the user gives blood glucose observation and the agent's action (Insulin Bolus Dose) at every timestep, "
            "then the assistant give explaination to both the observation and action at every timestep.")  # expertised system prompt for series information description and Q value prediction

act_exp_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa. The following would be conversation "
                  "history between user and you as an assistant, where the user gives blood glucose observation and "
                  "the agent's action (Insulin Bolus Dose) at every timestep, then the assistant give explaination to the "
                  "action at every timestep.")  # expertised system prompt of background knowledge for action explanation
summary_prompt = ("The Simglucose environment is a simulation environment designed to mimic the physiological dynamics "
                  "of glucose metabolism in humans, often used in research of glucose control. The primary goal in the "
                  "Simglucose environment is to maintain a patient's blood glucose levels (the observation) within a "
                  "target range through the administration of insulin (the action). The reason for a high value action "
                  "(high Insulin Bolus Dose measured in units (U) of insulin) is typically in current timestep or the "
                  "past several timesteps, a relatively high value of Blood Glucose Level (BG): the current blood "
                  "glucose concentration in mg/dL is observed (low observation), thus the patient needs more insulin "
                  "to prevent the blood glucose from getting too high, and vice versa. The reason for a high value "
                  "observation (high Blood Glucose Level (BG): the current blood glucose concentration in mg/dL) is "
                  "typically in last/last several timestep, more insulin (action) was injected, a raising or high level "
                  "of action is witnessed, and vice versa.")  # expertised system prompt of background knowledge for regulation summary


class LLM_DQN_HyperParams(offpolicyRLHparams.DQNHyperParams):
    # "llama-2-13b", "llama-13b",
    # "llama-3-8b", "llama-2-7b", "llama-7b",
    # "gpt2"
    _policy_hparams = {
        "lr": common_hparams["lr"],  # learning rate
        "stack_num": common_hparams["stack_num"],
        "cat_num": common_hparams["cat_num"],
        "eps_test": common_hparams["eps_test"],
        "eps_train": common_hparams["eps_test"],
        "eps_train_final": 0.005,
        "n_step": common_hparams["n_step"],
        "target_update_freq": common_hparams["target_update_freq"],
        "is_double": False,
        "use_dueling": False,
        "obs_exp_prompt": obs_exp_prompt,
        "Q_prompt": Q_prompt,
        "act_exp_prompt": act_exp_prompt,
        "summary_prompt": summary_prompt,
    }
