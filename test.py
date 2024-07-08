import torch
import numpy as np
from tianshou.data import Batch
from torch.optim import Adam
from DTRBench.utils.network import LLMNet
from DTRBench.src.offpolicyRLObj import define_llm_network
from DTRBench.src.discrete_policy.LLM_DQN import LLM_DQN_Policy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def define_policy(  # general hp
                    gamma,
                    lr,

                    # dqn hp
                    n_step,
                    target_update_freq,
                    **kwargs
                    ):
    # define model
    net = define_llm_network(1, 5, device="cuda" if torch.cuda.is_available() else "mps", 
                             llm="Qwen2-1.5B-Instruct", llm_dim=1536)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    # define policy
    policy = LLM_DQN_Policy(
        net,
        optim,
        gamma,
        n_step,
        target_update_freq=target_update_freq,
        need_obs_explain = True,
        need_act_explain = True,
        need_summary = True,
        exp_freq = 1,
    )
    return policy

policy = define_policy(0.9, 0.001, 1, 1)

# Generate synthetic observations and actions
synthetic_obs = np.random.uniform(100, 200, 1)
synthetic_batch = Batch(obs=synthetic_obs)

# Test the forward function
output = policy.forward(synthetic_batch)

# Print the results
print("Logits:", output.logits)
print("Actions:", output.act)
print("State:", output.state)