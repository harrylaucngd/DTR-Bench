import torch
from GlucoseLLM.LLM_policy import LLM_DQN_Policy
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from GlucoseLLM.models.llm_net import define_llm_network
from DTRBench.src.RLObj import DQNObjective


class LLM_DQN_Objective(DQNObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma, lr,
                      # dqn hp
                      n_step, target_update_freq,
                      # llm prompt
                      llm, llm_dim, need_obs_explain, need_act_explain, need_summary, exp_freq,
                      *args, **kwargs
                      ):
        # define model
        net = define_llm_network(self.state_shape, self.action_shape,  # Changing to GlucoseLLM
                                 device=self.device, llm=llm, llm_dim=llm_dim)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        # define policy
        policy = LLM_DQN_Policy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            need_obs_explain=need_obs_explain,
            need_act_explain=need_act_explain,
            need_summary=need_summary,
            exp_freq=exp_freq,
            action_space=self.action_space,
            observation_space=self.state_space,
        )
        return policy
