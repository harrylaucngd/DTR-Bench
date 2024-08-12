import torch
import wandb
from pathlib import Path
from GlucoseLLM.LLM_policy import LLM_DQN_Policy, LLMInference_Policy
from GlucoseLLM.LLM_hparams import LLMInference_HyperParams
from GlucoseLLM.model.net import LLMInference
from GlucoseLLM.model.net import timeLLM
from DTRBench.src.offpolicyRLHparams import OffPolicyRLHyperParameterSpace
from DTRBench.src.RLObj import DQNObjective, PPOObjective
from DTRBench.src.base_obj import RLObjective
from DTRBench.utils.wandb import WandbLogger
from DTRBench.utils.misc import set_global_seed


class LLM_DQN_Objective(DQNObjective):
    def __init__(self, env_name, env_args, hparam_space: OffPolicyRLHyperParameterSpace, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device, **kwargs)

    def define_policy(self,
                      # general hp
                      gamma, lr,
                      # dqn hp
                      n_step, target_update_freq, is_double,
                      # llm prompt
                      llm_mode, llm_modal, summary_prob, gradient_accumulation,
                      *args, **kwargs
                      ):
        # define model
        net = timeLLM(llm=llm_mode["llm"], n_vars=self.state_shape, output_dim=self.action_shape,
                      seq_len=12, d_model=16, max_new_tokens=512,
                      d_ff=32, patch_len=6, stride=3, token_dim=llm_mode["token_dim"], n_heads=8,
                      decoder_len=1,
                      keep_old=True, dropout=0.,
                      model_dir=Path(__file__).resolve().parent.absolute() / "model" / "model_hub",
                      device=self.device).to(self.device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)

        # define policy
        policy = LLM_DQN_Policy(
            net,
            optim,
            gamma,
            n_step,
            target_update_freq=target_update_freq,
            is_double=is_double,
            action_space=self.action_space,
            observation_space=self.state_space,
            # llm hparam
            llm_modal=llm_modal,
            summary_prob=summary_prob,
            gradient_accumulation=gradient_accumulation
        )
        return policy


class LLM_PPO_Objective(PPOObjective):
    pass


class LLM_Inference_Objective(RLObjective):
    def __init__(self, env_name, env_args, hparam_space: LLMInference_HyperParams, device, **kwargs):
        super().__init__(env_name, env_args, hparam_space, device=device, **kwargs)

    def define_policy(self, llm_mode, num_try, need_summary, need_meta_info, **kwargs):
        net = LLMInference(llm=llm_mode["llm"], context_window=llm_mode["context_window"],
                           device=self.device,
                           model_dir=Path(__file__).resolve().parent.absolute() / "model" / "model_hub").to(self.device)
        return LLMInference_Policy(
            net,
            action_space=self.action_space,
            observation_space=self.state_space,
            num_try=num_try,
            need_summary=need_summary,
            need_meta_info=need_meta_info
        )

    def wandb_search(self):
        self.logger = WandbLogger(train_interval=24 * 15)
        self.meta_param["training_num"] = 1
        self.meta_param["num_actions"] = None
        hparams = wandb.config

        self.prepare_env(int(hparams["seed"]), self.env_name, **self.env_args)
        set_global_seed(int(hparams["seed"]))

        # start training
        print("prepare policy")
        self.policy = self.define_policy(**{**hparams, **self.meta_param})

        # test on all envs
        self.test_all_patients(self.policy, None, int(hparams["seed"]), self.logger, n_episode=20)
