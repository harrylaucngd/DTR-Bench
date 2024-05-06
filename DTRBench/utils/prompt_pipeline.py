def act_prompt_reprogramming(obs, act, act_exp):
    history_prompt = ""
    if (act_exp == []) or all(exp == "" for exp in act_exp):    # first round or self.need_act_explain = False
        for i, (o, a) in enumerate(zip(obs, act)):
            history_prompt += f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}. "
    else:
        for i, (o, a, exp) in enumerate(zip(obs, act, act_exp)):
            history_prompt += f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}. Here's the explanation for the action: {exp}. "
    return history_prompt

def obs_prompt_reprogramming(obs, act, obs_exp):
    history_prompt = ""
    if (obs_exp == []) or all(exp == "" for exp in obs_exp):    # first round or self.need_obs_explain = False
        for i, (o, a) in enumerate(zip(obs, act)):
            history_prompt += f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}, The next blood glucose observation is {obs[i+1]}(unit). "
    else:
        for i, (o, a, exp) in enumerate(zip(obs, act, obs_exp)):
            history_prompt += f"In timestep {i}, The blood glucose observation is {o}(unit), the agent takes action {a}, The next blood glucose observation is {obs[i+1]}(unit). Here's the explanation for the state: {exp}. "
    return history_prompt

def q_prompt_reprogramming(obs, act, act_explain, obs_explain):
    history_prompt = ""

    return history_prompt