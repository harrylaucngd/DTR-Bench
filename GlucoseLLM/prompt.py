import re
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from datetime import timedelta
import re


SYS_PROMPT = """
You are a clinical specialist managing Type-1 Diabetic patients. Your goal is to regulate a patient's blood glucose levels (observed every 5 minutes) within the safe range of 70-140 mg/dL through appropriate insulin administration. 

- **Insulin Action**: Insulin lowers blood glucose, and your decisions will specify the insulin dose (in units per 5 minutes, ranging from 0 to 0.1 units/min, equivalent to a maximum of 6 units/hour).
- **Hidden Variable**: Food intake, which increases blood glucose levels, is not directly observable.
- **Penalties**: 
  - Blood glucose levels outside the 70-140 mg/dL range will incur penalties.
  - High insulin doses should be used with extra caution.
  - Low glucose levels (<70 mg/dL) are much more dangerous and should be completely avoided. When glucose levels fall below 70 mg/dL, stop using insulin immediately until its glucose level rises above 70 mg/dL.
- **Caution**: Avoid overdosing insulin to prevent hypoglycemia. Excessively high doses of insulin can rapidly lower blood glucose to dangerous levels. Prioritise patient safety by maintaining glucose levels within the target range. If in doubt, it is safer to administer a lower or zero insulin dose.

Your objective is to determine the optimal insulin dose every 5 minutes based on the current glucose level, balancing penalties and risks.
"""

Q_PROMPT = """
Please predict the expected discounted reward (i.e., Q(s, a)) for each insulin actions in the order of 
the following dosage for the current 5 minute interval: ['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1'].
"""


Q_RANKING_PROMPT = "Please rank the insulin dosage bins ['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1'] in the descending order of your preference to maintain a patient's blood glucose levels within 70-140 mg/dL. "

ACT_PROMPT = """Determine the optimal insulin dosage for the current 5-minute interval to maintain a patient's blood glucose levels within the safe range of 70-140 mg/dL. First, provide a short analysis. Then, choose a dosage value between 0 and 0.1 enclosed in square brackets. For example, if you choose 0 units/min, enter [0].
"""


SUMMARY_PROMPT = """Please summarize history glucose record and drug usage. Your answer should base on facts and be concise.
Extract as much information as possible while keeping the answer brief. Let's think step by step."""


def get_text_obs(batch) -> List[str]:
    """
    Convert insulin to absolute value (unit)
    """
    minutes = 5
    obs = batch.obs
    batch_size = len(obs)
    length = obs.shape[1]

    def adjust_time(datetime_input, min):
        adjusted_time = datetime_input + timedelta(minutes=min)
        time_str = adjusted_time.strftime("%H:%M:%S")  # Trim to 3 decimal places
        # Combine into desired format
        return f"Day {adjusted_time.day}, Time: {time_str}"

    conversations = []
    for i in range(batch_size):
        time = batch.info["time"][i]
        glucose = obs[:, :, 0][i]
        insulin = obs[:, :, 1][i]

        initial_sign = ""
        description = []
        for j in range(length):
            if glucose[j] == -1:
                # initial_sign = " (initial measurement)"
                continue
            if j == 0:
                description.append(f"{adjust_time(time, -(length-1)*minutes)}{initial_sign}, Insulin dose: {insulin[0] * minutes}.")
            else:
                description.append(
                    f"{adjust_time(time, -(length-j-1)*minutes)}{initial_sign}, glucose:{glucose[j]:.2f}mg/dL, insulin:{insulin[j]* minutes}."
                )
        description = "\n".join(description)
        conversations.append(description)
    return conversations


def get_patient_info_text(batch):
    # todo： add patient info
    return ""


import re
import numpy as np


def text2act(logits: str, action_space):
    try:
        matches = re.findall(r"\[(.*?)\]", logits)

        if len(matches) == 1:
            # If exactly one match is found, try to convert it to a float
            action = float(matches[0])
        elif len(matches) == 0:
            # If no matches, assume the entire logits is a number
            action = float(logits)
        else:
            # If more than one match, return a sample
            return action_space.sample()

        action = np.clip(action, action_space.low[0], action_space.high[0])
    except (ValueError, IndexError):
        return action_space.sample()

    return action
