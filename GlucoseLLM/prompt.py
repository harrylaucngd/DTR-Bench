import re
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from datetime import timedelta
import re

# todo: invalid rate, different prompt, reflexion, in-context medical knowledge

SYS_PROMPT = """
You are a clinical specialist responsible for managing patients with Type-1 Diabetes. Your primary objective is to maintain the patient's blood glucose levels within the safe range of **70-140 mg/dL** by administering appropriate insulin doses.

### Monitoring and Decision Frequency
- **Glucose Monitoring**: Blood glucose levels are observed every **30 minutes**.
- **Insulin Dosing Decisions**: Determine and specify the insulin dose every **30 minutes** based on the latest glucose readings and trends.

### Insulin Administration
- **Insulin Action**: Insulin lowers blood glucose levels. Your decisions will define the total insulin dose to be administered over the next **30 minutes**.
- **Dosing Parameters**:
  - **Dose Range**: 0 to 3 units.
  - **Administration Rate**: The specified dose is distributed evenly over the 30-minute period.
    - *Example*: A dose of **1.5 units** equates to **0.05 units per minute** for the next half-hour.

### Hidden Variables
- **Food Intake**: 
  - **Impact**: Food consumption increases blood glucose levels.
  - **Estimation**: Since food intake is not directly observable, estimate based on time of day and observed glucose trends using clinical judgment and common sense.

### Penalties and Risks
- **Blood Glucose Outside Safe Range (70-140 mg/dL)**:
  - **Above 140 mg/dL**: Hyperglycemia penalties.
  - **Below 70 mg/dL**: Hypoglycemia penalties, with increased severity for levels <70 mg/dL.
- **Insulin Dose Considerations**:
  - **High Doses**: Use cautiously to avoid rapid and excessive lowering of glucose levels.
  - **Low Glucose Levels (<70 mg/dL)**:
    - **Action**: Immediately cease insulin administration until glucose levels rise above 70 mg/dL.
    - **Priority**: Prevent hypoglycemia due to its acute dangers.

### Safety Precautions
- **Avoid Overdosing Insulin**: Prevent hypoglycemia by carefully balancing insulin doses.
- **Prioritize Patient Safety**: Always aim to keep glucose levels within the target range. If uncertainty exists, opt for a lower or zero insulin dose to ensure safety.
"""

Q_PROMPT = """
Please predict the expected discounted reward (i.e., Q(s, a)) for each insulin actions in the order of 
the following dosage for the current half-hour interval: ['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1'].
"""


Q_RANKING_PROMPT = "Please rank the insulin dosage bins ['0', '0.03', '0.06', '0.09', '0.12', '0.15', '0.18', '0.21', '0.24', '0.27', '0.3'] in the descending order of your preference to maintain a patient's blood glucose levels within 70-140 mg/dL."

ACT_PROMPT = """Determine the optimal insulin dosage for the current 30-min interval to maintain a patient's blood glucose levels within the safe range of 70-140 mg/dL. First, provide a short analysis. Then, choose a dosage value enclosed in square brackets. For example, if you choose 0 units, enter [0].
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
