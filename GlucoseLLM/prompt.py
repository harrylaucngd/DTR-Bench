import re
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from datetime import timedelta
import re
from DTRGym.simglucose_env import SAMPLE_TIME, MAX_DOSAGE_U_per_hour
# todo: invalid rate, different prompt, reflexion, in-context medical knowledge
NUM_ACTION = 11

dosage_intervals = str(np.linspace(0, MAX_DOSAGE_U_per_hour, NUM_ACTION).tolist())

SYS_PROMPT = f"""
You are a clinical specialist responsible for managing patients with Type-1 Diabetes. Your primary objective is to maintain the patient's blood glucose levels within the safe range of **70-140 mg/dL** by administering appropriate insulin doses.

### Monitoring and Decision Frequency
Blood glucose levels are observed every **{SAMPLE_TIME} minutes**. Your task is to determine the insulin dose rate in units/hour every **{SAMPLE_TIME} minutes** based on the latest glucose readings and trends.


### Insulin Administration
- Insulin lowers blood glucose levels with a delayed effect. Your decisions will define the insulin dose rate to be administered over the next **{SAMPLE_TIME} minutes** to maintain a normal glucose level in a long run.
- We do not distinguish between basal and bolus insulin. The dosing decision is based on the latest glucose readings and trends.
- **Rate Range**: You MUST provide a dose in the range of [0 to {MAX_DOSAGE_U_per_hour}] units/hour, inclusive. 
- **Administration Interval**: The specified dose is distributed evenly over the {SAMPLE_TIME}-minute period, i.e., the total dosage is your_action/60*{SAMPLE_TIME}. You only need to provide the dose rate in units/hour. Do NOT specify the total dose.
"""

HIDDEN_VIARABLES = """
### Hidden Variables
- **Food Intake**: Food consumption increases blood glucose levels.
- **Exercise**: Exercise reduces blood glucose levels.
- **Estimation**: Since food intake and exercise are not directly observable, estimate based on time of day and observed glucose trends using clinical judgment and common sense.
"""

OPERATION_GUIDE = """
### Penalties and Risks
- **Blood Glucose Outside Safe Range (70-140 mg/dL)**:
  - **Above 140 mg/dL**: Hyperglycemia penalties.
  - **Below 70 mg/dL**: Hypoglycemia penalties, with increased severity.
  - **Above 500 or below 40 mg/dL**: EXTREMELY DANGEROUS, your treatment will be considered a failure and the patient will die!
- **Insulin Dose Considerations**:
  - **High Doses**: Use cautiously to avoid rapid and excessive lowering of glucose levels.
  - **Low Glucose Levels (<70 mg/dL)**:
    - **Action**: Immediately cease insulin administration until glucose levels rise above 70 mg/dL.
    - **Priority**: Prevent hypoglycemia due to its acute dangers.

### Safety Precautions
- **Avoid Overdosing Insulin**: Prevent hypoglycemia by carefully balancing insulin doses.
- **Insulin Stacking Awareness**: You should consider the accumulated dosage and the delayed effect of insulin on glucose levels carefully.
- **Prioritize Patient Safety**: Always aim to keep glucose levels within the target range. If uncertainty exists, opt for a lower or zero insulin dose to ensure safety.
"""

Q_PROMPT = """
Please predict the expected discounted reward (i.e., Q(s, a)) for each insulin actions in the order of 
the following dosage for the current {SAMPLE_TIME}-minute interval: {dosage_intervals}.
"""


Q_RANKING_PROMPT = f"Please rank the insulin dosage bins {dosage_intervals} in the descending order of your preference to maintain a patient's blood glucose levels within 70-140 mg/dL."

DIRECT_ACT_PROMPT = f"""Determine the optimal insulin rate for the current {SAMPLE_TIME}-minute interval to maintain a patient's blood glucose levels within the safe range of 70-140 mg/dL. Choose a dosage value. For example, if you choose 0 units, enter 0. DO NOT say anything else.
"""

COT_ACT_PROMPT = f"""
Determine the optimal insulin rate for the current {SAMPLE_TIME}-minute interval to maintain a patient's blood glucose levels within the safe range of 70-140 mg/dL with analysis. Finally, you must choose a dosage value enclosed in answer tags (i.e., <ans> and </ans>]), for example, <ans>0</ans>, without any non-numerical word. Let's think step by step.
"""


SUMMARY_PROMPT = """Please summarize history glucose record and drug usage. Your answer should base on facts and be concise.
Extract as much information as possible while keeping the answer brief. Let's think step by step."""


def get_text_obs(batch, sample_time: int = SAMPLE_TIME) -> List[str]:
    """
    Convert observational data into a textual format with strict "initial measurement" logic.
    Raise an error if mixed padding is detected in the observations.
    :param batch: Batch data containing obs and info.
    :param sample_time: Time interval in minutes between samples.
    """
    obs = batch["obs"]
    batch_size = len(obs)
    length = obs.shape[1]

    def adjust_time(datetime_input, minutes_offset):
        """Adjust time by the given offset in minutes."""
        adjusted_time = datetime_input + timedelta(minutes=minutes_offset)
        time_str = adjusted_time.strftime("%H:%M:%S")
        return f"Day {adjusted_time.day}, Time: {time_str}"

    conversations = []
    for i in range(batch_size):
        observation_time = batch["info"]["time"][i]
        glucose_readings = obs[:, :, 0][i]
        insulin_rates = obs[:, :, 1][i]

        # Detect mixed padding only if padding appears after valid data
        has_valid_data_started = False
        for value in glucose_readings:
            if value == -1 and has_valid_data_started:
                raise ValueError(f"Mixed padding detected in batch entry {i + 1}.")
            if value != -1:
                has_valid_data_started = True

        description = []
        for j in range(length):
            glucose_value = glucose_readings[j]
            insulin_rate = insulin_rates[j]

            if glucose_value == -1:  # Skip padding
                continue

            # Check for initial measurement: all preceding values must be -1
            is_initial_measurement = all(value == -1 for value in glucose_readings[:j]) and j > 0

            time_offset = -(length - j - 1) * sample_time
            time_info = adjust_time(observation_time, time_offset)

            if is_initial_measurement:
                # Add the initial measurement tag if all preceding are -1
                description.append(
                    f"{time_info} (initial measurement), "
                    f"glucose: {glucose_value:.2f} mg/dL, "
                    f"insulin rate: {insulin_rate:.4f} unit/hour, "
                    f"insulin dose: {insulin_rate * sample_time/60:.2f} unit."
                )
            else:
                description.append(
                    f"{time_info}, glucose: {glucose_value:.2f} mg/dL, "
                    f"insulin rate: {insulin_rate:.4f} unit/hour, "
                    f"insulin dose: {insulin_rate * sample_time/60:.2f} unit."
                )

        conversations.append("\n".join(description))
    return conversations



def get_patient_info_text(batch):
    # todo： add patient info
    return ""


import re
import numpy as np


def text2act(logits: str, action_space):
    try:
        matches = re.findall(r"<ans>(.*?)</ans>", logits, re.DOTALL)
        matches = [match.replace("\n", "").replace(" ", "") for match in matches]

        if len(matches) == 1:
            # If exactly one match is found, try to convert it to a float
            action = float(matches[0])
        elif len(matches) == 0:
            # If no matches, assume the entire logits is a number

            action = float(logits)
        else:
            # If more than one match, return the last match if it can be converted to a float
            action = float(matches[-1])
        action = np.round(action, 4)
        action = np.clip(action, action_space.low[0], action_space.high[0])
        return action, True
    except (ValueError, IndexError):
        a = action_space.sample()
        return np.round(a, 4), False

