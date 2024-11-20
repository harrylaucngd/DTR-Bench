import re
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from datetime import timedelta


SYS_PROMPT = (
    "You are a clinical specialist working with Type-1 Diabetic patients. Your primary goal is to "
    "maintain a patient's blood glucose levels (the observation, received every 5 minutes) within "
    "70-140 mg/dL through the administration of insulin (the action). Insulin will reduce blood "
    "glucose levels, while food intake, which is hidden, will increase blood glucose levels. You will "
    "be penalized for blood glucose <70 or >140, and high insulin doses. Notably, low blood glucose "
    "levels are much more dangerous. You should take caution to avoid overdosing insulin, thus "
    "to avoid hypoglycemia. The insulin is given per 5 minutes and given in units, ranging from 0 to 2.5."
)

Q_PROMPT = (
    "Please predict the expected discounted reward (i.e., Q(s, a)) for each insulin bins in the order of "
    "the following insulin dosage bins for the current 5 minute interval: ['0', '0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1', "
    "'1-1.25', '1.25-1.5', '1.5-1.75', '1.75-2', '2-2.25', '2.25-2.5']."
)  # expertised system prompt for series information description and Q value prediction

Q_RANKING_PROMPT = (
    "Please rank the insulin dosage bins ['0', '0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1', "
    "'1-1.25', '1.25-1.5', '1.5-1.75', '1.75-2', '2-2.25', '2.25-2.5'] in the descending order of your preference to maintain a patient's blood glucose levels within 70-140 mg/dL. "
)

ACT_PROMPT = "What is the optimal insulin dosage for the current 5 minute interval to maintain a patient's blood glucose levels within 70-140 mg/dL? Generate a value between 0 and 2.5 without anything else."  # expertised system prompt for series information description and action prediction


SUMMARY_PROMPT = (
    "Please summarize history glucose record and drug usage. Your answer should base on facts and be concise. "
    "Extract as much information as possible while keeping the answer brief."
)  # expertised system prompt of background knowledge for regulation summary


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
                initial_sign = " (initial measurement)"
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


def text2act(logits, action_space):
    numbers = re.findall(r"-?\d+\.?\d*", logits)  # todo: make sure it select the correct number with 0.
    numbers = [float(num) for num in numbers]

    if len(numbers) == 0:
        return action_space.sample()

    # always select the first number
    return np.clip(numbers[0], 0, 0.5)
