import numpy as np
from datetime import timedelta
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from typing import Union


SYSTEM_PROMPT = ("You are a clinical specialist managing patients with Type-1 Diabetes. "
                 "Your primary objective is to maintain each patient's blood glucose levels within the range "
                 "of 70-180 mg/dL. Blood glucose levels are observed every 5 minutes, "
                 "and insulin is administered accordingly. "
                 "Insulin is dosed in units per hour per kg, ranging from 0 to 0.5, and may be administered per 5 minutes. "

                 "[State]: We can observe the patient's blood glucose level and the insulin dose administered. "

                 "[Action]: Actionable drug is Basal insulin. Insulin reduces blood glucose levels, "
                 "but there is a time delay before its effect is observable. "
                 "No other drugs or insulin regimes are available. "
                 "Standard total daily insulin requirement is 0.4-0.6 units/kg. "
                 "Since our action space covers very high doses, please start with a lower dose (0-0.1) and adjust as needed. "

                 "[Hidden variables]: Food consumption, which increases blood glucose levels, is not directly "
                 "observable. Patients are likely to eat during the following periods: "
                 "Morning: 6:00-9:00, "
                 "Noon: 11:00-13:00, "
                 "Night: 17:00-19:00. "
                 "Occasionally, patients may consume small snacks at other times. "

                 "[Safety Considerations]: Hypoglycemia (low blood glucose levels) is particularly dangerous. "
                 "Extra caution is necessary to avoid administering excessive insulin. ")
ACTOR_INSTRUCTION_PROMPT = ("[Instruction]: Please generate the action to administer "
                            "insulin dosage for the next 5 minutes. "
                            )
LLM_INFERENCE_INSTRUCTION_PROMPT = ("[Instruction]: Please generate the action to administer "
                                    "insulin dosage for the next 5 minutes. Only provide a numerical value "
                                    "between 0 and 0.5 without any additional information."
                                    )
LLM_INFERENCE_RETRY_PROMPT = ("Your previous answer cannot be converted to a valid action. "
                              "[Instruction]: Please provide a numerical value between 0 and 0.5 "
                              "without any additional information.")
SUMMARY_INSTRUCTION_PROMPT = (
    "[Instruction]: PLease summarize information such as indications of food intake, "
    "glucose record trend, drug dosage history, abnormal glucose signs and possible misuse of insulin. "
    "Summarize as much information as possible while keeping the answer short.")


def obs2text(batch: Union[Batch, BatchProtocol]) -> str:
    obs = batch.obs
    length = obs.shape[1]
    time = batch.info["time"][0]
    glucose = obs[0, :, 0]
    insulin = obs[0, :, 1]

    def adjust_time(datetime_input, min):
        adjusted_time = datetime_input + timedelta(minutes=min)
        day_number = adjusted_time.day
        return adjusted_time.strftime(f"day{day_number} %H:%M:%S")

    descriptions = []
    for i in range(length):
        if glucose[i] == -1:
            continue
        if i == 0:
            descriptions.append(f"Time:{adjust_time(time, -(length-1)*5)}, insulin:{insulin[0]:.3f}; ")
        if i < length - 1:
            descriptions.append(f"Time:{adjust_time(time, -(length-i-1)*5)}, glucose:{glucose[i]:.1f}, insulin:{insulin[i]:.3f}; ")
        else:
            descriptions.append(f"Current time: {adjust_time(time, 0)}, glucose:{glucose[i]:.1f}, insulin: TBD. ")
    assert descriptions
    return " ".join(descriptions)


def get_Q_instruction(n_action, max_dose) -> str:
    doses = [i / n_action * max_dose for i in range(n_action + 1)]
    return ("[instruction]: Please predict the expected discounted reward (i.e., Q(s, a)) for insulin "
            f"dosage in the order of {doses}")


def get_patient_info_prompt(age, CF, TIR, ) -> str:
    # todo: add more
    return f"[Patient]: {age} years old, with a correction factor of {CF} and a target insulin range of {TIR}.\n"
