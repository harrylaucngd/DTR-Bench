import numpy as np
from datetime import timedelta
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from typing import Union


SYSTEM_PROMPT = ("You are a clinical specialist managing patients with Type-1 Diabetes. "
                 "Your primary objective is to maintain each patient's blood glucose levels within the range "
                 "of 70-180 mg/dL. Blood glucose levels are observed every 5 minutes, "
                 "and insulin is administered accordingly. "
                 "Insulin is dosed in U/min, ranging from 0 to 0.5, and is adjusted per 5 minutes. "

                 "[State]: We can observe the patient's blood glucose level and the insulin dose administered. "

                 "[Action]: Actionable drug is Basal insulin. Insulin reduces blood glucose levels, "
                 "but there is a time delay before its effect is observable. "
                 "No other drugs or insulin regimes are available. "
                 "Standard total daily insulin requirement is 0.4-0.6 units/kg. The patient's weight is not provided."

                 "[Hidden variables]: Food consumption, which increases blood glucose levels, is not directly "
                 "observable. Patients are likely to eat during the following periods: "
                 "Morning: 6:00-9:00, "
                 "Noon: 11:00-13:00, "
                 "Night: 17:00-19:00. "
                 "Occasionally, patients may consume small snacks at other times. "

                 "[Safety Considerations]: Hypoglycemia (low blood glucose levels) is particularly dangerous. "
                 "Extra caution is necessary to avoid administering excessive insulin. Insulin has a long half-life, "
                 "so the effects of previous doses may still be present. Pay attention to the accumulated insulin dose "
                 "to prevent Hypoglycemia.")
ACTOR_INSTRUCTION_PROMPT = ("[Instruction]: Please generate the insulin dosage rate in U/min "
                            "for the next 5 minutes. "
                            )

LLM_INFERENCE_INSTRUCTION_PROMPT = ("[Instruction]: Please generate the insulin dosage rate in U/min "
                                    "for the next 5 minutes. Only provide a numerical value "
                                    "between 0 and 0.5 without any additional information."
                                    )
LLM_INFERENCE_RETRY_PROMPT = ("Your previous answer cannot be converted to a valid action. "
                              "[Instruction]: Please provide a numerical value between 0 and 0.5 "
                              "without any additional information.")
SUMMARY_INSTRUCTION_PROMPT = (
    "[Instruction]: PLease summarize information such as indications of food intake, patient's response to insulin,"
    "glucose record trend, drug dosage history, abnormal glucose signs and possible misuse of insulin. "
    "Summarize as much information as possible while keeping the answer short.")


def obs2text(batch: Union[Batch, BatchProtocol]) -> str:
    obs = batch.obs
    length = obs.shape[0]
    time = batch.info["time"]
    glucose = obs[:, 0]
    insulin = obs[:, 1]

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
    doses = [round(i / (n_action - 1) * max_dose, 3) for i in range(n_action)]
    return ("[instruction]: Please predict the expected discounted reward (i.e., Q(s, a)) for insulin "
            f"dosage in the order of {doses} without any additional information.")


def get_patient_info_prompt(age, CF, TIR, ) -> str:
    # todo: add more
    META_PROMPT = ("CR was calculated through a simulation where each subject received 50 g of CHO from their basal "
                   "level. The optimal insulin bolus was determined based on these criteria: (1) glucose concentration "
                   "3 hours post-meal is between 85% and 110% of the basal level; (2) minimum glucose concentration "
                   "stays above 90 mg/dl; (3) maximum glucose concentration is 40-80 mg/dl above the basal level. CR is "
                   "then the ratio of ingested CHO to the optimal insulin bolus (CR = ingestedCHO/optimalbolus). CF was "
                   "determined using the 1700 rule (CF = 1700/TDI), where TDI is the total daily insulin, calculated "
                   "using the optimal CR and basal infusion rate, assuming an average diet of 180 g of CHO for "
                   "adolescents and adults, and 135 g for children.")
    return f"[Patient]: {age} years old, with a correction factor of {CF} and a target insulin range of {TIR}.\n"
