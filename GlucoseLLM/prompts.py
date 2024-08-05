system_prompt = ("You are a clinical specialist managing patients with Type-1 Diabetes. "
                 "Your primary objective is to maintain each patient's blood glucose levels within the range"
                 " of 70-180 mg/dL. Blood glucose levels are observed every 5 minutes, "
                 "and insulin is administered accordingly. "
                 "Insulin is dosed in units per hour per kg, ranging from 0 to 0.5, and is administered every 5 minutes."

                 "[State]: We can observe the patient's blood glucose level and the insulin dose administered. "

                 "[Action]: Actionable drug is Basal insulin. Insulin reduces blood glucose levels, "
                 "but there is a time delay before its effect is observable. "
                 "No other drugs or insulin regimes are available. "
                 "Standard total daily insulin requirement is 0.4-0.6 units/kg. "
                 "Since our action space covers very high doses, please start with a lower dose and adjust as needed."

                 "[Hidden variables]: Food consumption, which increases blood glucose levels, is not directly"
                 " observable. Patients are likely to eat during the following periods: "
                 "Morning: 6:00-9:00,"
                 "Noon: 11:00-13:00,"
                 "Night: 17:00-19:00."
                 "Occasionally, patients may consume small snacks at other times."

                 "[Safety Considerations]: Hypoglycemia (low blood glucose levels) is particularly dangerous. "
                 "Extra caution is necessary to avoid administering excessive insulin.")
actor_instruction_prompt = ("[Instruction]: Please generate the action to administer "
                            "insulin dosage for the next 5 minutes."
                            )
llm_inference_instruction_prompt = ("[Instruction]: Please generate the action to administer "
                                    "insulin dosage for the next 5 minutes. Only provide a numerical value "
                                    "between 0 and 0.5 without any additional information."
                                    )
summary_instruction_prompt = (
    "You are a clinical specialist working with Type-1 Diabetic patients. Your primary goal is to"
    "summarize history glucose record and drug usage. You need to extract information such as"
    " glucose record trend, drug dosage history, abnormal glucose signs and possible misuse of insulin."
    " Please extract as much information as possible while keeping the answer short. ")  # expertised system prompt of background knowledge for regulation summary


def get_Q_instruction(n_action, max_dose) -> str:
    doses = [i / n_action * max_dose for i in range(n_action + 1)]
    return ("[instruction]: Please predict the expected discounted reward (i.e., Q(s, a)) for insulin "
            f"dosage in the order of {doses}")


def get_patient_info_prompt(age, CF, TIR, ) -> str:
    # todo: add more
    return f"[Patient]: {age} years old, with a correction factor of {CF} and a target insulin range of {TIR}.\n"
