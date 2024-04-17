# Abstract
1. Use LLM as a (part of) feature extracter for **prior knowledge embedding**.
2. The model is single-GPU trainable, and if necessary, a 8*H800 server is available to be used for maximal training utility.
3. It should generalize to unseen patient, improve sampling efficiency and
avoid unsafe exploration.
4. It can provide some positive examples in which the LLM gives interpretations for RL algorithm's decision-making as an usefull supplement for current treatment regime.

# Environment&Repo
1. **tianshou** (a modified version to support data parallel)
2. **simglucose** (an modified version to add food intake to the environment)
3. **Time-LLM** [1]

# Conception
A significant drawback for current RL-based diabetes blood glucose treatment regimes is that they're usually low-efficient in sampling, and unsafe exploration/lack of decision transparency is hard to avoid. A convenient LLM based time-series prediction paradigm (Time-LLM) could effectively extract prior knowledge from time-series data and contain RL algorithm's learning procedure in a more concentrated and promising area, enhancing both efficiency and exploration safety. The potential of LLM also lies in its inherent capability to interpret/instruct the treatment in natural language, which may boost its application in real-life clinical applications.

# Workflow
1. Find a proper pretrained LLM. (**Done**. Currently we have two candidates: llama-7b, gpt2-1.5b, and any changes would be easy to make under our framework that is model-ignorant. A further choice should be made leveraging the performance and training efficiency under acceptable batch size.)
2. Construct the LLM time-series reprograming model. (**Prototype Done**. Based on the original Time-LLM paper [1], we streamline the original repo and make it train-inference integrated. The model adds patching/reprogramming layers with prompt-as-prefix in front of a frozen LLM to ensure the model as a whole is not very resource-consuming to train.)
3. Online learning framework with grid hyperparameter tuning. 
4. Statistical count of 'safe exploration'.
5. Run baselines.
6. Compare ours vs baselines.
7. RL algorithm interpretability using time-series reprograming LLM. (We suspect that before July it's not very promising to align LLM's interpretation as reliable outputs at all scenarios, but a few enlightening examples or case studies is already a strong support for our first project.)
8. **Future improvements during summer internship**: The systematic alignment between RL algorithm's decision-making and LLM's interpretation; Real-life scenario online RL trials.

# Reference
1. TIME-LLM: TIME SERIES FORECASTING
BY REPROGRAMMING LARGE LANGUAGE MODELS
https://arxiv.org/pdf/2310.01728.pdf
