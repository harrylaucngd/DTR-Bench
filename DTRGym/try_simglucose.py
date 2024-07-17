import DTRGym
import gymnasium as gym
from DTRGym.simglucose_env import RandomPatientEnv
from DTRGym.utils import DiscreteActionWrapper
from DTRBench.utils.misc import set_global_seed
import pandas as pd
import matplotlib.pyplot as plt
import time
from DTRBench.src.collector import GlucoseCollector
from DTRBench.src.baseline_policy import ConstantPolicy
set_global_seed(0)
import warnings

warnings.filterwarnings("ignore")

simulation_time = 24 * 60
n_act = 11
env = RandomPatientEnv(candidates=["adolescent#001"],
                       max_t=simulation_time,
                       sample_time=1,
                       start_time=0,
                       random_init_bg=False,
                       random_obs=False, random_meal=True,
                       missing_rate=0)
policy = ConstantPolicy(dose=0.01, action_space=env.action_space)
collector = GlucoseCollector(policy, env, None)
step = 0
print(env.patient_name)
done = False
df1 = []
start = time.time()
collector.collect(n_episode=2)
end = time.time()
