import DTRGym
import gymnasium as gym
from DTRGym.simglucose_env import RandomPatientEnv
from DTRGym.utils import DiscreteActionWrapper
from DTRBench.utils.misc import set_global_seed
import pandas as pd
import matplotlib.pyplot as plt
import time
from DTRBench.src.collector import GlucoseCollector
from DTRBench.src.naive_baselines import ConstantPolicy, RandomPolicy
from DTRGym.base import make_env
import warnings

warnings.filterwarnings("ignore")

simulation_time = 24 * 60
n_act = 11
env,training_env, testing_env = make_env("SimGlucoseEnv-adult1", 1, 1, 1, num_actions=11, discrete=False)
# policy = ConstantPolicy(dose=0.01, action_space=env.action_space)
policy = RandomPolicy(min_act=0, max_act=0.25, action_space=env.action_space)
collector = GlucoseCollector(policy, env, None)
step = 0
print(env.patient_name)
done = False
df1 = []
start = time.time()
result = collector.collect(n_episode=10)
print(result)

