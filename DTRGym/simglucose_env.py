from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
import numpy as np
from importlib import resources
from gym.utils import seeding
from functools import partial

import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from DTRGym.utils import DiscreteActionWrapper


def risk_reward_fn(bg_current, bg_next, terminated, truncated, insulin):
    if terminated:
        reward = -100
    elif truncated:
        reward = 100
    else:
        _, _, risk = risk_index([bg_next], 1)
        risk_reward = -np.log10(risk)

        delta_bg = bg_next - bg_current
        if delta_bg < 30:
            delta_reward = 0
        elif delta_bg < 60:
            delta_reward = -1 / 30 * (delta_bg - 30)
        else:
            delta_reward = -1

        insulin_penalty = - np.log10(max(insulin[0], 1))

        reward = risk_reward + delta_reward + insulin_penalty

    return reward

def TIR_reward_fn(bg_current, bg_next, terminated, truncated, insulin):
    if terminated:
        reward = -100
    elif truncated:
        reward = 100
    else:
        # bg reward
        if 70 < bg_next < 180:
            bg_reward = 1
        elif 50 < bg_next < 70 or 180 < bg_next < 250:
            bg_reward = -1
        else:
            bg_reward = -2
        # delta reward
        delta_bg = bg_next - bg_current
        if delta_bg < 30:
            delta_reward = 0
        elif delta_bg < 60:
            delta_reward = -1/30 * (delta_bg-30)
        else:
            delta_reward = -1

        insulin_penalty = - np.log10(max(insulin[0], 1))

        reward = bg_reward + delta_reward + insulin_penalty

    return reward


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    randomly choose from 30 patients provided by the simulator
    Time unit is 1 minute. The max_t will change according to the sensor's sample time.
    '''
    metadata = {'render.modes': ['human']}
    # Accessing resources with files() in Python 3.9+
    vpatient_params_path = resources.files('simglucose.params') / 'vpatient_params.csv'
    with vpatient_params_path.open() as f:
        patient_list = pd.read_csv(f)["Name"].to_list()
    pump_params_path = resources.files('simglucose.params') / 'pump_params.csv'
    with pump_params_path.open() as f:
        pump_params = pd.read_csv(f)
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, max_t: int = 24*60, reward_fn=risk_reward_fn,
                 random_patient: bool = False,
                 random_init: bool = False,
                 random_obs: bool = False,
                 random_meal: bool = False,
                 missing_rate=0.0,
                 **kwargs):
        """
        random_patient is equalivalent to random_pkpd
        """
        self.patient_name = None
        self.env = None
        self.reward_fn = reward_fn
        self.max_t = max_t
        self.random_patient = random_patient
        self.random_init = random_init
        self.random_obs = random_obs
        self.random_meal = random_meal
        self.missing_rate = missing_rate

        self.last_obs = None
        self.observation_space = spaces.Box(low=10, high=600, shape=(1,), dtype=np.float32)
        # pump_upper_act = self.pump_params[self.pump_params["Name"] == self.INSULIN_PUMP_HARDWARE]["max_basal"].values
        pump_upper_act = 10  # U/h
        self.action_space = spaces.Box(low=0, high=pump_upper_act, shape=(1,), dtype=np.float32)
        self.env_info = {'action_type': 'continuous', 'reward_range': (-np.inf, np.inf),
                         "state_key": ["Continuous Glucose Monitoring", "Blood Glucose", "Risk"],
                         "obs_key": ["Continuous Glucose Monitoring (mg/dL)"],
                         "act_key": ["Insulin Dose (U/h)"],
                         "metric_key": ["TIR", "Hypo", "Hyper", "CV"],
                         }

    def reset(self, seed: int = None, **kwargs):
        self.seed(seed)
        self.t = 0
        self.idx = 0
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.terminated = False
        self.truncated = False

        if self.random_patient:
            self.patient_name = np.random.choice(self.patient_list)
        else:
            self.patient_name = self.patient_list[0]
            
        self.bg_records = []
        self.env, _, _, _ = self._create_env(random_init=self.random_init)
        obs, _, done, info = self.env.reset()
        bg = info["bg"]
        meal = info["meal"]

        state = self._get_state(obs[0], bg, meal)
        obs = self._state2obs(state, random_obs=self.random_obs, enable_missing=False)

        self.bg_records.append(bg)

        info = {"state": state, "action": np.zeros(shape=(1,)), "instantaneous_reward": 0}
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        if self.terminated or self.truncated:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}
        if action < self.action_space.low or action > self.action_space.high:
            raise ValueError(f"action should be in [{self.action_space.low}, {self.action_space.high}]")
        self.t += self.env.sample_time
        self.idx += 1
        # This gym only controls basal insulin
        act = Action(basal=action/60, bolus=0)  # U/h -> U/min
        obs, _, done, info = self.env.step(act)
        bg_next = info["bg"]
        bg_current = self.bg_records[self.idx - 1]
        meal_next = info["meal"]
        self.bg_records.append(bg_next)

        state = self._get_state(obs[0], bg_next, meal_next)
        obs = self._state2obs(state, random_obs=self.random_obs, enable_missing=True)


        if self.t >= self.max_t:
            self.terminated = False
            self.truncated = True
        if not (10 < bg_next < 600):
            self.terminated = True
            self.truncated = False

        reward = self.reward_fn(bg_current=bg_current, bg_next=bg_next,
                                terminated=self.terminated, truncated=self.truncated,
                                insulin=action)
        # reward = rew
        info = {"state": state, "action": action, "instantaneous_reward": reward}

        return obs, reward, self.terminated, self.truncated, info

    def seed(self, seed):
        self.np_random, seed1 = seeding.np_random(seed=seed)

    def get_metrics(self):
        obs_records = np.array(self.bg_records)
        TIR = np.sum(np.logical_and(obs_records >= 70, obs_records <= 180)) / len(obs_records)
        hypo = np.sum(obs_records < 70) / len(obs_records)
        hyper = np.sum(obs_records > 180) / len(obs_records)
        CV = np.std(obs_records) / np.mean(obs_records)
        metrics = {"TIR": TIR, "Hypo": hypo, "Hyper": hyper, "CV": CV}
        return metrics

    def _create_env(self, random_init=True):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        # available sensors are ['Dexcom', 'GuardianRT', 'Navigator']
        # the only sensor with sample time 5 is GuardianRT
        self.SENSOR_HARDWARE = 'GuardianRT'

        patient = T1DPatient.withName(self.patient_name, random_init_bg=random_init, seed=seed2)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed3)

        # hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, 0, 0, 0)
        if self.random_meal:
            scenario = RandomScenario(start_time=start_time, seed=seed4)
        else:
            scenario_info = [(7, 45), (12, 70), (15, 10), (18, 80)]  # (time, meal)
            scenario = CustomScenario(start_time=start_time, scenario=scenario_info)

        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4


    def render(self, mode='human', close=False):
        self.env.render(close=close)

    def _get_state(self, CGM: float, bg: float, meal: float) -> dict:
        _, _, risk = risk_index([bg], 1)

        state = {"Continuous Glucose Monitoring": CGM,
                 "Blood Glucose": bg,
                 "Risk": risk,
                 "Meal": meal}
        return state

    def _state2obs(self, state, random_obs: bool, enable_missing: bool):
        if random_obs:
            obs = state["Continuous Glucose Monitoring"]
        else:
            obs = state["Blood Glucose"]

        if enable_missing and np.random.uniform(0, 1) < self.missing_rate:
            obs = self.last_obs
        else:
            self.last_obs = obs
        return np.array([obs], dtype=np.float32)


def create_SimGlucoseEnv_continuous(max_t: int = 24*60, n_act: int = 5, **kwargs):
    env = T1DSimEnv(max_t, **kwargs)
    return env


def create_SimGlucoseEnv_discrete(max_t: int = 24*60, n_act: int = 5, **kwargs):
    env = T1DSimEnv(max_t, **kwargs)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_SimGlucoseEnv_discrete_setting1(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=False, random_init=False,
                    random_obs=False, random_meal=False,
                    missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_SimGlucoseEnv_discrete_setting2(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=False,
                    random_obs=False, random_meal=False,
                    missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_SimGlucoseEnv_discrete_setting3(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=True,
                    random_obs=True, random_meal=False,
                    missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_SimGlucoseEnv_discrete_setting4(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=True,
                    random_obs=True, random_meal=True,
                    missing_rate=0.0)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_SimGlucoseEnv_discrete_setting5(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=True,
                    random_obs=True, random_meal=True,
                    missing_rate=0.5)
    wrapped_env = DiscreteActionWrapper(env, n_act)
    return wrapped_env


def create_SimGlucoseEnv_continuous_setting1(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=False, random_init=False,
                    random_obs=False, random_meal=False,
                    missing_rate=0.0)
    return env


def create_SimGlucoseEnv_continuous_setting2(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=False,
                    random_obs=False, random_meal=False,
                    missing_rate=0.0)
    return env


def create_SimGlucoseEnv_continuous_setting3(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=True,
                    random_obs=True, random_meal=False,
                    missing_rate=0.0)
    return env


def create_SimGlucoseEnv_continuous_setting4(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=True,
                    random_obs=True, random_meal=True,
                    missing_rate=0.0)
    return env


def create_SimGlucoseEnv_continuous_setting5(max_t: int = 24*60, n_act: int = 5):
    env = T1DSimEnv(max_t, random_patient=True, random_init=True,
                    random_obs=True, random_meal=True,
                    missing_rate=0.5)
    return env

