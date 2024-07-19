from typing import SupportsFloat, Any

from gymnasium.core import ActType, ObsType
from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action
from simglucose.analysis.risk import risk_index

import numpy as np
import gymnasium
from gymnasium.utils import seeding

from gymnasium import spaces
from datetime import datetime, timedelta
from DTRGym.utils import DiscreteActionWrapper
import hashlib


def hash_seed(seed):
    if isinstance(seed, str):
        seed = seed.encode("utf8")
    elif isinstance(seed, int):
        seed = str(seed).encode("utf8")
    else:
        raise ValueError(f"seed must be str or int, got {type(seed)}")
    return int(hashlib.sha256(seed).hexdigest(), 16) % (2 ** 31)


def risk_reward_fn(bg_current, bg_next, terminated, truncated, insulin):
    # insulin is in U/min
    # if terminated:
    #     reward = -100
    # # elif truncated:
    # #     reward = 100
    # else:

    if 70 < bg_next < 140:
        risk_reward = 1
    else:
        _, _, risk = risk_index([bg_next], 1)
        risk_reward = -0.05 * risk

    # delta_bg = bg_next - bg_current
    # if delta_bg < 30:
    #     delta_reward = 0
    # elif delta_bg < 60:
    #     delta_reward = -1 / 30 * (delta_bg - 30)
    # else:
    #     delta_reward = -1

    insulin_penalty = - (insulin * 5)**2

    # reward = risk_reward + delta_reward + insulin_penalty
    reward = risk_reward + insulin_penalty

    return reward


def TIR_reward_fn(bg_current, bg_next, terminated, truncated, insulin):
    # if terminated:
    #     reward = -100
    # elif truncated:
    #     reward = 100
    # else:
        # bg reward
    if 100 < bg_next < 140:
        bg_reward = 1
    elif 70 < bg_next < 100 or 140 < bg_next < 180:
        bg_reward = - 0.5
    elif 54 < bg_next < 70 or 180 < bg_next < 250:
        bg_reward = -1
    else:
        bg_reward = -5
    # # delta reward
    # delta_bg = bg_next - bg_current
    # if delta_bg < 30:
    #     delta_reward = 0
    # elif delta_bg < 60:
    #     delta_reward = -1 / 30 * (delta_bg - 30)
    # else:
    #     delta_reward = -1

    insulin_penalty = - (insulin * 50)**2
    #
    reward = bg_reward + insulin_penalty

    return reward


class SinglePatientEnv(gymnasium.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    randomly choose from 30 patients provided by the simulator
    Time unit is 5 minute by default. The max_t will change according to the sensor's sample time.
    '''
    metadata = {'render.modes': ['human']}
    # Accessing resources with files() in Python 3.9+
    patient_list = ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
                    'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010',
                    'adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
                    'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
                    'child#001', 'child#002', 'child#003', 'child#004', 'child#005',
                    'child#006', 'child#007', 'child#008', 'child#009', 'child#010']
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name: str,
                 max_t: int = 24 * 60,
                 obs_window: int = 48,
                 reward_fn=risk_reward_fn,
                 random_init_bg: bool = False,
                 random_obs: bool = False,
                 random_meal: bool = False,
                 missing_rate=0.0,
                 sample_time=1,
                 start_time=0,
                 **kwargs):
        self.env = None
        self.reward_fn = reward_fn
        self.max_t = max_t
        self.patient_name = patient_name
        self.random_init_bg = random_init_bg
        self.random_obs = random_obs
        self.random_meal = random_meal
        self.missing_rate = missing_rate
        T1DPatient.SAMPLE_TIME = sample_time
        self.sample_time = sample_time
        self.start_time = start_time
        self.last_obs = None
        self.obs_window = obs_window
        self.episode_id = -1
        # pump_upper_act = self.pump_params[self.pump_params["Name"] == self.INSULIN_PUMP_HARDWARE]["max_basal"].values
        self.env_info = {'action_type': 'continuous', 'reward_range': (-np.inf, np.inf),
                         "state_key": ["Continuous Glucose Monitoring", "Blood Glucose", "Risk"],
                         "obs_key": ["Continuous Glucose Monitoring (mg/dL)"],
                         "act_key": ["Insulin Dose (U/h)"],
                         "metric_key": ["TIR", "Hypo", "Hyper", "CV"],
                         }
        self._action_space = None  # Backing attribute for lazy loading
        self._obs_space = None  # Backing attribute for lazy loading

    def reset(self, seed: int = None, **kwargs):
        self.seed(seed)
        self.t = 0
        self.step_counter = 0
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.terminated = False
        self.truncated = False
        self.episode_id += 1
        if self.patient_name not in self.patient_list:
            raise ValueError(f"patient_name must be in {self.patient_list}")

        self.env, _, _, _ = self._create_env(random_init_bg=self.random_init_bg)
        obs, _, _, info = self.env.reset()
        bg = info["bg"]
        meal = info["meal"]

        state = self._get_state(obs[0], bg, meal)
        obs = self._state2obs(state, random_obs=self.random_obs, enable_missing=False)
        self.bg_history = [float(obs)]
        self.drug_history = [0]
        all_info = {"action": 0, "instantaneous_reward": 0, "step": 0, "episode_id": self.episode_id}
        info.pop("patient_state")
        all_info.update(info)

        bg = np.zeros([self.obs_window], dtype=np.float32)
        act = np.zeros([self.obs_window], dtype=np.float32)
        bg[-len(self.bg_history):] = np.array(self.bg_history) * 0.01 # scale bg to [0, 6]
        bg[bg == 0] = -1
        act[-len(self.drug_history):] = self.drug_history
        obs = np.stack([bg, act], axis=1)
        return obs, all_info

    def step(self, action):
        if self.terminated or self.truncated:
            print("This treat is end, please call reset or export")
            return None, None, self.terminated, self.truncated, {}
        if action < self.action_space.low or action > self.action_space.high:
            raise ValueError(f"action should be in [{self.action_space.low}, {self.action_space.high}]")
        self.t += self.env.sample_time
        self.step_counter += 1
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)  # U/h -> U/min
        obs, _, _, info = self.env.step(act)
        bg_next = info["bg"]
        meal_next = info["meal"]

        state = self._get_state(obs[0], bg_next, meal_next)
        obs = self._state2obs(state, random_obs=self.random_obs, enable_missing=True)

        if self.t >= self.max_t:
            self.terminated = False
            self.truncated = True
        if not (10 < bg_next < 600):  # we define the lower bound of bg to be 54 since <54 is severe hypoglycemia
            self.terminated = True
            self.truncated = False

        reward = self.reward_fn(bg_current=self.bg_history[-1], bg_next=bg_next,
                                terminated=self.terminated, truncated=self.truncated,
                                insulin=action)

        self.bg_history.append(float(obs))
        self.drug_history.append(float(action))

        all_info = {"action": float(action), "instantaneous_reward": float(reward), "step": self.step_counter,
                    "episode_id": self.episode_id}
        info.pop("patient_state")
        all_info.update(info)

        # get rnn style obs
        bg = np.zeros([self.obs_window], dtype=np.float32)
        act = np.zeros([self.obs_window], dtype=np.float32)
        bg[-len(self.bg_history):] = np.array(self.bg_history[-self.obs_window:]) * 0.01  # scale bg to [0, 6]
        bg[bg == 0] = -1
        act[-len(self.drug_history):] = self.drug_history[-self.obs_window:]
        obs = np.stack([bg, act], axis=1)
        return obs, float(reward), self.terminated, self.truncated, all_info

    def seed(self, seed):
        self.np_random, seed1 = seeding.np_random(seed=seed)

    # def get_metrics(self):
    #     obs_records = np.array(self.bg_records)
    #     TIR = np.sum(np.logical_and(obs_records >= 70, obs_records <= 180)) / len(obs_records)
    #     hypo = np.sum(obs_records < 70) / len(obs_records)
    #     hyper = np.sum(obs_records > 180) / len(obs_records)
    #     CV = np.std(obs_records) / np.mean(obs_records)
    #     metrics = {"TIR": TIR, "Hypo": hypo, "Hyper": hyper, "CV": CV}
    #     return metrics

    def _create_env(self, random_init_bg=True):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash_seed(int(self.np_random.integers(0, 1000))) % 2 ** 31
        seed3 = hash_seed(seed2 + 1) % 2 ** 31
        seed4 = hash_seed(seed3 + 1) % 2 ** 31

        # available sensors are ['Dexcom', 'GuardianRT', 'Navigator']
        # the only sensor with sample time 5 is GuardianRT
        self.SENSOR_HARDWARE = 'GuardianRT'

        patient = T1DPatient.withName(self.patient_name, random_init_bg=random_init_bg, seed=seed2)

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed3)
        if self.start_time > 24 * 60 * 60:
            raise ValueError("start_time must be less than 24 hours")
        time_string = str(timedelta(seconds=self.start_time))
        hour, minute, second = map(int, time_string.split(':'))
        start_time = datetime(2018, 1, 1, hour, minute, second)
        if self.random_meal:
            scenario = RandomScenario(start_time=start_time, seed=seed4)
        else:
            scenario_info = [(7 - self.start_time / 60 / 60, 45), (12 - self.start_time / 60 / 60, 70),
                             (15 - self.start_time / 60 / 60, 10), (18 - self.start_time / 60 / 60, 80)]  # (time, meal)
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

    @property
    def action_space(self):
        if self._action_space is None:  # Check if it is already calculated
            pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
            ub = pump._params["max_basal"] / 60
            self._action_space = spaces.Box(low=0, high=ub, shape=(1,))
        return self._action_space

    @property
    def observation_space(self):
        if self._obs_space is None:
            self._obs_space = spaces.Box(low=np.array([10, self.action_space.low[0]]),
                                         high=np.array([600, self.action_space.high[0]]), dtype=np.float32)
        return self._obs_space

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]


class RandomPatientEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    # Accessing resources with files() in Python 3.9+
    patient_list = ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
                    'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010',
                    'adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
                    'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
                    'child#001', 'child#002', 'child#003', 'child#004', 'child#005',
                    'child#006', 'child#007', 'child#008', 'child#009', 'child#010']
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, candidates: list,
                 max_t: int = 24 * 60,
                 reward_fn=risk_reward_fn,
                 random_init_bg: bool = False,
                 random_obs: bool = False,
                 random_meal: bool = False,
                 missing_rate=0.0,
                 sample_time=1,
                 start_time=0, ):
        self.env = None
        self.reward_fn = reward_fn
        self.max_t = max_t
        self.candidates = candidates
        self.random_init_bg = random_init_bg
        self.random_obs = random_obs
        self.random_meal = random_meal
        self.missing_rate = missing_rate
        T1DPatient.SAMPLE_TIME = sample_time
        self.sample_time = sample_time
        self.start_time = start_time
        self.last_obs = None
        # pump_upper_act = self.pump_params[self.pump_params["Name"] == self.INSULIN_PUMP_HARDWARE]["max_basal"].values
        self.env_info = {'action_type': 'continuous', 'reward_range': (-np.inf, np.inf),
                         "state_key": ["Continuous Glucose Monitoring", "Blood Glucose", "Risk"],
                         "obs_key": ["Continuous Glucose Monitoring (mg/dL)"],
                         "act_key": ["Insulin Dose (U/h)"],
                         "metric_key": ["TIR", "Hypo", "Hyper", "CV"],
                         }
        self._action_space = None  # Backing attribute for lazy loading
        self._obs_space = None  # Backing attribute for lazy loading

    def reset(self, seed: int = None, **kwargs):
        self.patient_name = self.np_random.choice(self.candidates)
        self.env = SinglePatientEnv(patient_name=self.patient_name,
                                    max_t=self.max_t, reward_fn=self.reward_fn,
                                    random_init_bg=self.random_init_bg,
                                    random_obs=self.random_obs,
                                    random_meal=self.random_meal,
                                    start_time=self.start_time,
                                    missing_rate=self.missing_rate,
                                    sample_time=T1DPatient.SAMPLE_TIME)
        return self.env.reset(seed=seed, **kwargs)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def seed(self, seed):
        self.np_random, seed1 = seeding.np_random(seed=seed)

    @property
    def action_space(self):
        if self._action_space is None:  # Check if it is already calculated
            pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
            ub = pump._params["max_basal"] / 60
            self._action_space = spaces.Box(low=0, high=ub, shape=(1,))
        return self._action_space

    @property
    def observation_space(self):
        if self._obs_space is None:
            self._obs_space = spaces.Box(low=np.array([54, self.action_space.low[0]]),
                                         high=np.array([600, self.action_space.high[0]]), dtype=np.float32)
        return self._obs_space


def create_SimGlucoseEnv_single_patient(patient_name: str, max_t: int = 16 * 60, discrete: bool = False, n_act: int = 5,
                                        **kwargs):
    env = SinglePatientEnv(
        patient_name,
        max_t=max_t,
        sample_time=1,
        start_time=5 * 60,
        random_init_bg=True,
        random_obs=True, random_meal=True,
        missing_rate=0)
    if discrete:
        wrapped_env = DiscreteActionWrapper(env, n_act)
        return wrapped_env
    return env


def create_SimGlucoseEnv_adult1(n_act: int = 11, discrete=False, obs_window=48, **kwargs):
    env = SinglePatientEnv('adult#001', 16 * 60, random_init_bg=True,
                           random_obs=True, random_meal=True, start_time=5 * 60, obs_window=obs_window,
                           missing_rate=0.0)
    if discrete:
        wrapped_env = DiscreteActionWrapper(env, n_act)
        return wrapped_env
    return env


def create_SimGlucoseEnv_adult4(n_act: int = 11, discrete=False, **kwargs):
    env = RandomPatientEnv(candidates=["adult#001",
                                       "adult#002",
                                       "adult#003",
                                       "adult#004", ],
                           max_t=16 * 60,
                           sample_time=1,
                           random_init_bg=True,
                           start_time=5 * 60,
                           random_obs=True, random_meal=True,
                           missing_rate=0.)
    if discrete:
        wrapped_env = DiscreteActionWrapper(env, n_act)
        return wrapped_env
    return env


def create_SimGlucoseEnv_all4(n_act: int = 11, discrete=False, **kwargs):
    env = RandomPatientEnv(candidates=["adult#001",
                                       "adult#002",
                                       "adult#003",
                                       "adult#004",

                                       "child#001",
                                       "child#002",
                                       "child#003",
                                       "child#004",

                                       "adolescent#001",
                                       "adolescent#002",
                                       "adolescent#003",
                                       "adolescent#004"],
                           max_t=16 * 60,
                           sample_time=1,
                           random_init_bg=True,
                           start_time=5 * 60,
                           random_obs=True, random_meal=True,
                           missing_rate=0.)
    if discrete:
        wrapped_env = DiscreteActionWrapper(env, n_act)
        return wrapped_env
    return env
