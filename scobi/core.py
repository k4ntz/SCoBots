import scobi.environments.env_manager as em
import numpy as np
from scobi.environments.drivers import ocatari_step
from scobi.focus import Focus
from scobi.utils.logging import GeneralError
from gymnasium import spaces
from termcolor import colored

# TODO: high prio: time env step durations. there might be a bottleneck here
class Environment():
    def __init__(self, env_name, interactive=False, focus_dir="experiments/focusfiles", focus_file=None):
        self.oc_env = em.make(env_name)
        actions = self.oc_env._env.unwrapped.get_action_meanings() # TODO: oc envs should answer this, not the raw env
        self.oc_env.reset()
        obs, _, _, _, _ = ocatari_step(self.oc_env.step(1))

        self.did_reset = False
        self.focus = Focus(env_name, interactive, focus_dir, focus_file, obs, actions)
        self.focus_file = self.focus.FOCUSFILEPATH
        self.action_space = spaces.Discrete(len(self.focus.PARSED_ACTIONS))
        self.action_space_description = self.focus.PARSED_ACTIONS
        self.observation_space_description = self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS
        # TODO: define lower and upper bounds in scobi, -1000 to 1000 for now
        # TODO: inacurrate for now, counts the func entries, not the output of the functions, maybe use dummy object dict to evaluate once?
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.focus.FEATURE_VECTOR_SIZE,), dtype=np.float32)



    def step(self, action):
        if not self.did_reset:
            GeneralError("Cannot call env.step() before calling env.reset()")
        if self.action_space.contains(action):
            obs, reward, truncated, terminated, info = ocatari_step(self.oc_env.step(action))
            sco_obs = self.focus.get_feature_vector(obs)
            sco_reward = reward #reward shaping here
            sco_truncated = truncated
            sco_terminated = terminated
            sco_info = info
            return sco_obs, sco_reward, sco_truncated, sco_terminated, sco_info
        else:
            raise ValueError("scobi> Action not in action space")

    def reset(self):
        self.did_reset = True
        # additional scobi reset steps here
        self.oc_env.reset()

    def close(self):
        # additional scobi close steps here
        self.oc_env.close()
