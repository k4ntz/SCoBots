import scobi.environments.env_manager as em
import numpy as np
from scobi.environments.drivers import ocatari_step
from scobi.focus import Focus


class Environment():
    def __init__(self, env_name, focus_file=None):
        self.env = em.make(env_name)
        actions = self.env._env.unwrapped.get_action_meanings() # TODO: oc envs should answer this, not the raw env
        self.env.reset()
        obs, _, _, _, info = ocatari_step(self.env.step(1))
        focus_cfg = {}
        focus_cfg["env_name"] = env_name
        focus_cfg["focus_file"] = focus_file
        focus_cfg["focusdir"] = "experiments/focusfiles"
        focus_cfg["focus_mode"] = "scobot"
        focus_cfg["exp_name"] = "testexp"
        self.focus = Focus(cfg=focus_cfg, raw_features=obs, actions=actions)
        self.action_space = self.focus.PARSED_ACTIONS
        self.observation_space = self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS


    def step(self, action):
        obs, reward, truncated, terminated, info = ocatari_step(self.env.step(action))
        sco_obs = self.focus.get_feature_vector(obs)
        sco_reward = reward #reward shaping here
        sco_truncated = truncated
        sco_terminated = terminated
        sco_info = info
        return sco_obs, sco_reward, sco_truncated, sco_terminated, sco_info
        

    def reset(self):
        # additional scobi reset steps here
        self.env.reset()

    def close(self):
        # additional scobi close steps here
        self.env.close()
