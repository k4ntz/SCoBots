"""scobi core"""
import numpy as np
from gymnasium import spaces
import scobi.environments.env_manager as em
from scobi.utils.game_object import get_wrapper_class
from scobi.focus import Focus
from scobi.utils.logging import Logger


class Environment():
    def __init__(self, env_name, interactive=False, focus_dir="experiments/focusfiles", focus_file=None, silent=False):
        self.logger = Logger(silent=silent)
        self.oc_env = em.make(env_name, self.logger)

        # TODO: tie to em.make
        self.game_object_wrapper = get_wrapper_class()

        # TODO: oc envs should answer this, not the raw env
        actions = self.oc_env._env.unwrapped.get_action_meanings()

        self.oc_env.reset()
        max_objects = self._wrap_map_order_game_objects(self.oc_env.max_objects)
        self.did_reset = False
        self.focus = Focus(env_name, interactive, focus_dir, focus_file, max_objects, actions, self.logger)
        self.focus_file = self.focus.FOCUSFILEPATH
        self.action_space = spaces.Discrete(len(self.focus.PARSED_ACTIONS))
        self.action_space_description = self.focus.PARSED_ACTIONS
        self.observation_space_description = self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS

        # TODO: inacurrate for now, counts the func entries, not the output of the functions, maybe use dummy object dict to evaluate once?
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.focus.FEATURE_VECTOR_SIZE,), dtype=np.float32)


    def step(self, action):
        if not self.did_reset:
            self.logger.GeneralError("Cannot call env.step() before calling env.reset()")
        elif self.action_space.contains(action):
            obs, reward, truncated, terminated, info = self.oc_env.step(action)
            objects = self._wrap_map_order_game_objects(self.oc_env.objects)
            sco_obs = self.focus.get_feature_vector(objects)
            #print(sco_obs[0:100])
            sco_reward = reward #reward shaping here
            sco_truncated = truncated
            sco_terminated = terminated
            sco_info = info
            return sco_obs, sco_reward, sco_truncated, sco_terminated, sco_info, obs
        else:
            raise ValueError("scobi> Action not in action space")

    def reset(self):
        self.did_reset = True
        # additional scobi reset steps here
        self.oc_env.reset()

    def close(self):
        # additional scobi close steps here
        self.oc_env.close()

    def _wrap_map_order_game_objects(self, oc_obj_list):
        out = []
        counter_dict = {}
        player_obj = None

        # wrap
        scobi_obj_list = [self.game_object_wrapper(obj) for obj in oc_obj_list]

        # order
        for scobi_obj in scobi_obj_list:
            if "Player" in scobi_obj.name:
                player_obj = scobi_obj
                break
        scobi_obj_list = sorted(scobi_obj_list, key=lambda a : a.distance(player_obj))

        # map
        for scobi_obj in scobi_obj_list:
            if not scobi_obj.category in counter_dict:
                counter_dict[scobi_obj.category] = 1
            else:
                counter_dict[scobi_obj.category] +=1
            scobi_obj.number = counter_dict[scobi_obj.category]
            out.append(scobi_obj)
        # returns full objectlist [closest_visible : farest_visible] + [closest_invisible : farest invisibe]
        # if focus file specifies for example top3 closest objects (by selecting pin1, pin2, pin3), 
        # the features vector is always calculated based on the 3 closest visible objects of category pin.
        # so if for example pin1 becomes invisible during training, the top3 list is filled accordingly with closest visible objects of category pin
        # if there is none to fill, pin2 and pin3 are the closest visible and all derived features for the third pin (which is invisble) are frozen
        return out

