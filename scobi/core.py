"""scobi core"""
import numpy as np
from gymnasium import spaces
import scobi.environments.env_manager as em
from scobi.utils.game_object import get_wrapper_class
from scobi.focus import Focus
from scobi.utils.logging import Logger
from PIL import Image, ImageDraw

class Environment():
    def __init__(self, env_name, interactive=False, focus_dir="experiments/focusfiles", focus_file=None, reward=False, hide_properties=False, silent=False, refresh_yaml=True, draw_features=False):
        self.logger = Logger(silent=silent)
        self.oc_env = em.make(env_name, self.logger)

        # TODO: tie to em.make
        self.game_object_wrapper = get_wrapper_class()

        # TODO: oc envs should answer this, not the raw env
        actions = self.oc_env._env.unwrapped.get_action_meanings()

        self.oc_env.reset()
        max_objects = self._wrap_map_order_game_objects(self.oc_env.max_objects)
        self.did_reset = False
        self.focus = Focus(env_name, interactive, reward, hide_properties, focus_dir, focus_file, max_objects, actions, refresh_yaml, self.logger)
        self.focus_file = self.focus.FOCUSFILEPATH
        self.action_space = spaces.Discrete(len(self.focus.PARSED_ACTIONS))
        self.action_space_description = self.focus.PARSED_ACTIONS
        self.observation_space_description = self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS
        self.feature_vector_description = self.focus.get_feature_vector_description()
        self.draw_features = draw_features
        self.feature_attribution = []

        # TODO: inacurrate for now, counts the func entries, not the output of the functions, maybe use dummy object dict to evaluate once?
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.focus.FEATURE_VECTOR_SIZE,), dtype=np.float32)


    def step(self, action):
        if not self.did_reset:
            self.logger.GeneralError("Cannot call env.step() before calling env.reset()")
        elif self.action_space.contains(action):
            obs, reward, truncated, terminated, info = self.oc_env.step(action)
            objects = self._wrap_map_order_game_objects(self.oc_env.objects)
            sco_obs, sco_reward = self.focus.get_feature_vector(objects)
            freeze_mask = self.focus.get_current_freeze_mask()
            #print(sco_reward)
            #print(sco_obs[0:100])
            sco_truncated = truncated
            sco_terminated = terminated
            sco_info = info
            if self.draw_features:
                obs = self._draw_overlay(obs, sco_obs, freeze_mask)
            return sco_obs, reward, sco_reward, sco_truncated, sco_terminated, sco_info, obs # 7
        else:
            raise ValueError("scobi> Action not in action space")

    def reset(self):
        self.did_reset = True
        # additional scobi reset steps here
        self.oc_env.reset()

    def close(self):
        # additional scobi close steps here
        self.oc_env.close()

    def set_feature_attribution(self, att):
        self.feature_attribution = att

    def _wrap_map_order_game_objects(self, oc_obj_list):
        out = []
        counter_dict = {}
        player_obj = None

        # wrap
        scobi_obj_list = [self.game_object_wrapper(obj) for obj in oc_obj_list]

        # order
        for scobi_obj in scobi_obj_list:
            if "Player" in scobi_obj.name or "Chicken" in scobi_obj.name:
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

    def _make_darker(self, color, col_precent=0.8):
        """
        return a darker color
        """
        if not color:
            print("No color passed, using default black")
            return [0, 0, 0]
        return [int(col * col_precent) for col in color]

    def _mark_bb(self, image_array, bb, color=(255, 0, 0), surround=True):
        """
        marks a bounding box on the image
        """
        x, y, w, h = bb
        x = int(x)
        y = int(y)
        if surround:
            if x > 0:
                x, w = x - 1, w + 1
            else:
                x, w = x, w
            if y > 0:
                y, h = y - 1, h + 1
            else:
                y, h = y, h
        bottom = min(209, y + h)
        right = min(159, x + w)
        image_array[y:bottom + 1, x] = color
        image_array[y:bottom + 1, right] = color
        image_array[y, x:right + 1] = color
        image_array[bottom, x:right + 1] = color

    def _draw_overlay(self, obs_image, feature_vector, freeze_mask):
        if len(self.feature_attribution) == 0:
                return obs_image
        features = self.feature_vector_description[0]
        fv_backmap = self.feature_vector_description[1]
        i = 0
        img_shape = obs_image.shape
        img = Image.fromarray(obs_image)
        draw = ImageDraw.Draw(img, "RGBA")
        feature_attribution = (255*(self.feature_attribution - np.min(self.feature_attribution))/np.ptp(self.feature_attribution)).astype(int) 
        for feature in features:
            i += 1
            idxs = np.where(fv_backmap == i-1)[0]
            feature_name = feature[0]
            feature_signature = feature[1]
            fv_entries = feature_vector[idxs[0]:idxs[-1]+1]
            fv_attribution = feature_attribution[idxs[0]:idxs[-1]+1]
            fv_freeze_mask = freeze_mask[idxs[0]:idxs[-1]+1]
            alpha = int(np.mean(fv_attribution))
            if 0 in fv_freeze_mask:
                continue
            if feature_name == "POSITION":
                radius = 2
                alpha = 255
                x = fv_entries[0]
                y = fv_entries[1]
                coords = (x - radius, y - radius, x + radius, y + radius)
                draw.ellipse(coords, fill=(40,40,40,alpha), outline=(0,0,0,alpha))
            elif feature_name == "TODO": # POSITION_HISTORY
                radius = 2
                x_n = fv_entries[0]
                y_n = fv_entries[1]
                x_t = fv_entries[2]
                y_t = fv_entries[3]
                coords_now = (x_n - radius, y_n - radius, x_n + radius, y_n + radius)
                coords_then = (x_t - radius, y_t - radius, x_t + radius, y_t + radius)
                draw.ellipse(coords_now, fill=(100,100,alpha), outline=(0,0,0, alpha))
                draw.ellipse(coords_then, fill=(200,200,alpha), outline=(0,0,0, alpha))
            elif feature_name == "CENTER":
                x = fv_entries[0]
                y = fv_entries[1]
                coords = (x- radius, y - radius, x + radius, y + radius)
                draw.ellipse(coords, fill=(10,100,10, alpha), outline=(0,0,0, alpha))
            elif feature_name == "DISTANCE":
                delta = [fv_entries[0], fv_entries[1]]
                source_object_coords = feature_signature[0]
                idx = -1
                for f in features:
                    idx += 1
                    if f == source_object_coords:
                        break
                idxs = np.where(fv_backmap == idx)[0]
                source_object_coord_values = feature_vector[idxs[0]:idxs[-1]+1]
                vector = np.add(source_object_coord_values, delta).tolist()
                draw.line(source_object_coord_values + vector, fill=(0,0,255,alpha), width=1)
            elif feature_name == "EUCLIDEAN_DISTANCE":
                source_object_coords = feature_signature[0]
                target_object_coords = feature_signature[1]
                
                idx = -1
                for f in features:
                    idx += 1
                    if f == source_object_coords:
                        break
                idxs = np.where(fv_backmap == idx)[0]
                source_object_coord_values = feature_vector[idxs[0]:idxs[-1]+1]

                idx = -1
                for f in features:
                    idx += 1
                    if f == target_object_coords:
                        break
                idxs = np.where(fv_backmap == idx)[0]
                target_object_coord_values = feature_vector[idxs[0]:idxs[-1]+1]

                draw.line(source_object_coord_values + target_object_coord_values , fill=(0,0,255,alpha), width=1)
            elif feature_name == "TODO": # LINEAR_TRAJECTORY
                delta = [fv_entries[0], fv_entries[1]]
                source_object_coords = feature_signature[0]
                idx = -1
                for f in features:
                    idx += 1
                    if f == source_object_coords:
                        break
                idxs = np.where(fv_backmap == idx)[0]
                source_object_coord_values = feature_vector[idxs[0]:idxs[-1]+1]
                vector = np.add(source_object_coord_values, delta)#.tolist()
                vector = vector / np.sqrt(np.sum(vector**2))
                vector *= 100
                draw.line(source_object_coord_values + vector.tolist(), fill=(0,155,155,alpha), width=1)
        return np.array(img)