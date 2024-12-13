"""scobi core"""
import numpy as np
from gymnasium import spaces, Env
import scobi.environments.env_manager as em
from scobi.utils.game_object import get_wrapper_class
from scobi.focus import Focus
from scobi.utils.logging import Logger
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy


class Environment(Env):
    def __init__(self, env_name, seed=None, focus_dir="resources/focusfiles", focus_file=None, reward=0, hide_properties=False, silent=False, refresh_yaml=True, draw_features=False, hud=False):
        self.logger = Logger(silent=silent)
        # set buffer_window=2, s.t. we can build POSITION_HISTORY properties, which are needed by all envs.
        self.oc_env = em.make(env_name, self.logger, hud=hud, buffer_window_size=2)
        self.seed = seed
        self.randomstate = np.random.RandomState(self.seed)
        # TODO: tie to em.make
        self.game_object_wrapper = get_wrapper_class()

        # TODO: oc envs should answer this, not the raw env
        actions = self.oc_env._env.unwrapped.get_action_meanings()

        self.oc_env.reset(seed=self.seed)

        # not possible anymore
        # self.noisy_objects = os.environ["SCOBI_OBJ_EXTRACTOR"] == "Noisy_OC_Atari"
        self.noisy_objects = False

        # use the initialized slots from OC_Atari (ensures that there are no NoObjects)
        init_objects = self.oc_env._slots
        max_obj_dict = self.oc_env.max_objects_per_cat
        self.did_reset = False
        self.focus = Focus(env_name, reward, hide_properties, focus_dir, focus_file, init_objects, max_obj_dict, actions, refresh_yaml, self.logger)
        self.focus_file = self.focus.FOCUSFILEPATH
        self.action_space = spaces.Discrete(len(self.focus.PARSED_ACTIONS))
        self.action_space_description = self.focus.PARSED_ACTIONS
        self.observation_space_description = self.focus.PARSED_PROPERTIES + self.focus.PARSED_FUNCTIONS #this and feature_vector_desc is redundant
        self.feature_vector_description = self.focus.get_feature_vector_description()
        self.num_envs = 1
        self.draw_features = draw_features
        self.feature_attribution = []
        self.render_font = ImageFont.truetype(str(Path(__file__).parent / 'resources' / 'Gidole-Regular.ttf'), size=38)
        self.obj_obs = None  # observation augmented with objects
        self._rel_obs = None  # observation augmented with relations
        self._top_features = []

        self.original_obs = []
        self.original_reward = []
        self.ep_env_reward = None
        self.ep_env_reward_buffer = 0
        self.reset_ep_reward = True

        if reward == 2: # mix rewards
            self._reward_composition_func = lambda a, b : a + b
        elif reward == 1: # scobi only
            self._reward_composition_func = lambda a, b : a
        else: # env only
            self._reward_composition_func = lambda a, b : b

        if self.noisy_objects:
            self.logger.GeneralInfo("Using noisy object detection (default: std 3, detection error rate 5%)")

        self.reset()
        self.step(0) # step once to set the feature vector size
        self.observation_space = spaces.Box(low=-2**63, high=2**63 - 2, shape=(self.focus.OBSERVATION_SIZE,), dtype=np.float32)
        self.ale = self.oc_env._env.unwrapped.ale
        self.reset()
        self.did_reset = False # still require user to properly call a (likely seeded) reset()

    def step(self, action):
        if not self.did_reset:
            self.logger.GeneralError("Cannot call env.step() before calling env.reset()")
        elif self.action_space.contains(action):
            obs, reward, truncated, terminated, info = self.oc_env.step(action)
            sco_obs, sco_reward = self.focus.get_feature_vector(obs)
            freeze_mask = self.focus.get_current_freeze_mask()
            if self.draw_features:
                #for drawing features, we need image here, but obs is ns_repr
                img_obs = self.oc_env._state_buffer_rgb[-1]
                self.obj_obs = self._draw_objects_overlay(img_obs)
                self._rel_obs = self._draw_relation_overlay(img_obs, sco_obs, freeze_mask, action)
            self.original_obs = obs
            self.original_reward = reward
            self.ep_env_reward_buffer += self.original_reward
            if self.reset_ep_reward:
                self.ep_env_reward = None
                self.reset_ep_reward = False
            if terminated or truncated:
                self.ep_env_reward = self.ep_env_reward_buffer
                self.ep_env_reward_buffer = 0
                self.reset_ep_reward = True
                self.focus.reward_subgoals = 0
            final_reward = self._reward_composition_func(sco_reward, reward)
            # self.sco_obs = sco_obs
            return sco_obs, final_reward, truncated, terminated, info # 5
        else:
            raise ValueError("scobi> Action not in action space")

    def reset(self, *args, **kwargs):
        self.did_reset = True
        # additional scobi reset steps here
        self.focus.reward_threshold = -1
        self.focus.reward_history = [0, 0]
        obs, info = self.oc_env.reset(*args, **kwargs)
        sco_obs, _ = self.focus.get_feature_vector(obs)
        return sco_obs, info
    
    @property
    def unwrapped(self):
        return self.oc_env.unwrapped

    def close(self):
        # additional scobi close steps here
        self.oc_env.close()

    def set_feature_attribution(self, att):
        self.feature_attribution = att
    
    # def _mark_bb(self, image_array, bb, color=(255, 0, 0), surround=True):
    #     """
    #     marks a bounding box on the image
    #     """
    #     x, y, w, h = bb
    #     x = int(x)
    #     y = int(y)
    #     if surround:
    #         if x > 0:
    #             x, w = x - 1, w + 1
    #         else:
    #             x, w = x, w
    #         if y > 0:
    #             y, h = y - 1, h + 1
    #         else:
    #             y, h = y, h
    #     bottom = min(209, y + h)
    #     right = min(159, x + w)
    #     image_array[y:bottom + 1, x] = color
    #     image_array[y:bottom + 1, right] = color
    #     image_array[y, x:right + 1] = color
    #     image_array[bottom, x:right + 1] = color


    def _add_margin(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result


    def _draw_objects_overlay(self, obs_image, action=None):
        obs_mod = deepcopy(obs_image)
        for obj in self.oc_env.objects:
            mark_bb(obs_mod, obj.xywh, color=obj.rgb, name=str(obj))
        return obs_mod


    def _draw_relation_overlay(self, obs_image, feature_vector, freeze_mask, action=None):
        scale = 4
        img = Image.fromarray(obs_image)
        draw = ImageDraw.Draw(img, "RGBA")
        if len(self.feature_attribution) == 0:
            img = img.resize((img.size[0]*scale, img.size[1]*scale), resample=Image.BOX)
            # img = self._add_margin(img,0,img.size[0],0,0, (255,255,255))
            return np.array(img)
        features = self.feature_vector_description[0]
        fv_backmap = self.feature_vector_description[1]
        i = 0
        top_features_k = 5
        top_features_names = ["" for _ in range(top_features_k)]
        if np.ptp(self.feature_attribution):
            feature_attribution = (255*(self.feature_attribution - np.min(self.feature_attribution))/np.ptp(self.feature_attribution)).astype(int)
            top_features_idxs = np.argsort(feature_attribution)[-top_features_k:][::-1]
            for feature in features:
                i += 1
                idxs = np.where(fv_backmap == i-1)[0]
                feature_name = feature[0]
                feature_signature = feature[1]
                fv_entries = feature_vector[idxs[0]:idxs[-1]+1]
                fv_attribution = feature_attribution[idxs[0]:idxs[-1]+1]
                fv_freeze_mask = freeze_mask[idxs[0]:idxs[-1]+1]
                alpha = int(np.mean(fv_attribution)**2/255)
                for ii, idx in enumerate(idxs):
                    if idx in top_features_idxs:
                        k_idx = np.where(top_features_idxs == idx)[0][0]
                        top_features_names[k_idx] = format_feature(feature_name, feature_signature, ii)
                if 0 in fv_freeze_mask:
                    continue
                if feature_name == "POSITION":
                    radius = 2
                    x = fv_entries[0]
                    y = fv_entries[1]
                    coords = (x - radius, y - radius, x + radius, y + radius)
                    draw.ellipse(coords, fill=(0,0,0,alpha), outline=(0,0,0,alpha))
                elif feature_name == "POSITION_HISTORY":
                    radius = 2
                    x_t = fv_entries[2]
                    y_t = fv_entries[3]
                    coords_then = (x_t - radius, y_t - radius, x_t + radius, y_t + radius)
                    draw.ellipse(coords_then, fill=(0,0,0,alpha), outline=(0,0,0, alpha))
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
                    draw.line(source_object_coord_values + vector.tolist(), fill=(0,255,255,alpha), width=1)
                elif feature_name == "DIR_VELOCITY":
                    velocity_scaling = 2
                    velocity_vector = [fv_entries[0], fv_entries[1]]
                    velocity_vector = np.multiply(velocity_vector, velocity_scaling)
                    source_object_phistory = feature_signature[0]
                    idx = -1
                    for f in features:
                        idx += 1
                        if f == source_object_phistory:
                            break
                    idxs = np.where(fv_backmap == idx)[0]
                    source_object_phistory_values = feature_vector[idxs[0]:idxs[-1]+1]
                    current_coords = source_object_phistory_values[:2]
                    vector = np.subtract(current_coords, velocity_vector).tolist()
                    draw.line(current_coords + vector, fill=(0,255,255,alpha), width=2)
                elif feature_name == "VELOCITY":
                    velocity_scaling = 2
                    velocity_value = [0, fv_entries[0]] #draw velocity as vertical bar
                    velocity_vector = np.multiply(velocity_value, velocity_scaling)
                    source_object_phistory = feature_signature[0]
                    idx = -1
                    for f in features:
                        idx += 1
                        if f == source_object_phistory:
                            break
                    idxs = np.where(fv_backmap == idx)[0]
                    source_object_phistory_values = feature_vector[idxs[0]:idxs[-1]+1]
                    current_coords = source_object_phistory_values[:2]
                    vector = np.subtract(current_coords, velocity_vector).tolist()
                    draw.line(current_coords + vector, fill=(0,255,255,alpha), width=2)
        #print(top_features_names)
        # import ipdb; ipdb.set_trace()
        img = img.resize((img.size[0]*scale, img.size[1]*scale), resample=Image.BOX)
        self._top_features = top_features_names
        # img = self._add_margin(img,0,img.size[0],0,0, (255,255,255))
        # draw = ImageDraw.Draw(img, "RGBA")
        # draw.text((img.size[0]/2 +20, 50), to_draw, (5, 5, 5), self.render_font)
        return np.array(img)

    def get_vector_entry_descriptions(self):
        features = self.feature_vector_description[0]
        fv_backmap = self.feature_vector_description[1]
        i = 0
        features_names = []
        for feature in features:
            idxs = np.where(fv_backmap == i)[0]
            feature_name = feature[0]
            feature_signature = feature[1]
            for ii, idx in enumerate(idxs):
                features_names.append(format_feature(feature_name, feature_signature, ii))
            i += 1
        return features_names


def format_feature(feature_name, feature_signature, ii):
    if feature_name == 'RGB':
        axis = ["R", "G", "B"][ii]
        return f"RGB({feature_signature}.{axis})"
    if feature_name == "POSITION_HISTORY":
        if ii < 2:
            axis = ["x", "y"][ii]
            return f"{feature_signature}.{axis}"
        axis = ["x", "y"][ii-2]
        return f"{feature_signature}.{axis}[t-1]"
    axis = ["x", "y"][ii]
    if ii > 3:
        print("feature render formatting error. exiting...")
        exit()
    if feature_name == 'POSITION':
        return f"{feature_signature}.{axis}"
    elif feature_name == "EUCLIDEAN_DISTANCE":
        return f"ED({feature_signature[0][1]}, {feature_signature[1][1]})"
    elif feature_name == "DISTANCE":
        return f"D({feature_signature[0][1]}, {feature_signature[1][1]}).{axis}"
    elif feature_name == "VELOCITY":
        return f"V({feature_signature[0][1]}).{axis}"
    elif feature_name == "DIR_VELOCITY":
        return f"DV({feature_signature[0][1]}).{axis}"
    elif feature_name == "CENTER":
        return f"C({feature_signature[0][1]}, {feature_signature[1][1]}).{axis}"
    elif feature_name == "ORIENTATION":
        return f"O({feature_signature})"
    elif feature_name == "LINEAR_TRAJECTORY":
        return f"LT({feature_signature[0][1]}, {feature_signature[1][1]}).{axis}"
    elif feature_name == "COLOR":
        return f"COL({feature_signature[0][1]})"
    print("feature render formatting error. exiting...")
    exit()


def _make_darker(color, col_precent=0.8):
    """
    return a darker color
    """
    if not color:
        print("No color passed, using default black")
        return [0, 0, 0]
    return [int(col * col_precent) for col in color]


def mark_bb(image_array, bb, color=(255, 0, 0), surround=True, name=None):
    """
    marks a bounding box on the image
    """
    x, y, w, h = bb
    color = _make_darker(color)
    if surround:
        if x > 0:
            x, w = bb[0] - 1, bb[2] + 1
        else:
            x, w = bb[0], bb[2]
        if y > 0:
            y, h = bb[1] - 1, bb[3] + 1
        else:
            y, h = bb[1], bb[3]
    y = min(209, y)
    x = min(159, x)
    bottom = min(209, y + h)
    right = min(159, x + w)
    image_array[y:bottom + 1, x] = color
    image_array[y:bottom + 1, right] = color
    image_array[y, x:right + 1] = color
    image_array[bottom, x:right + 1] = color