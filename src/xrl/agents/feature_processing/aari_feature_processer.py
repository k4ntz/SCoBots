# file for raw feature processing to meaningful features
# with atari ari gym environments

import numpy as np
import math
import inspect
from typing import Tuple
from xrl.agents.game_object import GameObject
from utils_color_categorisation import _colordist, COLOR_TO_CATEGORY, \
    colors_matrices


FUNCTIONS = dict()
PROPERTIES = dict()
eps = np.finfo(np.float32).eps.item()

# decorator to register properties and functions
def register(*args, **kwargs):

    def inner(func):
        sig = inspect.signature(func)
        ret_ano = sig.return_annotation
        sig_list = sig.parameters
        param_descs = kwargs["params"]
        ret_desc = kwargs["desc"]
        sig_dict = {"object": func, "expects": [], "returns": None}
        sig_dict["returns"] = (ret_ano, ret_desc)
        if len(sig_list) == len(param_descs):
            for k in sig_list.keys():
                desc = param_descs.pop(0)
                sig_dict["expects"].append((sig_list[k], desc))
        else:
            print("id error")
        name = kwargs["name"]
        if name in FUNCTIONS.keys():
            print("name already registered")
        else:
            if kwargs["type"] == "F": # function
                FUNCTIONS[kwargs["name"]] = sig_dict
            elif kwargs["type"] == "P": # property
                PROPERTIES[kwargs["name"]] = sig_dict
            else:
                print("unknown type")
    return inner


# REGISTERED PROPERTIES
@register(type="P", name="POSITION", params= ["OBJECT"], desc="get the position for given object")
def get_position(obj: GameObject) -> Tuple[int, int]:
    return tuple(obj.get_coords()[0])

@register(type="P", name="POSITION_HISTORY", params= ["OBJECT"], desc="get the current and last position for given object")
def get_position_history(obj: GameObject) -> Tuple[int, int, int, int]:
    coords = obj.get_coords()
    return tuple(coords[0] + coords[1])

@register(type="P", name="RGB", params= ["OBJECT"], desc="get the rgb value for given object")
def get_rgb(obj: GameObject) -> Tuple[int, int, int]:
    return tuple(obj.rgb)


# REGISTERED FUNCTIONS
@register(type="F", name="LINEAR_TRAJECTORY", params=["POSITION", "POSITION_HISTORY"], desc="x, y distance to trajectory")
def calc_lin_traj(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    obj1 = a_position
    obj2 = b_history[0:2]
    obj2_past = b_history[2:4]
    # append trajectory cutting points
    m, c = _get_lineq_param(obj2, obj2_past)
    # now calc target pos
    # y = mx + c substracted from its y pos
    disty = np.int16(m * obj1[0] + c) - obj1[1]
    # x = (y - c)/m substracted from its x pos
    distx = np.int16((obj1[1] - c) / (m+eps))  - obj1[0]
    return disty, distx


@register(type="F", name="DISTANCE", params=["POSITION", "POSITION"], desc="distance between two coordinates")
def calc_distance(a_position: Tuple[int, int], b_position: Tuple[int, int]) -> Tuple[int, int]:
    distx = b_position[0] - a_position[0]
    disty = b_position[1] - a_position[1]
    return distx, disty


@register(type="F", name="EUCLIDEAN_DISTANCE", params=["POSITION", "POSITION"], desc="euclidean distance between two coordinates")
def calc_euclidean_distance(a_position: Tuple[int, int], b_position: Tuple[int, int]) -> float:
    dist = math.sqrt((b_position[1] - a_position[1])**2 + (b_position[0] - a_position[0])**2)
    return dist


@register(type="F", name="CENTER", params=["POSITION", "POSITION"], desc="center position of two objects")
def get_center(a_position: Tuple[int, int], b_position: Tuple[int, int]) -> Tuple[int, int]:
    return (a_position[0] + b_position[0])/2, (a_position[1] + b_position[1])/2


@register(type="F", name="VELOCITY", params=["POSITION_HISTORY"], desc="velocity of object")
def get_velocity(pos_history: Tuple[int, int, int, int]) -> float:
    obj = pos_history[0:2]
    obj_past = pos_history[2:4]
    vel = math.sqrt((obj_past[0] - obj[0])**2 + (obj_past[1] - obj[1])**2)
    return vel


# helper function to calc linear equation
def _get_lineq_param(obj1, obj2):
    x = obj1
    y = obj2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# TODO: remove dummy function
def calc_preset_mifs(temp):
    return temp


def get_color_name(rgb: Tuple[int, int, int]) -> str:
    comp_list = [(c[0], _colordist(rgb, c[1])) for c in colors_matrices.items()]
    comp_list2 = sorted(comp_list, key=lambda el: el[1])
    return COLOR_TO_CATEGORY[comp_list2[0][0]]
