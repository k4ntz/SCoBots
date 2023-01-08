import numpy as np
import math
from typing import Tuple
from scobi.utils.game_object import GameObject
from scobi.utils.colors import get_closest_color
from scobi.utils.decorators import register

COLOR_INT_MEMORY = dict()
EPS = np.finfo(np.float32).eps.item()

# dummy init 
def init():
    pass

##########################
# PROPERTIES TO REGISTER
##########################
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


##########################
# FUNCTIONS TO REGISTER
##########################
@register(type="F", name="LINEAR_TRAJECTORY", params=["POSITION", "POSITION_HISTORY"], desc="x, y distance to trajectory")
def calc_lin_traj(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    obj1 = a_position
    obj2 = b_history[0:2]
    obj2_past = b_history[2:4]
    x, y = obj2, obj2_past
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    disty = np.int16(m * obj1[0] + c) - obj1[1]
    distx = np.int16((obj1[1] - c) / (m+EPS))  - obj1[0]
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


@register(type="F", name="COLOR", params=["RGB"], desc="Index of colorname")
def get_color_name(rgb: Tuple[int, int, int]) -> int:
    # only calc distances if new unseen rgb value
    if rgb in COLOR_INT_MEMORY.keys():
        return COLOR_INT_MEMORY[rgb]
    else:
        _, col_int = get_closest_color(rgb)
        return col_int