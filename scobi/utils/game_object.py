# game object interface
# switch depending on object extractor
# only ocatari implemented for now
from scobi.utils.interfaces import GameObjectInterface
from typing import Tuple
from ocatari.ram.game_objects import GameObject as Ocatari_GameObject
import numpy as np
import os


def get_wrapper_class():
    if not "SCOBI_OBJ_EXTRACTOR" in os.environ:
        os.environ["SCOBI_OBJ_EXTRACTOR"] = "OC_ATARI"
        print("Set env var 'SCOBI_OBJ_EXTRACTOR' as 'OC_Atari'. Other option is 'Noisy_OC_Atari' which can be set manually")
    if os.environ["SCOBI_OBJ_EXTRACTOR"] == "Noisy_OC_Atari":
        return NoisyOCAGameObject
    elif os.environ["SCOBI_OBJ_EXTRACTOR"] == "OC_Atari":
        return OCAGameObject
    else:
        return OCAGameObject
    # add other object extractors here and its wrapper classe below


# OC Atari GameObject wrapper classes implementing scobi GameObjectInterface
class OCAGameObject(GameObjectInterface):
    def __init__(self, ocgo):
        self._number = 1
        if issubclass(type(ocgo), Ocatari_GameObject):
            self.ocgo = ocgo
        else:
            incoming_type = type(ocgo)
            raise ValueError("Incompatible Wrapper, expects OC_Atari GameObject. Got: "+str(incoming_type))


    @property
    def category(self):
        return self.ocgo.category
    
    @property
    def number(self):
        return self._number
        
    @number.setter
    def number(self, number):
        self._number = number
    
    @property
    def xy(self):
        if len(self.ocgo.xy) != 2:
            raise ValueError(f"Bad xy dimension from ocatari: {self.name} : {self.ocgo.xy}") #TODO: generalize and improve dimension checks
        x = self.ocgo.xy[0] + int(self.w / 2)
        y = self.ocgo.xy[1] + int(self.h / 2)
        return x, y

    @xy.setter
    def xy(self, xy):
        if len(self.ocgo.xy) != 2:
            raise ValueError(f"Bad xy dimension from ocatari: {self.name} : {self.ocgo.xy}")
        self.ocgo.xy = xy

    @property
    def dx(self):
        return self.ocgo.dx

    @property
    def dy(self):
        return self.ocgo.dy

    @property
    def h_coords(self):
        shc = self.ocgo.h_coords
        if None in shc or len(shc) != 2 or len([*shc[0], *shc[1]]) != 4:
            raise ValueError(f"Bad h_coords dimension from ocatari: {self.name} : {self.ocgo.h_coords}")
        ncord = shc[0][0] + int(self.w / 2), shc[0][1] + int(self.h / 2)
        ocord = shc[1][0] + int(self.w / 2), shc[1][1] + int(self.h / 2)
        return ncord, ocord
    
    @property
    def w(self):
        return self.ocgo.w

    @property
    def h(self):
        return self.ocgo.h
    
    @property
    def xywh(self):
        return self.ocgo.xywh

    @property
    def rgb(self):
        if len(self.ocgo.rgb) != 3:
            raise ValueError(f"Bad rgb dimension from ocatari: {self.name} : {self.ocgo.rgb}")
        return self.ocgo.rgb
    
    @property
    def orientation(self):
        return self.ocgo.orientation
    
    @property
    def value_diff(self):
        if not hasattr(self.ocgo, "value_diff"):
            return None
        return self.ocgo.value_diff
    

class NoisyOCAGameObject(OCAGameObject):
    def __init__(self, ocgo, std, error_rate, random_state):
        super().__init__(ocgo)
        self.std = std
        self.error_rate = error_rate
        self.random_state = random_state


    def _add_noise(self, xy):
        x_noise = self.random_state.normal(0.0, self.std)
        y_noise = self.random_state.normal(0.0, self.std)
        return xy[0] + x_noise, xy[1] + y_noise

    @property
    def xy(self):
        if self.random_state.rand(1)[0]  <= self.error_rate:
            x, y = self._add_noise(super().h_coords[1]) # noise and return past coord
        else:
            x, y = self._add_noise(super().xy)
        return x, y
    
    @property
    def h_coords(self):
        x, y = self._add_noise(super().h_coords[0])
        x_old, y_old = self._add_noise(super().h_coords[1])
        if self.random_state.rand(1)[0]  <= self.error_rate:
            return (x_old, y_old), (x_old, y_old)
        else:
            return (x, y), (x_old, y_old)
