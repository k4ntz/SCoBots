# game object interface
# switch depending on object extractor
# only ocatari implemented for now
from scobi.utils.interfaces import GameObjectInterface
from typing import Tuple
from ocatari.ram.game_objects import GameObject as Ocatari_GameObject

OBJ_EXTRACTOR = "OC_Atari" #TODO: pass or fetch from global env


def get_wrapper_class():
    if OBJ_EXTRACTOR == "OC_Atari":
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
    def xy(self) -> tuple[float, float]:
        if len(self.ocgo.xy) != 2:
            raise ValueError(f"Bad xy dimension from ocatari: {self.name} : {self.ocgo.xy}") #TODO: generalize and improve dimension checks
        return self.ocgo.xy

    @xy.setter
    def xy(self, xy):
        if len(self.ocgo.xy) != 2:
            raise ValueError(f"Bad xy dimension from ocatari: {self.name} : {self.ocgo.xy}")
        self.ocgo.xy = xy

    @property
    def h_coords(self):
        shc = self.ocgo.h_coords
        if len(shc) != 2 or len([*shc[0], *shc[1]]) != 4:
            raise ValueError(f"Bad h_coords dimension from ocatari: {self.name} : {self.ocgo.h_coords}")
        return self.ocgo.h_coords
    
    @property
    def rgb(self):
        if len(self.ocgo.rgb) != 3:
            raise ValueError(f"Bad rgb dimension from ocatari: {self.name} : {self.ocgo.rgb}")
        return self.ocgo.rgb
    
    @property
    def visible(self):
        return self.ocgo.visible