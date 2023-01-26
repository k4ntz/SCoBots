# game object interface
# switch depending on object extractor
# hardoced to ocatari for now
from scobi.utils.interfaces import GameObjectInterface

from ocatari.ram.game_objects import GameObject as Ocatari_GameObject

OBJ_EXTRACTOR = "OC_Atari" #TODO: pass or fetch from global env


def get_wrapper_class():
    if OBJ_EXTRACTOR == "OC_Atari":
        return OCAGameObject
    # add other object extractors here and its wrapper classe below


# OC Atari GameObject wrapper classes implementing scobi GameObjectInterface
class OCAGameObject(GameObjectInterface):
    def __init__(self, ocgo):
        if issubclass(type(ocgo), Ocatari_GameObject):
            self.ocgo = ocgo
        else:
            incoming_type = type(ocgo)
            raise ValueError("Incompatible Wrapper, expects OC_Atari GameObject. Got: "+str(incoming_type))

    @property
    def category(self):
        return self.ocgo.category
    
    @property
    def xy(self):
        return self.ocgo.xy

    @xy.setter
    def xy(self, xy):
        self.ocgo.xy = xy

    @property
    def h_coords(self):
        return self.ocgo.h_coords
    
    @property
    def rgb(self):
        return self.ocgo.rgb