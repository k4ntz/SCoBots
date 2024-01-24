# game object interface
# switch depending on object extractor
# only ocatari implemented for now
from scobi.utils.interfaces import GameObjectInterface

from ocatari.ram.game_objects import GameObject as Ocatari_GameObject
from scobi.utils.SPACEGameObject import KFandSPACEGameObject

def get_wrapper_class(game_object_extractor):
    if game_object_extractor == "OC_Atari":
        return OCAGameObjectWrapped
    # add other object extractors here and its wrapper classe below
    elif game_object_extractor == "KFandSPACE":
        return KFandSPACEGameObjectWrapped


# OC Atari GameObject wrapper classes implementing scobi GameObjectInterface
#Format for coordinates:
# x and y are the center coordinates of the bbox
# w and h are the width and height of the bbox

class OCAGameObjectWrapped(GameObjectInterface):
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
        self.ocgo.xy = xy[0] - int(self.w / 2), xy[1] - int(self.h / 2)

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
        x_min, y_min, w, h = self.ocgo.xywh
        return x_min + int(w / 2), y_min + int(h / 2), w, h

    @property
    def rgb(self):
        if len(self.ocgo.rgb) != 3:
            raise ValueError(f"Bad rgb dimension from ocatari: {self.name} : {self.ocgo.rgb}")
        return self.ocgo.rgb
    
    @property
    def orientation(self):
        return self.ocgo.orientation
    

class KFandSPACEGameObjectWrapped(GameObjectInterface):

    def __init__(self, kfandspacego):
        self._number = 1
        if issubclass(type(kfandspacego), KFandSPACEGameObject):
            self.kfandspacego = kfandspacego
        elif issubclass(type(kfandspacego), Ocatari_GameObject):
            self.kfandspacego = kfandspacego
        else:
            incoming_type = type(kfandspacego)
            raise ValueError("Incompatible Wrapper, expects KFandSPACEGameObject. Got: "+str(incoming_type))

    @property
    def category(self):
        return self.kfandspacego.category
    
    @property
    def number(self):
        return self._number
        
    @number.setter
    def number(self, number):
        self._number = number

    @property
    def xy(self):
        if len(self.kfandspacego.xy) != 2:
            raise ValueError(f"Bad xy dimension: {self.name} : {self.kfandspacego.xy}") #TODO: generalize and improve dimension checks
        x = self.kfandspacego.xy[0] + int(self.w / 2)
        y = self.kfandspacego.xy[1] + int(self.h / 2)
        return x, y
    
    @xy.setter
    def xy(self, xy):
        if len(self.kfandspacego.xy) != 2:
            raise ValueError(f"Bad xy dimension: {self.name} : {self.ocgo.xy}")
        self.kfandspacego.xy = xy[0] - int(self.w / 2), xy[1] - int(self.h / 2)
    
    @property
    def w(self):
        return self.kfandspacego.w
    
    @property
    def h(self):
        return self.kfandspacego.h
    
    @property
    def rgb(self): #TODO: decide what to do with rgb
        return self.kfandspacego.rgb
        #if self.category == "Player":
        #    return  (92, 186, 92) #Boxing (214, 214, 214) # Pong (92, 186, 92)
        #elif self.category == "Ball":
        #    return (236, 236, 236)
        #elif self.category == "Enemy":
        #    return (213, 130, 74)#Boxing (0, 0, 0)  # Pong (213, 130, 74)
        #else:
        #    return (0, 0, 0)
    
    @property
    def xywh(self):
        x_min, y_min, w, h = self.kfandspacego.xywh
        return x_min + int(w / 2), y_min + int(h / 2), w, h
    
    @property
    def orientation(self):
        return self.kfandspacego.orientation
    
    @property
    def h_coords(self):
        shc = self.kfandspacego.h_coords
        if None in shc or len(shc) != 2 or len([*shc[0], *shc[1]]) != 4:
            raise ValueError(f"Bad h_coords dimension{self.name} : {self.kfandspacego.h_coords}")
        ncord = shc[0][0] + int(self.w / 2), shc[0][1] + int(self.h / 2)
        ocord = shc[1][0] + int(self.w / 2), shc[1][1] + int(self.h / 2)
        return ncord, ocord
