from ocatari.ram.pong import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_PONG
from ocatari.ram.boxing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_BOXING
from ocatari.ram.tennis import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_TENNIS
from ocatari.ram.skiing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SKIING
from ocatari.ram.carnival import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_CARNIVAL
from ocatari.ram.spaceinvaders import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SPACE_INVADERS
from ocatari.ram.riverraid import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_RIVER_RAID
from ocatari.ram.mspacman import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_MSPACMAN

from ocatari.ram.pong import Player as pongPlayer, Ball as pongBall, Enemy as pongEnemy
from ocatari.ram.boxing import Player as boxingPlayer, Enemy as boxingEnemy

#def get_max_nb_objects_hud_for_game(env_name):
#    if "pong" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_PONG
#    elif "boxing" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_BOXING
#    elif "tennis" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_TENNIS
#    elif "skiing" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_SKIING
#    elif "carnival" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_CARNIVAL
#    elif "spaceinvaders" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_SPACE_INVADERS
#    elif "riverraid" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_RIVER_RAID
#    elif "mspacman" in env_name.lower():
#        return MAX_NB_OBJECTS_ALL_MSPACMAN
#    else:
#        raise ValueError("scobi env_name not recognized")

#def create_game_object_list(env_name):
#    max_nb_objects_hud = get_max_nb_objects_hud_for_game(env_name)
#    game_object_list = []
#    for object_name, max_instances in max_nb_objects_hud.items():
#        for _ in range(max_instances):
#            game_object = KFandSPOCGameObject(
#                x_min=0,
#                y_min=0,
#                w=0,
#                h=0,
#                class_id= sorted_label_list_for(env_name).index(object_name),
#                confidence= 0.5, #TODO check which value to put here
#            )
#            game_object_list.append(game_object)
#    return game_object_lis

def sorted_label_list_for(env_name):
    no_label_str = "no_label"
    if "pong" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_PONG.keys()))
    elif "boxing" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_BOXING.keys()))
    elif "tennis" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_TENNIS.keys()))
    elif "skiing" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_SKIING.keys()))
    elif "carnival" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_CARNIVAL.keys()))
    elif "spaceinvaders" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_SPACE_INVADERS.keys()))
    elif "riverraid" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_RIVER_RAID.keys()))
    elif "mspacman" in env_name.lower():
        return [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_MSPACMAN.keys()))
    else:
        raise ValueError("scobi env_name not recognized")

class KFandSPACEGameObject:

    def __init__(self, x_min, y_min, w, h, class_id, confidence, game_name):
        self._number = 1
        self._x_min = x_min
        self._y_min = y_min
        self._w = w
        self._h = h
        self._class_id = class_id
        self._confidence = confidence
        self._prev_xy = None
        self.game_name = game_name

    @property
    def class_id(self):
        return self._class_id
    
    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id

    @property
    def category(self):
        return sorted_label_list_for(self.game_name)[self._class_id]
    
    @property
    def xy(self):
        x = self._x_min + int(self._w / 2)
        y = self._y_min + int(self._h / 2)
        return x, y

    @xy.setter
    def xy(self, xy):
        self._x_min = xy[0] - int(self._w / 2)
        self._y_min = xy[1] - int(self._h / 2)

    @property
    def number(self):
        return self._number
        
    @number.setter
    def number(self, number):
        self._number = number

    @property
    def h(self):
        return self._h
    
    @h.setter
    def h(self, h):
        self._h = h

    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, w):
        self._w = w

    @property
    def xywh(self):
        return self._x_min , self._y_min, self._w, self._h

    @property
    def rgb(self):
        eval_str = self.game_name + self.category
        return eval(eval_str)().rgb

    
    @property
    def orientation(self):
        return 0 #TODO
    
    @property
    def h_coords(self):
        ncord = self.xy[0], self.xy[1]
        ocord = self.prev_xy[0], self.prev_xy[1]
        return ncord, ocord
        

    @property
    def prev_xy(self):
        if self._prev_xy is not None:
            return self._prev_xy
        else:
            return self.xy

    @prev_xy.setter
    def prev_xy(self, newval):
        self._prev_xy = newval