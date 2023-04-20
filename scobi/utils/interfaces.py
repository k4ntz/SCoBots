# game object interface
from abc import ABC, abstractmethod
import math

class GameObjectInterface(ABC):
    
    @property
    @abstractmethod
    def category(self):
        pass

    @property
    @abstractmethod
    def xy(self):
        pass

    @xy.setter
    @abstractmethod
    def xy(self, xy):
        pass

    @property
    @abstractmethod
    def orientation(self):
        pass

    @orientation.setter
    @abstractmethod
    def orientation(self, o):
        pass

    @property
    @abstractmethod
    def h_coords(self):
        pass

    @property
    @abstractmethod
    def w(self):
        pass

    @property
    @abstractmethod
    def h(self):
        pass

    @property
    @abstractmethod
    def rgb(self):
        pass
    
    @property
    @abstractmethod
    def number(self):
        pass
    
    @number.setter
    @abstractmethod
    def number(self, number):
        pass

    #@property
    #@abstractmethod
    #def visible(self):
    #    pass

    # default behaviour
    @property
    def name(self):
        return str(self.category) + str(self.number)
    
    def distance(self, game_object):
        return math.sqrt((self.xy[0] - game_object.xy[0])**2 + (self.xy[1] - game_object.xy[1])**2)

    def x_distance(self, game_object):
        return self.xy[0] - game_object.xy[0]

    def y_distance(self, game_object):
        return self.xy[1] - game_object.xy[1]

    def __repr__(self):
        return f"{self.name} at ({self.xy[0]}, {self.xy[1]})"
    
    # @property
    # @abstractmethod
    # def dx(self):
    #     pass

    # @property
    # @abstractmethod
    # def dy(self):
    #     pass

    # @property
    # @abstractmethod
    # def xywh(self):
    #     pass

    # @property
    # @abstractmethod
    # def x(self):
    #     pass

    # @property
    # @abstractmethod
    # def y(self):
    #     pass


