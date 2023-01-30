# game object interface
from abc import ABC, abstractmethod


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
    def h_coords(self):
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

    @property
    @abstractmethod
    def visible(self):
        pass

    # default behaviour
    @property
    def name(self):
        return str(self.category) + str(self.number)
    
    def __repr__(self):
        visible_status = "V" if self.visible else "H"
        return f"{self.name}({visible_status}) at ({self.xy[0]}, {self.xy[1]})"
    
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

    # @property
    # @abstractmethod
    # def w(self):
    #     pass

    # @property
    # @abstractmethod
    # def h(self):
    #     pass

