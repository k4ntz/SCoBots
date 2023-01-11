# class for game objects
# containing current and past coordinates
# and rgb values

import numpy as np

class GameObject():
    def __init__(self, name, rgb=[0,0,0], wh=[0,0]):
        self.name = name
        # contains current and old coords
        self.coord = [0,0,0,0]
        self.rgb = rgb
        self.wh = wh
        self.category = "None"
    # set new coords
    def update_coords(self, x, y):
        self.coord = np.roll(self.coord, 2)
        self.coord[0] = x
        self.coord[1] = y

    # returns 2 lists with current and past coords
    def get_coords(self):
        return [self.coord[0], self.coord[1]], [self.coord[2], self.coord[3]]



