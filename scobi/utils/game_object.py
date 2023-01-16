# class for game objects
# containing current and past coordinates
# and rgb values


class GameObject():
    def __init__(self, name, rgb=[0,0,0], wh=[0,0]):
        self.name = name
        # contains current and old coords
        self.coord_now = [0, 0]
        self.coord_past = [0, 0]
        self.rgb = rgb
        self.wh = wh
        self.category = "None"
    # set new coords
    def update_coords(self, x, y):
        self.coord_past = self.coord_now
        self.coord_now[0] = x
        self.coord_now[1] = y

    # returns 2 lists with current and past coords
    def get_coords(self):
        return self.coord_now, self.coord_past
