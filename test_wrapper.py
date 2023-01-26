from scobi.utils.game_object import get_wrapper_class
from ocatari.ram.game_objects import GameObject as OCA

oca = OCA()
print(oca)
GameObject = get_wrapper_class()
a = GameObject(oca)
print(a)
print(a.category)
print(a.xy)
a.xy = (1, 0)
print(a.xy)
print(a.h_coords)