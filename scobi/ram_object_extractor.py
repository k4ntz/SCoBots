from ocatari.ram.game_objects import GameObject
from scobi.object_extractor import ObjectExtractor
from typing import List

class RAMObjectExtractor(ObjectExtractor):
    def __init__(self, oc_env):
        self.oc_env = oc_env

    def get_objects(self, img) -> List[GameObject]:
        return self.oc_env.objects