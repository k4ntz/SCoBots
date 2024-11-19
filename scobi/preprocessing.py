from scobi.focus import Focus
import numpy as np
from typing import Callable


NORMALIZATION_FNS = {
    "POSITION_HISTORY": lambda x: x / 164,  # any object coords
    "POSITION": lambda x: x / 164,  # any object coords
    "DISTANCE": lambda x: x / 164,  # any object coords
    "LINEAR_TRAJECTORY": lambda x: x / 164,  # any object coords
    "EUCLIDEAN_DISTANCE": lambda x: x / 164,  # any object coords
    "CENTER": lambda x: x / 164,  # any object coords
    "DIR_VELOCITY": lambda x: np.tanh(x / 5),  # velocities (typically inside [-5, 5])
    "VELOCITY": lambda x: np.tanh(x / 5),  # velocities (typically inside [-5, 5])
    "ORIENTATION": lambda x: x / 15,  # inside [0, 15]
    # "ORIENTATION": lambda x: x / 12,  # kangaroo either 4 or 12
    "WIDTH": lambda x: x / 64,  # oxygen bar
    "VALUE": lambda x: x / 3,  # lives
    # "VALUE": lambda x: x / 2000,  # kangaroo time [0, 2000]
    "COUNT": lambda x: x / 6,  # collected divers
    # "COUNT": lambda x: x / 2,  # kangaroo lives 
    "RGB": lambda x: x / 255,  # color
    "COLOR": lambda x: x / 255,  # color
}

# kangarrooo
# pos-hist: [20 - 109]
# orientation: either 4 or 12
# count: 2
# value: 0

#['POSITION_HISTORY', 'Player1']
# ['POSITION', 'Child1']
# ['POSITION', 'Fruit1']
# ['POSITION', 'Fruit2']
# ['POSITION', 'Fruit3']
# ['POSITION', 'Bell1']
# ['POSITION', 'Platform1']
# ['POSITION', 'Platform2']
# ['POSITION', 'Platform3']
# ['POSITION', 'Platform4']
# ['POSITION', 'Platform5']
# ['POSITION', 'Platform6']
# ['POSITION', 'Platform7']
# ['POSITION', 'Platform8']
# ['POSITION', 'Platform9']
# ['POSITION', 'Platform10']
# ['POSITION', 'Platform11']
# ['POSITION', 'Platform12']
# ['POSITION', 'Platform13']
# ['POSITION', 'Platform14']
# ['POSITION', 'Platform15']
# ['POSITION', 'Platform16']
# ['POSITION', 'Platform17']
# ['POSITION', 'Platform18']
# ['POSITION', 'Platform19']
# ['POSITION', 'Platform20']
# ['POSITION', 'Ladder1']
# ['POSITION', 'Ladder2']
# ['POSITION', 'Ladder3']
# ['POSITION', 'Ladder4']
# ['POSITION', 'Ladder5']
# ['POSITION', 'Ladder6']
# ['POSITION_HISTORY', 'Monkey1']
# ['POSITION_HISTORY', 'Monkey2']
# ['POSITION_HISTORY', 'Monkey3']
# ['POSITION_HISTORY', 'Monkey4']
# ['POSITION_HISTORY', 'FallingCoconut1']
# ['POSITION_HISTORY', 'ThrownCoconut1']
# ['POSITION_HISTORY', 'ThrownCoconut2']
# ['POSITION_HISTORY', 'ThrownCoconut3']
# ['VALUE', 'Time1']
# ['ORIENTATION', 'Player1']
# ['COUNT', [['category', 'Life']]]

class Normalizer:
    """Normalizes a given feature vector according to the concepts as
    specified in the focus file."""
    # normalization_functions: list[Callable[[float], float]]  # for each concept one normalization function

    def __init__(self, focus: Focus):
        self.focus = focus
        self.feature_backmap = np.asarray(self.focus.FEATURE_VECTOR_BACKMAP)
        self._setup_normalization_functions()

    def _setup_normalization_functions(self):
        concepts = self.focus.PARSED_CONCEPTS
        self.normalization_functions = []
        for concept in concepts:
            concept_name = concept[0]
            normalization_function = NORMALIZATION_FNS[concept_name]
            self.normalization_functions.append(normalization_function)

    def normalize(self, features: np.array):
        """Takes a feature vector and normalizes its values."""
        assert len(features) == self.focus.FEATURE_VECTOR_SIZE

        features = features.copy()
        # print("player: ", features[0])
        # print("monkey: ", features[-11])
        # print("VALUE: ", features[-3])
        # print("ORIENTATION: ", features[-2])
        # print("COUNT: ", features[-1])

        for i, normalization_fn in enumerate(self.normalization_functions):
            this_feature = self.feature_backmap == i
            features[this_feature] = normalization_fn(features[this_feature])

        return features

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)