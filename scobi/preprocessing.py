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
    "WIDTH": lambda x: x / 64,  # oxygen bar
    "VALUE": lambda x: x / 3,  # lives
    "COUNT": lambda x: x / 6,  # collected divers
    "RGB": lambda x: x / 255,  # color
    "COLOR": lambda x: x / 255,  # color
}


class Normalizer:
    """Normalizes a given feature vector according to the concepts as
    specified in the focus file."""
    normalization_functions: list[Callable[[float], float]]  # for each concept one normalization function

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

        for i, normalization_fn in enumerate(self.normalization_functions):
            this_feature = self.feature_backmap == i
            features[this_feature] = normalization_fn(features[this_feature])

        return features

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)
