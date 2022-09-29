import random

class RandomPolicy():
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.__name__ = f"RandomPolicy ({n_actions} possible actions)"

    def __call__(self, x):
        return random.randint(0, self.n_actions - 1)
