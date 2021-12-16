"""
Agents
"""

class Agent():
    def __init__(self, components):
        for el in components:
            if not callable(el):
                print("You should build an agent with callable components")
        self.pipeline = components

    def __call__(self, x):
        for func in self.pipeline:
            x = func(x)
        return x

    def choose(self):
        pass

    def __repr__(self):
        str_ret = "Pipeline Agent consisting of:\n"
        for feat in self.pipeline:
            if "__name__" in dir(feat):
                str_ret += f"\t-> {feat.__name__}\n"
            else:
                str_ret += f"\t-> {feat}\n"
        return str_ret
