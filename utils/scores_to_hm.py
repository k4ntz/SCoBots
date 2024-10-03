import pandas as pd
import numpy as np


class MNSTD(object):
    def __init__(self, el):
        if isinstance(el, str):
            self.str = el
            self.array = np.array(el.split("±")).astype(float)
        elif isinstance(el, np.ndarray):
            self.str = "±".join(el.astype(str))
            self.array = el
    
    def __str__(self):
        return f"{round(self.array[0], 1)} ± {round(self.array[1], 1)}"
    
    def __repr__(self):
        return f"{round(self.array[0], 1)} ± {round(self.array[1], 1)}"
    
    def __truediv__(self, other):
        return MNSTD(self.array / other)
    
    def __mul__(self, other):
        return MNSTD(self.array * float(other))
    
    def __add__(self, other):
        return MNSTD(self.array + other)
    
    def __sub__(self, other):
        return MNSTD(self.array - other)

def human_normalize(df):
    ndf = df.copy()
    human = ndf.pop("Human")
    random = ndf.pop("Random")
    for col in ndf.columns:
        if col == "Game": # index
            continue
        ndf[col] = ndf[col].apply(lambda x: MNSTD(x))
        ndf[col] = (ndf[col]-random)/(human-random)*100
    return ndf


df = pd.read_csv("scores_ori.csv")
print(human_normalize(df))
