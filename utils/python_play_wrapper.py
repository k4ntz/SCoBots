
import numpy as np
import os
import importlib
from utils.interpreter import mask_features

class PythonFunctionWrapper:
    """
    (Temporary) Wrapper for a Python function that can be used as a gym environment.
    """
    def __init__(self, file_path, ff_file=None):
        self.file_path = file_path
        self.play_function = None
        self._ff_file = ff_file
        self.load_function()
        
    def predict(self, obs, deterministic=True):
        obs = mask_features(S=obs, ff_file=self._ff_file)
        state = obs[0]
        return np.array([self.play_function(state)]), None
    
    def load_function(self):
        # go through the directory and find the file with the most recent modification time
        if os.path.isfile(self.file_path):
            # check if the file is a python file
            if str(self.file_path).endswith('.py'):
                module_name = os.path.splitext(os.path.basename(self.file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, self.file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.play_function = module.play
                print("Loaded function from " + str(module_name+".py"))
            else:
                raise ValueError("The file is not a python file")
        else:
            file_list = [f for f in os.listdir(self.file_path) if f.endswith('.py')]
            file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.file_path, x)))
            module_name = os.path.splitext(file_list[-1])[0]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(self.file_path, module_name+".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.play_function = module.play
            print("Loaded function from " + str(module_name+".py"))