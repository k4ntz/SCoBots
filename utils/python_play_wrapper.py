import numpy as np
import os
import importlib
import yaml
from utils.feature_utils import mask_features, auto_generate_mask

class PythonFunctionWrapper:
    """
    (Temporary) Wrapper for a Python function that can be used as a gym environment.
    """
    def __init__(self, file_path, ff_file=None, feature_descriptions=None):
        self.file_path = file_path
        self.play_function = None
        self._ff_file = ff_file
        self._mask_indices = None
        self._feature_descriptions = feature_descriptions
        self.load_function()
        self._setup_masking()
        
    def _setup_masking(self):
        """set the feature mask, if the configuration file does not exist or the FEATURE_MASK field is not found, auto generate the mask"""
        try:
            if self._ff_file:
                with open(self._ff_file, 'r') as f:
                    config = yaml.safe_load(f)
                self._mask_indices = config.get('FEATURE_MASK', {}).get('keep_indices', None)
        except (FileNotFoundError, yaml.YAMLError):
            print(f"Failed to load feature mask from {self._ff_file}")
            
        # if no valid mask_indices is found, auto generate the mask
        if self._mask_indices is None and self._feature_descriptions is not None:
            try:
                self._mask_indices = auto_generate_mask(self._feature_descriptions)
                print("Generated feature mask automatically")
            except Exception as e:
                print(f"Failed to generate feature mask: {e}")
        
    def predict(self, obs, deterministic=True):
        obs = mask_features(obs, self._mask_indices)
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