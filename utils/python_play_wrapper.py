
import numpy as np
class PythonFunctionWrapper:
    """
    (Temporary)Wrapper for a Python function that can be used as a gym environment.
    """
    def __init__(self, play_function):
        self.play_function = play_function

    def get_oblique_data(self, obs):
        """
        Get oblique data from observations.
        
        Args:
            obs: Input observation array, can be 1D or 2D
            
        Returns:
            Concatenated array of original features and their pairwise differences.
            For n features, returns n original features plus n*(n-1)/2 difference features.
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        n_features = obs.shape[1]
        
        indices = np.tril_indices(n_features, k=-1)
        
        a_mat = np.tile(obs[:, np.newaxis, :], (1, n_features, 1))
        b_mat = np.transpose(a_mat, axes=(0, 2, 1))
        
        diffs = a_mat - b_mat
        result = diffs[:, indices[0], indices[1]]
        
        return np.hstack((obs, result))
        
    def predict(self, obs, deterministic=True):
        state = self.get_oblique_data(obs)[0]
        return np.array([self.play_function(state)]), None