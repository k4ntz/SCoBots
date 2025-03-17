import numpy as np

def mask_features(S, mask_indices=None):
    """
    filter the features based on the mask indices.

    Parameters
    ----------
    S : np.ndarray or list
        the observation or feature list to be filtered
    mask_indices : list, optional
        the list of indices to keep. If None, return the original input

    Returns
    -------
    np.ndarray or list
        the filtered observation or feature list
    """
    if mask_indices is None:
        return S
        
    if isinstance(S, list):
        return [S[idx] for idx in mask_indices]
    return np.array([s[mask_indices] for s in S])

def auto_generate_mask(feature_descriptions):
    """
    auto generate the feature mask
    
    Parameters
    ----------
    feature_descriptions : list
        the feature descriptions provided by the environment
    
    Returns
    -------
    list
        the indices of the features to keep
    """
    # store the seen feature names, for deduplication
    seen_features = set()
    # store the indices of the features to keep
    keep_indices = []
    
    for i, feature in enumerate(feature_descriptions):
        # skip the features with parentheses (compound features)
        if '(' in feature:
            continue
        
        # use the full feature name (including the time suffix)
        if feature not in seen_features:
            seen_features.add(feature)
            keep_indices.append(i)
    
    return keep_indices 